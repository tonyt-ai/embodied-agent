""" WorldState: lightweight belief state over tracked objects.

Maintains a short-lived, in-memory representation of the scene with
stable object identities, smoothed positions, velocity estimates, and
a history of observations. This state serves as the input to the
world model, enabling prediction, planning, and goal-directed control.

It is designed to work with the embodied-agent code
(detection -> world_state.update -> planner).
"""

import math
import time

# Dimensionality expected for DINO embeddings used in the state vector
DINO_STATE_DIM = 32


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors safely.

    Returns 0.0 when either vector is empty or near-zero to avoid
    division-by-zero issues.
    """
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


class WorldState:
    """Tracks visible objects, assigns stable ids, and records changes.

    Each tracked object is a dict with keys such as `id`, `label`, `x`,
    `y`, `vx`, `vy`, `first_seen`, `last_seen`, optional `embedding`, and
    `confidence`.
    """

    def __init__(self, collection_mode: bool = False):
        # Active objects keyed by stable id
        self.objects = {}
        self.next_id = 1
        # Short history of snapshots and recent change events
        self.history = []
        self.last_changes = []

        # Modes and parameters: collection mode disables smoothing and
        # shortens missing-timeouts to favor accurate dataset collection
        self.collection_mode = collection_mode
        self.max_missing_seconds = 1.2 if collection_mode else 3.0
        self.move_threshold = 0.03
        self.smoothing_alpha = 0.0 if collection_mode else 0.20

    def set_collection_mode(self, enabled: bool):
        """Toggle collection mode which affects smoothing and timeouts."""
        self.collection_mode = enabled
        self.max_missing_seconds = 1.2 if enabled else 3.0
        self.smoothing_alpha = 0.0 if enabled else 0.20

    def _dist(self, a, b):
        # Euclidean distance between two objects using x,y keys
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

    def _match_existing(self, det, max_dist=0.30):
        """Find an existing object id that best matches the detection.

        Matching requires the same label, that the object is not marked
        missing, and that the distance is within `max_dist`. A simple
        scoring function prefers closer objects and slightly favors
        higher confidence.
        """
        best_id = None
        best_score = -1e9

        for obj_id, obj in self.objects.items():
            if obj["label"] != det["label"]:
                continue
            if obj.get("missing_since") is not None:
                # currently flagged as missing
                continue

            dist = self._dist(obj, det)
            if dist > max_dist:
                continue

            # score: closer distance -> higher score; small bonus for confidence
            score = (1.0 - dist) + 0.05 * obj.get("confidence", 0.0)

            if score > best_score:
                best_score = score
                best_id = obj_id

        return best_id

    def update(self, detections):
        """Update tracker with a list of detection dicts.

        - New detections get a fresh id and an 'appeared' event.
        - Matched detections are smoothed (unless smoothing disabled),
          velocity is computed, and a 'moved' event is recorded when
          displacement exceeds `move_threshold`.
        - Objects not observed in this update get `missing_since` set
          and are removed after `max_missing_seconds`.
        """
        now = time.time()
        updated_ids = set()

        for det in detections:
            matched_id = self._match_existing(det)

            if matched_id is None:
                # New object: assign stable id and initialize metadata
                matched_id = f"obj_{self.next_id}"
                self.next_id += 1

                det = det.copy()
                det["id"] = matched_id
                det["first_seen"] = now
                det["last_seen"] = now
                det["vx"] = 0.0
                det["vy"] = 0.0
                det["missing_since"] = None
                det["embedding"] = det.get("embedding", [0.0] * DINO_STATE_DIM)

                self.objects[matched_id] = det
                self.last_changes.append({
                    "type": "appeared",
                    "label": det["label"],
                    "id": matched_id,
                    "time": now,
                })
            else:
                # Existing object: update timestamps and optionally smooth
                prev = self.objects[matched_id]
                det = det.copy()
                det["id"] = matched_id
                det["first_seen"] = prev["first_seen"]
                det["last_seen"] = now
                det["missing_since"] = None

                alpha = self.smoothing_alpha
                if alpha <= 0.0:
                    # No smoothing: take raw detection
                    x = det["x"]
                    y = det["y"]
                else:
                    # Exponential smoothing between previous and new position
                    x = alpha * prev["x"] + (1 - alpha) * det["x"]
                    y = alpha * prev["y"] + (1 - alpha) * det["y"]

                dx = x - prev["x"]
                dy = y - prev["y"]

                det["x"] = round(x, 3)
                det["y"] = round(y, 3)
                det["vx"] = round(dx, 3)
                det["vy"] = round(dy, 3)
                det["embedding"] = det.get("embedding", prev.get("embedding", [0.0] * DINO_STATE_DIM))

                moved = math.sqrt(dx * dx + dy * dy)
                if moved > self.move_threshold:
                    # Record a concise moved event for downstream consumers
                    self.last_changes.append({
                        "type": "moved",
                        "label": det["label"],
                        "id": matched_id,
                        "from": [round(prev["x"], 3), round(prev["y"], 3)],
                        "to": [round(det["x"], 3), round(det["y"], 3)],
                        "time": now,
                    })

                self.objects[matched_id] = det

            updated_ids.add(matched_id)

        # Mark objects that were not updated as missing and remove timed-out ones
        to_delete = []
        for obj_id, obj in self.objects.items():
            if obj_id in updated_ids:
                continue

            if obj.get("missing_since") is None:
                obj["missing_since"] = now

            if now - obj["missing_since"] > self.max_missing_seconds:
                to_delete.append(obj_id)

        for obj_id in to_delete:
            old_obj = self.objects[obj_id]
            self.last_changes.append({
                "type": "disappeared",
                "label": old_obj["label"],
                "id": obj_id,
                "time": now,
            })
            del self.objects[obj_id]

        # Append a compact snapshot for debugging/visualization and keep history small
        snapshot = {
            "time": now,
            "objects": [obj.copy() for obj in self.objects.values()]
        }
        self.history.append(snapshot)
        self.history = self.history[-30:]
        self.last_changes = self.last_changes[-20:]

    def find_by_label(self, label):
        """Return currently visible objects with given label sorted by recency.

        Objects marked as missing are excluded. Sorting prefers more
        recently seen and higher-confidence objects.
        """
        matches = [
            o for o in self.objects.values()
            if o["label"] == label and o.get("missing_since") is None
        ]
        matches.sort(
            key=lambda x: (
                x.get("last_seen", 0.0),
                x.get("confidence", 0.0),
            ),
            reverse=True,
        )
        return matches

    def get_recent_changes(self):
        """Return the most recent change events (appeared/moved/disappeared)."""
        return self.last_changes[-10:]

    def get_state_vector(self):
        """Produce a compact state vector for the highest-priority `cup`.

        The returned vector has layout: [x, y, vx, vy, *embedding] where
        the embedding is renormalized to unit length and padded/truncated
        to `DINO_STATE_DIM`. If no cup is present a zero vector is
        returned.
        """
        matches = self.find_by_label("cup")
        if not matches:
            return [0.0] * (4 + DINO_STATE_DIM)

        obj = matches[0]
        x = obj["x"]
        y = obj["y"]
        vx = obj.get("vx", 0.0)
        vy = obj.get("vy", 0.0)

        emb = obj.get("embedding", [0.0] * DINO_STATE_DIM)
        if len(emb) < DINO_STATE_DIM:
            emb = emb + [0.0] * (DINO_STATE_DIM - len(emb))
        else:
            emb = emb[:DINO_STATE_DIM]

        # renormalize embedding to unit length (avoid division by zero)
        norm = math.sqrt(sum(e * e for e in emb)) + 1e-6
        emb = [e / norm for e in emb]

        return [x, y, vx, vy, *emb]

    def export_objects(self):
        """Return visible objects sorted with cups first for UI convenience."""
        objects = [
            obj for obj in self.objects.values()
            if obj.get("missing_since") is None
        ]
        objects.sort(
            key=lambda x: (
                x.get("label") == "cup",
                x.get("last_seen", 0.0),
                x.get("confidence", 0.0),
            ),
            reverse=True,
        )
        return objects