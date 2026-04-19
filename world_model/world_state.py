""" WorldState: lightweight belief state over tracked objects.

Maintains a short-lived, in-memory representation of the scene with
stable object identities, smoothed positions, velocity estimates, and
a history of observations. This state serves as the input to the
world model, enabling prediction, planning, and goal-directed control.

It is designed to work with the embodied-agent code
detection -> world_state.update -> planner.
"""

import math
import time

DINO_STATE_DIM = 32


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


class WorldState:
    """Tracks visible objects, assigns stable ids, and records changes."""

    def __init__(self, collection_mode: bool = False):
        self.objects = {}
        self.next_id = 1
        self.history = []
        self.last_changes = []
        self.collection_mode = collection_mode
        self.max_missing_seconds = 1.2 if collection_mode else 3.0
        self.move_threshold = 0.03
        self.smoothing_alpha = 0.0 if collection_mode else 0.20

        self.camera_pose = None
        self.objects_3d = []
        self.hands = []
        self.world_debug = {}
        self.sparse_map = []

    def set_collection_mode(self, enabled: bool):
        self.collection_mode = enabled
        self.max_missing_seconds = 1.2 if enabled else 3.0
        self.smoothing_alpha = 0.0 if enabled else 0.20

    def _dist(self, a, b):
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

    def _match_existing(self, det, max_dist=0.30):
        best_id = None
        best_score = -1e9

        for obj_id, obj in self.objects.items():
            if obj["label"] != det["label"]:
                continue
            if obj.get("missing_since") is not None:
                continue

            dist = self._dist(obj, det)
            if dist > max_dist:
                continue

            score = (1.0 - dist) + 0.05 * obj.get("confidence", 0.0)
            if score > best_score:
                best_score = score
                best_id = obj_id

        return best_id

    def update(self, detections, camera_pose=None, hands=None, world_debug=None, sparse_map=None):
        now = time.time()
        updated_ids = set()
        self.camera_pose = camera_pose or self.camera_pose
        self.hands = hands or []
        self.world_debug = world_debug or {}
        self.sparse_map = sparse_map or []

        for det in detections:
            matched_id = self._match_existing(det)

            if matched_id is None:
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
                det["position_3d"] = det.get("position_3d", [0.0, 0.0, 0.0])
                det["position_camera_3d"] = det.get("position_camera_3d", det["position_3d"])
                det["velocity_3d"] = det.get("velocity_3d", [0.0, 0.0, 0.0])
                det["depth"] = det.get("depth", 0.0)
                det["depth_confidence"] = det.get("depth_confidence", 0.0)
                det["landmark_support"] = det.get("landmark_support", 0)
                det["landmark_blend_weight"] = det.get("landmark_blend_weight", 0.0)

                self.objects[matched_id] = det
                self.last_changes.append({
                    "type": "appeared",
                    "label": det["label"],
                    "id": matched_id,
                    "time": now,
                })
            else:
                prev = self.objects[matched_id]
                det = det.copy()
                det["id"] = matched_id
                det["first_seen"] = prev["first_seen"]
                det["last_seen"] = now
                det["missing_since"] = None

                alpha = self.smoothing_alpha
                if alpha <= 0.0:
                    x = det["x"]
                    y = det["y"]
                else:
                    x = alpha * prev["x"] + (1 - alpha) * det["x"]
                    y = alpha * prev["y"] + (1 - alpha) * det["y"]

                dx = x - prev["x"]
                dy = y - prev["y"]

                det["x"] = round(x, 3)
                det["y"] = round(y, 3)
                det["vx"] = round(dx, 3)
                det["vy"] = round(dy, 3)
                det["embedding"] = det.get("embedding", prev.get("embedding", [0.0] * DINO_STATE_DIM))

                prev_pos_3d = prev.get("position_3d", [0.0, 0.0, 0.0])
                curr_pos_3d = det.get("position_3d", prev_pos_3d)
                det["position_3d"] = curr_pos_3d
                det["position_camera_3d"] = det.get(
                    "position_camera_3d",
                    prev.get("position_camera_3d", curr_pos_3d),
                )
                det["velocity_3d"] = [
                    round(curr_pos_3d[i] - prev_pos_3d[i], 4)
                    for i in range(3)
                ]
                det["depth"] = det.get("depth", prev.get("depth", 0.0))
                det["depth_confidence"] = det.get(
                    "depth_confidence",
                    prev.get("depth_confidence", 0.0),
                )
                det["landmark_support"] = det.get(
                    "landmark_support",
                    prev.get("landmark_support", 0),
                )
                det["landmark_blend_weight"] = det.get(
                    "landmark_blend_weight",
                    prev.get("landmark_blend_weight", 0.0),
                )

                moved = math.sqrt(dx * dx + dy * dy)
                if moved > self.move_threshold:
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

        snapshot = {
            "time": now,
            "objects": [obj.copy() for obj in self.objects.values()],
            "camera_pose": self.camera_pose,
            "hands": self.hands,
            "sparse_map": self.sparse_map,
        }
        self.history.append(snapshot)
        self.history = self.history[-30:]
        self.last_changes = self.last_changes[-20:]
        self.objects_3d = [
            {
                "id": obj["id"],
                "label": obj["label"],
                "position_3d": obj.get("position_3d", [0.0, 0.0, 0.0]),
                "position_camera_3d": obj.get("position_camera_3d", [0.0, 0.0, 0.0]),
                "velocity_3d": obj.get("velocity_3d", [0.0, 0.0, 0.0]),
                "depth": obj.get("depth", 0.0),
                "depth_confidence": obj.get("depth_confidence", 0.0),
                "landmark_support": obj.get("landmark_support", 0),
                "landmark_blend_weight": obj.get("landmark_blend_weight", 0.0),
            }
            for obj in self.export_objects()
        ]

    def find_by_label(self, label):
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
        return self.last_changes[-10:]

    def get_state_vector(self):
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

        norm = math.sqrt(sum(e * e for e in emb)) + 1e-6
        emb = [e / norm for e in emb]

        return [x, y, vx, vy, *emb]

    def export_objects(self):
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

    def export_objects_3d(self):
        return self.objects_3d

    def export_world_state(self):
        return {
            "camera_pose": self.camera_pose,
            "objects_3d": self.export_objects_3d(),
            "hands": self.hands,
            "world_debug": self.world_debug,
            "sparse_map": self.sparse_map,
        }
