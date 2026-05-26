"""
Grounded scene memory for persistent object and place semantics.

This is intentionally not a synonym table. It stores entities observed in the
scene, keeps visual/latent prototypes, and retrieves the most likely persistent
entity when the current detector label is noisy or missing.
"""

import math
from semantic_labels import normalize_label


def _as_vector(value, limit=None):
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value[:limit] if limit else value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _cosine(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    na = math.sqrt(sum(float(a[i]) * float(a[i]) for i in range(n)))
    nb = math.sqrt(sum(float(b[i]) * float(b[i]) for i in range(n)))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _blend(old, new, alpha):
    if not new:
        return old or []
    if not old:
        return list(new)
    n = min(len(old), len(new))
    out = [(1.0 - alpha) * float(old[i]) + alpha * float(new[i]) for i in range(n)]
    if len(old) > n:
        out.extend(old[n:])
    elif len(new) > n:
        out.extend(new[n:])
    return out


def _distance3(a, b):
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) < 3 or len(b) < 3:
        return None
    try:
        return math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )
    except (TypeError, ValueError):
        return None


class SceneMemory:
    """Persistent entity memory for a known physical scene."""

    def __init__(self, profile="", support_labels=None, deny_labels=None):
        self.profile = str(profile or "").strip().lower()
        self.support_labels = {normalize_label(x) for x in (support_labels or set()) if normalize_label(x)}
        self.deny_labels = {normalize_label(x) for x in (deny_labels or set()) if normalize_label(x)}
        self.entities = {}
        self.next_id = 1
        self.match_threshold = 0.42
        self.create_threshold = 0.32
        self.max_3d_dist_m = 0.55

    def _entity_kind(self, obj):
        label = normalize_label((obj or {}).get("label", ""))
        if label in self.support_labels:
            return "support"
        if label in self.deny_labels:
            return "ignored"
        return "object"

    def _semantic_observation(self, obj):
        obj = obj or {}
        visual = str(obj.get("visual_identity_class") or "").strip().lower()
        if visual == "baby_bottle":
            return "baby bottle", 0.95, "visual_identity"
        if visual == "toy_giraffe":
            return "toy giraffe", 0.95, "visual_identity"

        label = normalize_label(obj.get("label", ""))
        if not label or label in {"unknown", "unknown_seg", "unknown seg"}:
            return "", 0.0, ""
        confidence = float(obj.get("confidence", 0.5) or 0.5)
        if label in self.support_labels:
            return label, min(0.85, max(0.35, confidence)), "support_detector"
        if label in self.deny_labels:
            return "", 0.0, ""
        return label, min(0.75, max(0.20, confidence)), "detector"

    def _best_label(self, entity):
        labels = entity.get("label_scores", {})
        if not labels:
            return ""
        return max(labels.items(), key=lambda item: (float(item[1]), item[0]))[0]

    def _match_score(self, entity, obj, kind):
        if entity.get("kind") != kind:
            return -1.0
        score = 0.0
        weights = 0.0

        dino = _as_vector(obj.get("dino_embedding") or obj.get("embedding"), 96)
        if dino and entity.get("dino_embedding"):
            sim = max(0.0, _cosine(dino, entity.get("dino_embedding")))
            score += 0.46 * sim
            weights += 0.46

        jepa = _as_vector(obj.get("jepa_temporal_embedding") or obj.get("jepa_embedding"), 96)
        if jepa and entity.get("jepa_embedding"):
            sim = max(0.0, _cosine(jepa, entity.get("jepa_embedding")))
            score += 0.26 * sim
            weights += 0.26

        dist = _distance3(obj.get("position_3d"), entity.get("position_3d"))
        if dist is not None and math.isfinite(dist):
            pos_score = max(0.0, 1.0 - min(1.0, dist / max(self.max_3d_dist_m, 1e-6)))
            score += 0.20 * pos_score
            weights += 0.20

        label, label_conf, _ = self._semantic_observation(obj)
        best = self._best_label(entity)
        if label and best:
            label_score = 1.0 if label == best else 0.0
            score += 0.12 * label_score * max(0.25, label_conf)
            weights += 0.12

        support = normalize_label(obj.get("support_target_label", ""))
        if support and entity.get("last_support_label"):
            support_score = 1.0 if support == entity.get("last_support_label") else 0.25
            score += 0.06 * support_score
            weights += 0.06

        if weights <= 1e-6:
            return -1.0
        return score / weights

    def _new_entity(self, kind, now):
        memory_id = f"scene_{self.next_id}"
        self.next_id += 1
        self.entities[memory_id] = {
            "id": memory_id,
            "kind": kind,
            "label_scores": {},
            "label_sources": {},
            "dino_embedding": [],
            "jepa_embedding": [],
            "position_3d": [],
            "bbox": [],
            "last_support_label": "",
            "first_seen": float(now),
            "last_seen": float(now),
            "hits": 0,
        }
        return memory_id

    def update_object(self, obj, now):
        if not isinstance(obj, dict):
            return None
        kind = self._entity_kind(obj)
        if kind == "ignored":
            return None

        explicit_id = obj.get("scene_memory_id")
        best_id = explicit_id if explicit_id in self.entities else None
        best_score = 1.0 if best_id else -1.0
        if best_id is None:
            for memory_id, entity in self.entities.items():
                score = self._match_score(entity, obj, kind)
                if score > best_score:
                    best_score = score
                    best_id = memory_id

        if best_id is None or best_score < self.match_threshold:
            best_id = self._new_entity(kind, now)
            best_score = max(best_score, self.create_threshold)

        entity = self.entities[best_id]
        entity["last_seen"] = float(now)
        entity["hits"] = int(entity.get("hits", 0)) + 1

        label, label_conf, source = self._semantic_observation(obj)
        if label:
            labels = entity.setdefault("label_scores", {})
            labels[label] = float(labels.get(label, 0.0)) + max(0.05, label_conf)
            if source:
                entity.setdefault("label_sources", {})[label] = source

        dino = _as_vector(obj.get("dino_embedding") or obj.get("embedding"), 96)
        if dino:
            entity["dino_embedding"] = _blend(entity.get("dino_embedding", []), dino, 0.20)

        jepa = _as_vector(obj.get("jepa_temporal_embedding") or obj.get("jepa_embedding"), 96)
        if jepa:
            entity["jepa_embedding"] = _blend(entity.get("jepa_embedding", []), jepa, 0.18)

        pos = _as_vector(obj.get("position_3d"), 3)
        if len(pos) >= 3:
            entity["position_3d"] = _blend(entity.get("position_3d", []), pos[:3], 0.18)

        bbox = _as_vector(obj.get("bbox"), 4)
        if len(bbox) >= 4:
            entity["bbox"] = _blend(entity.get("bbox", []), bbox[:4], 0.25)

        support = normalize_label(obj.get("support_target_label", ""))
        if support:
            entity["last_support_label"] = support

        best_label = self._best_label(entity)
        total_score = sum(float(v) for v in entity.get("label_scores", {}).values())
        label_score = 0.0
        if best_label and total_score > 1e-6:
            label_score = float(entity["label_scores"].get(best_label, 0.0)) / total_score

        obj["scene_memory_id"] = best_id
        obj["scene_memory_label"] = best_label
        obj["scene_memory_score"] = round(max(best_score, label_score), 4)
        obj["scene_memory_kind"] = kind
        if kind == "object" and best_label and obj["scene_memory_score"] >= 0.45:
            obj["semantic_label"] = best_label
        return entity

    def export(self, limit=24):
        rows = []
        for entity in self.entities.values():
            best_label = self._best_label(entity)
            total_score = sum(float(v) for v in entity.get("label_scores", {}).values())
            label_score = 0.0
            if best_label and total_score > 1e-6:
                label_score = float(entity["label_scores"].get(best_label, 0.0)) / total_score
            rows.append({
                "id": entity.get("id"),
                "kind": entity.get("kind"),
                "label": best_label,
                "label_score": round(label_score, 4),
                "label_scores": {
                    label: round(float(score), 3)
                    for label, score in sorted(entity.get("label_scores", {}).items(), key=lambda item: -float(item[1]))[:4]
                },
                "position_3d": [round(float(v), 4) for v in entity.get("position_3d", [])[:3]],
                "bbox": [round(float(v), 4) for v in entity.get("bbox", [])[:4]],
                "last_support_label": entity.get("last_support_label", ""),
                "hits": int(entity.get("hits", 0)),
                "last_seen": float(entity.get("last_seen", 0.0)),
            })
        rows.sort(key=lambda item: (-int(item.get("hits", 0)), item.get("id", "")))
        return rows[:limit]
