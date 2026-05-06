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
import os
import numpy as np
from jepa_interaction_head import interaction_score as jepa_interaction_score
from semantic_labels import normalize_label
from temporal_interaction_head import (
    TemporalInteractionPredictor,
    build_feature_vector,
)

DINO_STATE_DIM = 32


def _parse_label_float_map(raw: str):
    out = {}
    for part in str(raw or "").split(","):
        if ":" not in part:
            continue
        label, value = part.split(":", 1)
        label = normalize_label(label)
        try:
            out[label] = float(value)
        except (TypeError, ValueError):
            continue
    return out


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
        self.object_memory_seconds = float(os.environ.get("WORLD_OBJECT_MEMORY_SECONDS", "30.0"))
        self.move_threshold = 0.03
        self.smoothing_alpha = 0.0 if collection_mode else 0.20
        self.scene_profile = os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower()
        if self.scene_profile == "sophie":
            os.environ.setdefault("HAND_LABEL_CONTACT_ENTER_FRAMES", "bottle:1,donut:1,mouse:1,toy:1")
            os.environ.setdefault("HAND_LABEL_TOUCH_DISTANCE_M", "bottle:0.12,donut:0.12,mouse:0.11,toy:0.11")
            os.environ.setdefault("HAND_LABEL_TOUCH_START_DISTANCE_M", "bottle:0.13,donut:0.13,mouse:0.12,toy:0.12")
            os.environ.setdefault("HAND_LABEL_TOUCH_END_DISTANCE_M", "bottle:0.16,donut:0.16,mouse:0.15,toy:0.15")
            os.environ.setdefault("STATIC_TARGET_LABELS", "tray,mat,black mat,table mat,placemat,dish,plate,unknown_seg")
            os.environ.setdefault("STATIC_TARGET_INFER_LARGE_LABEL", "tray")
            os.environ.setdefault("STATIC_TARGET_INFER_SMALL_LABEL", "")
            os.environ.setdefault("STATIC_TARGET_INFER_DARK_LABEL", "mat")
            os.environ.setdefault("STATIC_TARGET_DARK_LUMA_MAX", "85")
            os.environ.setdefault("STATIC_TARGET_LOCK_UNKNOWN", "0")
            os.environ.setdefault("STATIC_TARGET_HITS_MIN", "2")

        self.camera_pose = None
        self.objects_3d = []
        self.hands = []
        self.world_debug = {}
        self.sparse_map = []
        self.hand_tracks = {}
        self.hand_object_interactions = []
        self.hand_contact_events = []
        self.hand_trajectories = {}
        self.hand_trail_max_points = int(os.environ.get("HAND_TRAIL_MAX_POINTS", "40"))
        self.manipulation_active = {}
        self.learned_manipulation_active = {}
        self.learned_manipulation_events = []
        self.learned_contact_threshold = float(os.environ.get("JEPA_EVENT_CONTACT_THRESHOLD", "0.55"))
        self.learned_release_threshold = float(os.environ.get("JEPA_EVENT_RELEASE_THRESHOLD", "0.75"))
        self.learned_release_ui_threshold = float(os.environ.get("JEPA_EVENT_RELEASE_UI_THRESHOLD", "0.55"))
        self.learned_min_hold_s = float(os.environ.get("JEPA_EVENT_MIN_HOLD_S", "1.2"))
        self.learned_release_linger_s = float(os.environ.get("JEPA_EVENT_RELEASE_LINGER_S", "2.2"))
        self.manipulation_events = []
        self.manipulation_min_move_distance_m = float(os.environ.get("MANIP_MIN_MOVE_DISTANCE_M", "0.05"))
        self.manip_relation_near_xy_m = float(os.environ.get("MANIP_REL_NEAR_XY_M", "0.12"))
        self.manip_relation_behind_dz_m = float(os.environ.get("MANIP_REL_BEHIND_DZ_M", "0.05"))
        self.manip_relation_near_3d_m = float(os.environ.get("MANIP_REL_NEAR_3D_M", "0.18"))
        self.hand_contact_distance_m = float(os.environ.get("HAND_CONTACT_DISTANCE_M", "0.09"))
        self.hand_near_distance_m = float(os.environ.get("HAND_NEAR_DISTANCE_M", "0.16"))
        self.hand_contact_2d_distance_norm = float(os.environ.get("HAND_CONTACT_2D_DISTANCE_NORM", "0.08"))
        self.hand_contact_2d_overlap_min = float(os.environ.get("HAND_CONTACT_2D_OVERLAP_MIN", "0.025"))
        self.hand_contact_2d_effective_distance_m = float(os.environ.get("HAND_CONTACT_2D_EFFECTIVE_DISTANCE_M", "0.075"))
        self.hand_contact_enter_frames = int(os.environ.get("HAND_CONTACT_ENTER_FRAMES", "2"))
        self.hand_contact_exit_frames = int(os.environ.get("HAND_CONTACT_EXIT_FRAMES", "3"))
        self.label_contact_enter_frames = {
            label: max(1, int(value))
            for label, value in _parse_label_float_map(
                os.environ.get("HAND_LABEL_CONTACT_ENTER_FRAMES", "apple:1,banana:1,orange:1")
            ).items()
        }
        self.hand_capsule_contact_padding_m = float(os.environ.get("HAND_CAPSULE_CONTACT_PADDING_M", "0.015"))
        self.hand_touch_distance_m = float(os.environ.get("HAND_TOUCH_DISTANCE_M", "0.055"))
        self.hand_touch_start_distance_m = float(os.environ.get("HAND_TOUCH_START_DISTANCE_M", "0.065"))
        self.hand_touch_end_distance_m = float(os.environ.get("HAND_TOUCH_END_DISTANCE_M", "0.095"))
        self.label_touch_distance_m = _parse_label_float_map(
            os.environ.get("HAND_LABEL_TOUCH_DISTANCE_M", "apple:0.08,banana:0.08,orange:0.08")
        )
        self.label_touch_start_distance_m = _parse_label_float_map(
            os.environ.get("HAND_LABEL_TOUCH_START_DISTANCE_M", "apple:0.09,banana:0.09,orange:0.09")
        )
        self.label_touch_end_distance_m = _parse_label_float_map(
            os.environ.get("HAND_LABEL_TOUCH_END_DISTANCE_M", "apple:0.12,banana:0.12,orange:0.12")
        )
        self.require_3d_for_contact_start = os.environ.get("HAND_REQUIRE_3D_CONTACT_START", "1").lower() in {"1", "true", "yes"}
        self.match_max_dist = float(os.environ.get("WORLD_MATCH_MAX_DIST", "0.34"))
        self.match_confidence_weight = float(os.environ.get("WORLD_MATCH_CONFIDENCE_WEIGHT", "0.05"))
        self.match_embedding_weight = float(os.environ.get("WORLD_MATCH_EMBEDDING_WEIGHT", "0.55"))
        self.match_jepa_weight = float(os.environ.get("WORLD_MATCH_JEPA_WEIGHT", "0.18"))
        self.dino_embedding_ema_alpha = float(os.environ.get("WORLD_DINO_EMBEDDING_EMA_ALPHA", "0.28"))
        self.match_3d_weight = float(os.environ.get("WORLD_MATCH_3D_WEIGHT", "0.30"))
        self.match_max_3d_dist = float(os.environ.get("WORLD_MATCH_MAX_3D_DIST", "0.45"))
        self.open_vocab_reid = os.environ.get("WORLD_OPEN_VOCAB_REID", "1").lower() in {"1", "true", "yes"}
        self.hidden_reid_max_dist = float(os.environ.get("WORLD_HIDDEN_REID_MAX_DIST", "0.58"))
        self.hidden_reid_min_score = float(os.environ.get("WORLD_HIDDEN_REID_MIN_SCORE", "0.42"))
        self.interaction_label_denylist = {
            item.strip().lower()
            for item in os.environ.get(
                "HAND_INTERACTION_LABEL_DENYLIST",
                "person,bed,couch,chair,dining table,tv,potted plant,sports ball,unknown,unknown_seg,unknown seg,coaster,dish,plate,platter,tray,mat,black mat,table mat,placemat",
            ).split(",")
            if item.strip()
        }
        self.object_surface_points_max = int(os.environ.get("OBJECT_SURFACE_POINTS_MAX", "140"))
        self.jepa_use_for_contact = os.environ.get("JEPA_USE_FOR_CONTACT", "1").lower() in {"1", "true", "yes"}
        self.jepa_distance_reweight = float(os.environ.get("JEPA_DISTANCE_REWEIGHT", "0.25"))
        self.jepa_contact_score_threshold = float(os.environ.get("JEPA_CONTACT_SCORE_THRESHOLD", "0.58"))
        self.jepa_history_len = int(os.environ.get("JEPA_HISTORY_LEN", "12"))
        self.jepa_temporal_decay = float(os.environ.get("JEPA_TEMPORAL_DECAY", "0.82"))
        self.jepa_min_hist_for_temporal = int(os.environ.get("JEPA_MIN_HIST_FOR_TEMPORAL", "3"))
        self.temporal_head = TemporalInteractionPredictor()
        self.static_targets = {}
        self.static_target_hits_min = int(os.environ.get("STATIC_TARGET_HITS_MIN", "3"))
        self.static_target_vel_max = float(os.environ.get("STATIC_TARGET_VEL_MAX", "0.08"))
        self.static_target_infer_large_label = normalize_label(os.environ.get("STATIC_TARGET_INFER_LARGE_LABEL", "dish"))
        self.static_target_infer_small_label = normalize_label(os.environ.get("STATIC_TARGET_INFER_SMALL_LABEL", "coaster"))
        self.static_target_infer_dark_label = normalize_label(os.environ.get("STATIC_TARGET_INFER_DARK_LABEL", ""))
        self.static_target_dark_luma_max = float(os.environ.get("STATIC_TARGET_DARK_LUMA_MAX", "80.0"))
        self.static_target_lock_unknown = os.environ.get("STATIC_TARGET_LOCK_UNKNOWN", "1").lower() in {"1", "true", "yes"}
        self.static_target_labels = {normalize_label(item) for item in os.environ.get("STATIC_TARGET_LABELS", "coaster,dish,plate,platter,cake stand,tray,mat,black mat,table mat,placemat,unknown_seg").split(",") if item.strip()}
        self.static_target_exclude_labels = {
            normalize_label(item)
            for item in os.environ.get(
                "STATIC_TARGET_EXCLUDE_LABELS",
                "cup,mug,bottle,baby bottle,toy giraffe,apple,banana,orange,person,hand,cell phone,book,mouse,cake",
            ).split(",")
            if item.strip()
        }
        self.hand_interaction_sides = {
            item.strip().lower()
            for item in os.environ.get("HAND_INTERACTION_SIDES", "right").split(",")
            if item.strip()
        }
        self.static_target_match_dist_m = float(os.environ.get("STATIC_TARGET_MATCH_DIST_M", "0.18"))
        self.static_target_min_area = float(os.environ.get("STATIC_TARGET_MIN_BBOX_AREA", "0.003"))
        self.static_target_max_area = float(os.environ.get("STATIC_TARGET_MAX_BBOX_AREA", "0.42"))
        self.static_target_embed_sim_min = float(os.environ.get("STATIC_TARGET_EMBED_SIM_MIN", "0.08"))
        self.static_target_persist_locked = os.environ.get("STATIC_TARGET_PERSIST_LOCKED", "1").lower() in {"1", "true", "yes"}
        self.static_target_export_max = int(os.environ.get("STATIC_TARGET_EXPORT_MAX", "64"))
        self.label_refinements = {}

    def _normalized_label(self, label):
        return normalize_label(label)

    def _touch_threshold(self, label, kind: str):
        norm = self._normalized_label(label)
        threshold_label = norm
        if norm == "baby bottle":
            threshold_label = "bottle"
        elif norm in {"toy giraffe", "giraffe toy", "sophie", "sophie giraffe"}:
            threshold_label = "toy"
        if kind == "distance":
            return float(self.label_touch_distance_m.get(threshold_label, self.hand_touch_distance_m))
        if kind == "start":
            return float(self.label_touch_start_distance_m.get(threshold_label, self.hand_touch_start_distance_m))
        if kind == "end":
            return float(self.label_touch_end_distance_m.get(threshold_label, self.hand_touch_end_distance_m))
        return float(self.hand_touch_distance_m)

    def _contact_enter_frames(self, label):
        norm = self._normalized_label(label)
        threshold_label = "bottle" if norm == "baby bottle" else ("toy" if norm in {"toy giraffe", "giraffe toy", "sophie", "sophie giraffe"} else norm)
        return int(self.label_contact_enter_frames.get(threshold_label, self.hand_contact_enter_frames))

    def set_collection_mode(self, enabled: bool):
        self.collection_mode = enabled
        self.max_missing_seconds = 1.2 if enabled else 3.0
        self.smoothing_alpha = 0.0 if enabled else 0.20

    def _dist(self, a, b):
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

    def _normalize_embedding(self, emb):
        if not isinstance(emb, list) or not emb:
            return []
        vals = []
        for v in emb:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                vals.append(fv)
        if not vals:
            return []
        n = math.sqrt(sum(v * v for v in vals))
        if n < 1e-8:
            return []
        return [v / n for v in vals]

    def _append_emb_history(self, hist, emb):
        emb_n = self._normalize_embedding(emb)
        if not emb_n:
            return list(hist) if isinstance(hist, list) else []
        out = list(hist) if isinstance(hist, list) else []
        out.append(emb_n)
        max_len = max(2, int(self.jepa_history_len))
        if len(out) > max_len:
            out = out[-max_len:]
        return out

    def _blend_embeddings(self, prev, curr, alpha: float):
        if not isinstance(prev, list) or not isinstance(curr, list):
            return curr if isinstance(curr, list) else []
        if not prev or not curr or len(prev) != len(curr):
            return curr
        try:
            blended = [
                (1.0 - float(alpha)) * float(prev[i]) + float(alpha) * float(curr[i])
                for i in range(len(curr))
            ]
        except (TypeError, ValueError):
            return curr
        return self._normalize_embedding(blended)

    def _temporal_embedding(self, hist):
        if not isinstance(hist, list) or len(hist) < max(1, int(self.jepa_min_hist_for_temporal)):
            return []
        base = None
        wsum = 0.0
        decay = min(0.999, max(0.05, float(self.jepa_temporal_decay)))
        for age, emb in enumerate(reversed(hist)):
            if not isinstance(emb, list) or not emb:
                continue
            w = decay ** age
            if base is None:
                base = [0.0 for _ in emb]
            if len(emb) != len(base):
                continue
            for i, v in enumerate(emb):
                base[i] += w * float(v)
            wsum += w
        if base is None or wsum <= 1e-8:
            return []
        avg = [v / wsum for v in base]
        return self._normalize_embedding(avg)

    def _smooth_bbox(self, prev_bbox, curr_bbox, alpha: float):
        if not (
            isinstance(prev_bbox, (list, tuple)) and len(prev_bbox) >= 4
            and isinstance(curr_bbox, (list, tuple)) and len(curr_bbox) >= 4
        ):
            return curr_bbox
        try:
            return [
                round(float(alpha) * float(prev_bbox[i]) + (1.0 - float(alpha)) * float(curr_bbox[i]), 4)
                for i in range(4)
            ]
        except (TypeError, ValueError):
            return curr_bbox

    def _labels_compatible(self, det, obj):
        det_label = self._normalized_label(det.get("label", ""))
        det_raw = self._normalized_label(det.get("raw_label", det_label))
        obj_label = self._normalized_label(obj.get("label", ""))
        obj_raw = self._normalized_label(obj.get("raw_label", obj_label))
        obj_vlm = self._normalized_label(obj.get("vlm_label", ""))
        labels = {v for v in [obj_label, obj_raw, obj_vlm] if v}
        return det_label in labels or det_raw in labels

    def _sophie_visual_class(self, item):
        if self.scene_profile != "sophie":
            return ""
        label = self._normalized_label(item.get("label", item.get("raw_label", "")))
        raw = self._normalized_label(item.get("raw_label", label))
        if label in {"toy giraffe", "toy", "donut", "mouse"} or raw in {"toy giraffe", "toy", "donut", "mouse"}:
            return "toy_giraffe"
        if label in {"baby bottle", "bottle", "cup", "mug", "vase"} or raw in {"baby bottle", "bottle", "cup", "mug", "vase"}:
            bbox = item.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    bw = max(0.0, float(bbox[2]) - float(bbox[0]))
                    bh = max(0.0, float(bbox[3]) - float(bbox[1]))
                    area = bw * bh
                    aspect = bw / max(bh, 1e-6)
                    if area >= 0.006 and aspect >= 0.72:
                        return "toy_giraffe"
                    if aspect <= 0.68:
                        return "baby_bottle"
                except (TypeError, ValueError):
                    pass
            return "baby_bottle"
        return ""

    def _sophie_visual_classes_compatible(self, det, obj):
        det_cls = self._sophie_visual_class(det)
        obj_cls = self._sophie_visual_class(obj)
        return not det_cls or not obj_cls or det_cls == obj_cls

    def _interaction_display_label(self, item, fallback="object"):
        visual_class = self._sophie_visual_class(item or {})
        if visual_class == "baby_bottle":
            return "baby bottle"
        if visual_class == "toy_giraffe":
            return "toy giraffe"
        return self._normalized_label((item or {}).get("label", fallback) or fallback) or fallback

    def _is_trackable_object_label(self, label):
        norm = self._normalized_label(label)
        return bool(norm) and norm not in self.static_target_labels and norm not in self.interaction_label_denylist

    def _match_existing(self, det, max_dist=None):
        max_dist = self.match_max_dist if max_dist is None else float(max_dist)
        best_id = None
        best_score = -1e9
        det_emb = det.get("embedding")
        det_emb_source = str(det.get("embedding_source", "") or "").lower()
        det_pos_3d = det.get("position_3d")
        has_det_3d = isinstance(det_pos_3d, (list, tuple)) and len(det_pos_3d) >= 3

        for obj_id, obj in self.objects.items():
            labels_match = self._labels_compatible(det, obj)
            labels_are_trackable = self._is_trackable_object_label(det.get("label")) and self._is_trackable_object_label(obj.get("label"))
            if not labels_match and not (self.open_vocab_reid and labels_are_trackable):
                continue
            if not self._sophie_visual_classes_compatible(det, obj):
                continue

            is_hidden = obj.get("missing_since") is not None
            gate_dist = max_dist
            if is_hidden:
                gate_dist = max(gate_dist, self.hidden_reid_max_dist)
            dist = self._dist(obj, det)
            if dist > gate_dist:
                continue

            score = (1.0 - min(1.0, dist / max(gate_dist, 1e-6)))
            score += self.match_confidence_weight * float(obj.get("confidence", 0.0) or 0.0)
            if not labels_match:
                score -= 0.22
            if is_hidden:
                try:
                    missing_age = max(0.0, time.time() - float(obj.get("missing_since") or time.time()))
                except (TypeError, ValueError):
                    missing_age = self.object_memory_seconds
                score -= 0.16 * min(1.0, missing_age / max(self.object_memory_seconds, 1e-6))
            emb_sim = cosine_similarity(det_emb, obj.get("embedding"))
            obj_emb_source = str(obj.get("embedding_source", "") or "").lower()
            if det_emb_source and obj_emb_source and det_emb_source != obj_emb_source:
                emb_sim *= 0.20
            score += self.match_embedding_weight * emb_sim
            jepa_sim = cosine_similarity(
                det.get("jepa_embedding") or det.get("jepa_temporal_embedding"),
                obj.get("jepa_temporal_embedding") or obj.get("jepa_embedding"),
            )
            if jepa_sim > 0.0:
                score += self.match_jepa_weight * jepa_sim

            if has_det_3d:
                obj_pos_3d = obj.get("position_3d")
                if isinstance(obj_pos_3d, (list, tuple)) and len(obj_pos_3d) >= 3:
                    try:
                        d3 = math.sqrt(
                            (float(det_pos_3d[0]) - float(obj_pos_3d[0])) ** 2
                            + (float(det_pos_3d[1]) - float(obj_pos_3d[1])) ** 2
                            + (float(det_pos_3d[2]) - float(obj_pos_3d[2])) ** 2
                        )
                        if math.isfinite(d3) and d3 <= self.match_max_3d_dist:
                            score += self.match_3d_weight * (1.0 - min(1.0, d3 / max(self.match_max_3d_dist, 1e-6)))
                    except (TypeError, ValueError):
                        pass

            if is_hidden and score < self.hidden_reid_min_score:
                continue
            if score > best_score:
                best_score = score
                best_id = obj_id

        return best_id

    def update(self, detections, camera_pose=None, hands=None, world_debug=None, sparse_map=None):
        now = time.time()
        updated_ids = set()
        self.camera_pose = camera_pose or self.camera_pose
        self.hands = hands or []
        self.world_debug = (world_debug or {}).copy()
        self.sparse_map = sparse_map or []

        for det in detections:
            det = det.copy()
            raw_label = det.get("label", "")
            det["raw_label"] = det.get("raw_label", raw_label)
            det["label"] = self._normalized_label(raw_label)
            visual_class = self._sophie_visual_class(det)
            if visual_class == "toy_giraffe" and det["label"] in {"bottle", "cup", "mug", "vase"}:
                det["label"] = "toy giraffe"
            elif visual_class == "baby_bottle" and det["label"] in {"cup", "mug", "vase"}:
                det["label"] = "baby bottle"
            if visual_class:
                det["visual_identity_class"] = visual_class
            matched_id = self._match_existing(det)

            if matched_id is None:
                matched_id = f"obj_{self.next_id}"
                self.next_id += 1

                det["id"] = matched_id
                det["first_seen"] = now
                det["last_seen"] = now
                det["observation_count"] = 1
                det["vx"] = 0.0
                det["vy"] = 0.0
                det["missing_since"] = None
                det["visible"] = True
                det["embedding"] = det.get("embedding", [0.0] * DINO_STATE_DIM)
                det["embedding_source"] = det.get("embedding_source", "unknown")
                det["hsv_embedding"] = det.get("hsv_embedding", [])
                det["dino_embedding"] = det.get("dino_embedding", [])
                det["position_3d"] = det.get("position_3d", [0.0, 0.0, 0.0])
                det["position_camera_3d"] = det.get("position_camera_3d", det["position_3d"])
                det["velocity_3d"] = det.get("velocity_3d", [0.0, 0.0, 0.0])
                det["depth"] = det.get("depth", 0.0)
                det["depth_confidence"] = det.get("depth_confidence", 0.0)
                det["landmark_support"] = det.get("landmark_support", 0)
                det["landmark_blend_weight"] = det.get("landmark_blend_weight", 0.0)
                det["proxy_radius_m"] = float(det.get("proxy_radius_m", 0.06) or 0.06)
                det["proxy_extent_m"] = det.get("proxy_extent_m", [0.08, 0.08, 0.08])
                det["surface_points_3d"] = det.get("surface_points_3d", [])
                det["jepa_embedding"] = det.get("jepa_embedding", [])
                det["jepa_history"] = self._append_emb_history([], det.get("jepa_embedding", []))
                det["jepa_temporal_embedding"] = self._temporal_embedding(det["jepa_history"])

                self.objects[matched_id] = det
                self.last_changes.append({
                    "type": "appeared",
                    "label": det["label"],
                    "id": matched_id,
                    "time": now,
                })
            else:
                prev = self.objects[matched_id]
                refined_label = prev.get("label") if bool(prev.get("label_refined", False)) else None
                refined_raw = prev.get("vlm_label") if bool(prev.get("label_refined", False)) else None
                det["id"] = matched_id
                det["first_seen"] = prev["first_seen"]
                det["last_seen"] = now
                det["observation_count"] = int(prev.get("observation_count", 1) or 1) + 1
                det["missing_since"] = None
                det["visible"] = True
                if refined_label:
                    det["label"] = refined_label
                    det["vlm_label"] = refined_raw or refined_label
                    det["label_refined"] = True
                    det["label_confidence"] = prev.get("label_confidence", det.get("label_confidence", 0.0))

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
                det_source = str(det.get("embedding_source", "") or "").lower()
                prev_source = str(prev.get("embedding_source", "") or "").lower()
                if det_source == "dino" or prev_source != "dino":
                    curr_emb = det.get("embedding", prev.get("embedding", [0.0] * DINO_STATE_DIM))
                    if det_source == "dino" and prev_source == "dino":
                        curr_emb = self._blend_embeddings(
                            prev.get("embedding", []),
                            curr_emb,
                            self.dino_embedding_ema_alpha,
                        )
                    det["embedding"] = curr_emb
                    det["embedding_source"] = det_source or prev_source or "unknown"
                else:
                    # Preserve sparse DINO identity memory; HSV is retained separately.
                    det["embedding"] = prev.get("embedding", det.get("embedding", [0.0] * DINO_STATE_DIM))
                    det["embedding_source"] = prev_source or "dino"
                det["hsv_embedding"] = det.get("hsv_embedding", prev.get("hsv_embedding", []))
                if det.get("embedding_source") == "dino":
                    det["dino_embedding"] = det.get("embedding", det.get("dino_embedding", []))
                else:
                    det["dino_embedding"] = det.get("dino_embedding", prev.get("dino_embedding", []))
                det["bbox"] = self._smooth_bbox(prev.get("bbox"), det.get("bbox"), min(0.45, max(0.0, alpha)))

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
                det["proxy_radius_m"] = float(
                    det.get("proxy_radius_m", prev.get("proxy_radius_m", 0.06)) or 0.06
                )
                det["proxy_extent_m"] = det.get(
                    "proxy_extent_m",
                    prev.get("proxy_extent_m", [0.08, 0.08, 0.08]),
                )
                prev_surf = prev.get("surface_points_3d", []) if isinstance(prev.get("surface_points_3d", []), list) else []
                curr_surf = det.get("surface_points_3d", []) if isinstance(det.get("surface_points_3d", []), list) else []
                merged_surf = curr_surf + prev_surf
                if len(merged_surf) > self.object_surface_points_max:
                    merged_surf = merged_surf[: self.object_surface_points_max]
                det["surface_points_3d"] = merged_surf
                det["jepa_embedding"] = det.get("jepa_embedding", prev.get("jepa_embedding", []))
                prev_hist = prev.get("jepa_history", [])
                det["jepa_history"] = self._append_emb_history(prev_hist, det.get("jepa_embedding", []))
                det["jepa_temporal_embedding"] = self._temporal_embedding(det["jepa_history"])
                for key in ("support_relation", "support_target_id", "support_target_label", "support_updated_time"):
                    if key in prev and key not in det:
                        det[key] = prev[key]

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
            obj["visible"] = False
            if now - obj["missing_since"] > self.object_memory_seconds:
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
                "proxy_radius_m": obj.get("proxy_radius_m", 0.06),
                "proxy_extent_m": obj.get("proxy_extent_m", [0.08, 0.08, 0.08]),
                "surface_points_3d": obj.get("surface_points_3d", [])[:32] if isinstance(obj.get("surface_points_3d", []), list) else [],
            }
            for obj in self.export_objects()
        ]
        self._update_static_targets()
        self._update_hand_tracks(now)
        self._update_hand_object_interactions(now)
        self.world_debug["hand_object_interactions"] = self.hand_object_interactions
        self.world_debug["hand_contact_events"] = self.hand_contact_events[-20:]
        self.world_debug["hand_trajectories"] = self._export_hand_trajectories()
        self.world_debug["manipulation_events"] = self.manipulation_events[-20:]
        self.world_debug["manipulation_active"] = list(self.manipulation_active.values())[-8:]
        self._update_object_support_memory(now)
        self.world_debug["hands_tracked"] = len(self.hand_tracks)
        static_targets_export = self._export_static_targets()
        self.world_debug["static_targets"] = static_targets_export
        self.world_debug["static_targets_locked"] = int(sum(1 for t in self.static_targets.values() if bool(t.get("locked", False))))
        self.world_debug["static_targets_persistent"] = int(sum(1 for t in static_targets_export if bool(t.get("persistent", False))))
        object_memory_export = self.export_object_memory()
        self.world_debug["object_memory"] = object_memory_export[:12]
        self.world_debug["object_memory_count"] = int(len(object_memory_export))
        self.world_debug["object_memory_hidden_count"] = int(sum(1 for o in object_memory_export if not bool(o.get("visible", False))))
        self.world_debug["label_refinements"] = {
            "count": int(len(self.label_refinements)),
            "latest": self.label_refinements,
        }
        self.world_debug["jepa_temporal"] = {
            "enabled": bool(self.jepa_use_for_contact),
            "temporal_head_ready": bool(getattr(self.temporal_head, "ready", False)),
            "temporal_head_model": getattr(self.temporal_head, "model_path", None),
            "history_len": int(self.jepa_history_len),
            "decay": float(self.jepa_temporal_decay),
            "objects_with_temporal": int(sum(1 for o in self.objects.values() if isinstance(o.get("jepa_temporal_embedding"), list) and len(o.get("jepa_temporal_embedding", [])) > 0)),
            "hands_with_temporal": int(sum(1 for h in self.hand_tracks.values() if isinstance(h.get("jepa_temporal_embedding"), list) and len(h.get("jepa_temporal_embedding", [])) > 0)),
        }

    def _as_vec3(self, value):
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            xyz = [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in xyz):
            return None
        return xyz

    def _distance3(self, a, b):
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    def apply_label_refinements(self, labels):
        """Persist open-vocabulary label refinements from VLM/Gemini."""
        if not isinstance(labels, list):
            return {"updated_objects": 0, "updated_static_targets": 0}
        updated_objects = 0
        updated_targets = 0
        for item in labels:
            if not isinstance(item, dict):
                continue
            obj_id = str(item.get("id", "")).strip()
            raw_label = str(item.get("label", "")).strip().lower()
            label = self._normalized_label(raw_label)
            if not obj_id or not label or label in {"unknown", "unknown_seg", "unknown seg"}:
                continue
            try:
                confidence = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < 0.35:
                continue
            record = {"label": label, "raw_label": raw_label, "confidence": round(confidence, 3)}
            obj = self.objects.get(obj_id)
            if obj is None:
                hint = self._normalized_label(item.get("label_hint", ""))
                if hint and confidence >= 0.75:
                    for candidate in self.objects.values():
                        if candidate.get("missing_since") is not None:
                            continue
                        if self._normalized_label(candidate.get("raw_label", candidate.get("label", ""))) == hint:
                            obj = candidate
                            obj_id = str(candidate.get("id", obj_id))
                            break
            self.label_refinements[obj_id] = record
            if obj is not None:
                obj["raw_label"] = obj.get("raw_label", obj.get("label"))
                obj["vlm_label"] = raw_label
                obj["label"] = label
                obj["label_refined"] = True
                obj["label_confidence"] = round(confidence, 3)
                self.objects[obj_id] = obj
                updated_objects += 1
            target = self.static_targets.get(obj_id)
            if target is not None:
                target["raw_label"] = target.get("raw_label", target.get("label"))
                target["vlm_label"] = raw_label
                target["label"] = label
                target["label_refined"] = True
                target["label_confidence"] = round(confidence, 3)
                self.static_targets[obj_id] = target
                updated_targets += 1
        self.world_debug["label_refinements"] = {
            "count": int(len(self.label_refinements)),
            "latest": self.label_refinements,
        }
        return {"updated_objects": updated_objects, "updated_static_targets": updated_targets}

    def _point_segment_distance(self, p, a, b):
        ab = [b[i] - a[i] for i in range(3)]
        ap = [p[i] - a[i] for i in range(3)]
        ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]
        if ab2 <= 1e-12:
            return self._distance3(p, a)
        t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab2
        t = max(0.0, min(1.0, t))
        q = [a[i] + t * ab[i] for i in range(3)]
        return self._distance3(p, q)

    def _hand_object_capsule_distance(self, hand_track, obj):
        obj_pos = self._as_vec3(obj.get("position_3d"))
        if obj_pos is None:
            return None
        obj_r = float(obj.get("proxy_radius_m", 0.06) or 0.06)
        hand_caps = None
        vol = hand_track.get("volume_3d")
        if isinstance(vol, dict):
            caps = vol.get("capsules")
            if isinstance(caps, list) and caps:
                hand_caps = caps
        if not hand_caps:
            center = self._as_vec3(hand_track.get("center_3d"))
            if center is None:
                return None
            return max(0.0, self._distance3(center, obj_pos) - obj_r), 0.0

        best = None
        for cap in hand_caps:
            a = self._as_vec3(cap.get("a"))
            b = self._as_vec3(cap.get("b"))
            cr = float(cap.get("r", 0.0) or 0.0)
            if a is None or b is None:
                continue
            d = self._point_segment_distance(obj_pos, a, b) - (obj_r + cr)
            if best is None or d < best:
                best = d
        if best is None:
            return None
        return max(0.0, float(best)), min(0.0, float(best))

    def _hand_capsule_to_surface_points(self, hand_track, surface_points):
        if not isinstance(surface_points, list) or not surface_points:
            return None
        hand_caps = None
        vol = hand_track.get("volume_3d")
        if isinstance(vol, dict):
            caps = vol.get("capsules")
            if isinstance(caps, list) and caps:
                hand_caps = caps
        parsed_points = []
        for p in surface_points[: self.object_surface_points_max]:
            pv = self._as_vec3(p)
            if pv is not None:
                parsed_points.append(pv)
        if not parsed_points:
            return None
        if not hand_caps:
            center = self._as_vec3(hand_track.get("center_3d"))
            if center is None:
                return None
            d = min(self._distance3(center, p) for p in parsed_points)
            return max(0.0, d), 0.0
        best = None
        for cap in hand_caps:
            a = self._as_vec3(cap.get("a"))
            b = self._as_vec3(cap.get("b"))
            cr = float(cap.get("r", 0.0) or 0.0)
            if a is None or b is None:
                continue
            for p in parsed_points:
                d = self._point_segment_distance(p, a, b) - (cr + 0.01)
                if best is None or d < best:
                    best = d
        if best is None:
            return None
        return max(0.0, float(best)), min(0.0, float(best))

    def _hand_track_key(self, hand, index: int):
        side = str(hand.get("side", "unknown")).lower()
        if side in {"left", "right"}:
            return f"hand_{side}"
        return str(hand.get("id", f"hand_unknown_{index}"))

    def _update_hand_tracks(self, now: float):
        seen = set()
        updated_tracks = {}
        for index, hand in enumerate(self.hands):
            key = self._hand_track_key(hand, index)
            seen.add(key)
            center = self._as_vec3(hand.get("center_3d"))
            if center is None:
                continue
            previous = self.hand_tracks.get(key)
            if previous:
                velocity = [
                    round(center[i] - previous["center_3d"][i], 4)
                    for i in range(3)
                ]
                age_frames = int(previous.get("age_frames", 0)) + 1
                missing_frames = 0
            else:
                velocity = [0.0, 0.0, 0.0]
                age_frames = 1
                missing_frames = 0
            updated_tracks[key] = {
                "id": key,
                "side": hand.get("side", "unknown"),
                "confidence": float(hand.get("confidence", 0.0) or 0.0),
                "pixel_center": hand.get("pixel_center", [0.0, 0.0]),
                "image_norm_center": hand.get("image_norm_center", [0.0, 0.0]),
                "image_size": hand.get("image_size", previous.get("image_size", []) if previous else []),
                "landmarks_px": hand.get("landmarks_px", previous.get("landmarks_px", []) if previous else []),
                "depth": float(hand.get("depth", 0.0) or 0.0),
                "center_3d": center,
                "velocity_3d": velocity,
                "age_frames": age_frames,
                "missing_frames": missing_frames,
                "last_seen": now,
                "contact_streak": int(previous.get("contact_streak", 0)) if previous else 0,
                "non_contact_streak": int(previous.get("non_contact_streak", 0)) if previous else 0,
                "is_contacting": bool(previous.get("is_contacting", False)) if previous else False,
                "active_object_id": previous.get("active_object_id") if previous else None,
                "volume_3d": hand.get("volume_3d") if isinstance(hand.get("volume_3d"), dict) else (previous.get("volume_3d") if previous else None),
                "jepa_embedding": hand.get("jepa_embedding", previous.get("jepa_embedding", []) if previous else []),
                "jepa_history": self._append_emb_history(
                    previous.get("jepa_history", []) if previous else [],
                    hand.get("jepa_embedding", previous.get("jepa_embedding", []) if previous else []),
                ),
            }
            updated_tracks[key]["jepa_temporal_embedding"] = self._temporal_embedding(
                updated_tracks[key]["jepa_history"]
            )
            history = list(self.hand_trajectories.get(key, []))
            history.append([round(center[0], 4), round(center[1], 4), round(center[2], 4)])
            if len(history) > max(4, self.hand_trail_max_points):
                history = history[-self.hand_trail_max_points:]
            self.hand_trajectories[key] = history

        for key, prev in self.hand_tracks.items():
            if key in seen:
                continue
            missing = int(prev.get("missing_frames", 0)) + 1
            if missing > 15:
                continue
            keep = dict(prev)
            keep["missing_frames"] = missing
            keep["confidence"] = max(0.0, float(prev.get("confidence", 0.0)) * 0.9)
            updated_tracks[key] = keep
        self.hand_tracks = updated_tracks

    def _export_hand_trajectories(self):
        items = []
        for hand_id, points in self.hand_trajectories.items():
            if not points:
                continue
            items.append({
                "hand_id": hand_id,
                "points_3d": points[-max(4, self.hand_trail_max_points):],
            })
        return items

    def _bbox_area(self, bbox):
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            return 0.0
        try:
            w = max(0.0, float(bbox[2]) - float(bbox[0]))
            h = max(0.0, float(bbox[3]) - float(bbox[1]))
        except (TypeError, ValueError):
            return 0.0
        return float(w * h)

    def _bbox_iou(self, a, b):
        if not (isinstance(a, (list, tuple)) and len(a) >= 4 and isinstance(b, (list, tuple)) and len(b) >= 4):
            return 0.0
        try:
            ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
            bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
        except (TypeError, ValueError):
            return 0.0
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 1e-9:
            return 0.0
        return float(inter / union)

    def _bbox_intersection_over_min_area(self, a, b):
        if not (isinstance(a, (list, tuple)) and len(a) >= 4 and isinstance(b, (list, tuple)) and len(b) >= 4):
            return 0.0
        try:
            ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
            bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
        except (TypeError, ValueError):
            return 0.0
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = max(1e-9, min(area_a, area_b))
        return float(inter / denom)

    def _hand_bbox_norm(self, hand):
        points = hand.get("landmarks_px", [])
        size = hand.get("image_size", [])
        if not (isinstance(points, list) and points and isinstance(size, (list, tuple)) and len(size) >= 2):
            return None
        try:
            w = max(1.0, float(size[0]))
            h = max(1.0, float(size[1]))
            xs = [float(p[0]) / w for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
            ys = [float(p[1]) / h for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
        except (TypeError, ValueError):
            return None
        if len(xs) < 4 or len(ys) < 4:
            return None
        pad = 0.015
        return [
            max(0.0, min(1.0, min(xs) - pad)),
            max(0.0, min(1.0, min(ys) - pad)),
            max(0.0, min(1.0, max(xs) + pad)),
            max(0.0, min(1.0, max(ys) + pad)),
        ]

    def _static_target_persistence_source(self):
        colmap = self.world_debug.get("colmap_depth_prior", {})
        if isinstance(colmap, dict) and bool(colmap.get("enabled", False)):
            return "colmap_static_prior"
        return "static_bootstrap"

    def _static_target_diameter_m(self, label):
        norm = self._normalized_label(label)
        if norm == "coaster":
            return 0.095
        if norm in {"dish", "plate", "platter"}:
            return 0.18
        if norm in {"mat", "black mat", "table mat", "placemat"}:
            return 0.385
        if norm == "tray":
            return 0.32
        return 0.11

    def _project_static_target(self, target):
        pos = self._as_vec3(target.get("position_3d"))
        pose = self.camera_pose if isinstance(self.camera_pose, dict) else {}
        cam = self._as_vec3(pose.get("camera_position_world"))
        intr = self.world_debug.get("intrinsics", {}) if isinstance(self.world_debug, dict) else {}
        if pos is None or cam is None or not isinstance(intr, dict):
            return {}

        rot = pose.get("rotation_cw")
        if not (isinstance(rot, (list, tuple)) and len(rot) >= 3):
            rot_wc = pose.get("rotation_wc")
            if isinstance(rot_wc, (list, tuple)) and len(rot_wc) >= 3:
                try:
                    rot = [
                        [rot_wc[0][0], rot_wc[1][0], rot_wc[2][0]],
                        [rot_wc[0][1], rot_wc[1][1], rot_wc[2][1]],
                        [rot_wc[0][2], rot_wc[1][2], rot_wc[2][2]],
                    ]
                except Exception:
                    rot = None
        if not (isinstance(rot, (list, tuple)) and len(rot) >= 3):
            return {}

        try:
            rotation_cw = np.asarray(rot, dtype=np.float32).reshape(3, 3)
            world_point = np.asarray(pos, dtype=np.float32)
            camera_world = np.asarray(cam, dtype=np.float32)
            camera_point = rotation_cw @ (world_point - camera_world)
            x, y, z = [float(v) for v in camera_point[:3]]
            fx = float(intr.get("fx", 0.0) or 0.0)
            fy = float(intr.get("fy", 0.0) or 0.0)
            cx = float(intr.get("cx", 0.0) or 0.0)
            cy = float(intr.get("cy", 0.0) or 0.0)
        except Exception:
            return {}
        if not all(math.isfinite(v) for v in [x, y, z, fx, fy, cx, cy]) or fx <= 0.0 or fy <= 0.0:
            return {}

        width = float(intr.get("width", 0.0) or 0.0)
        height = float(intr.get("height", 0.0) or 0.0)
        if width <= 1.0:
            width = max(1.0, cx * 2.0)
        if height <= 1.0:
            height = max(1.0, cy * 2.0)

        out = {
            "position_camera_3d_current": [round(x, 4), round(y, 4), round(z, 4)],
            "current_depth": round(z, 4),
            "visible_in_current_view": False,
        }
        if z <= 0.05:
            return out

        u = (fx * (x / z) + cx) / width
        v = (fy * (y / z) + cy) / height
        if not (math.isfinite(u) and math.isfinite(v)):
            return out

        diameter = self._static_target_diameter_m(target.get("label", ""))
        bw = max(0.035, min(0.32, (fx * diameter / z) / width))
        bh = max(0.035, min(0.32, (fy * diameter / z) / height))
        bbox = [
            max(0.0, min(1.0, u - bw * 0.5)),
            max(0.0, min(1.0, v - bh * 0.5)),
            max(0.0, min(1.0, u + bw * 0.5)),
            max(0.0, min(1.0, v + bh * 0.5)),
        ]
        out.update({
            "projected_center": [round(u, 4), round(v, 4)],
            "projected_bbox": [round(float(c), 4) for c in bbox],
            "visible_in_current_view": bool(-0.25 <= u <= 1.25 and -0.25 <= v <= 1.25),
        })
        return out

    def _export_static_targets(self):
        now = time.time()
        exported = []
        for target in self.static_targets.values():
            item = target.copy()
            item.update(self._project_static_target(item))
            item["persistent"] = bool(
                item.get("persistent", False)
                or (self.static_target_persist_locked and item.get("locked", False))
            )
            item.setdefault("persistence_source", self._static_target_persistence_source())
            if item.get("last_observed_time") is not None:
                try:
                    item["last_observed_age_s"] = round(max(0.0, now - float(item.get("last_observed_time"))), 3)
                except (TypeError, ValueError):
                    pass
            exported.append(item)

        exported.sort(
            key=lambda item: (
                bool(item.get("locked", False)),
                bool(item.get("persistent", False)),
                int(item.get("hits", 0) or 0),
            ),
            reverse=True,
        )
        return exported[: max(1, int(self.static_target_export_max))]

    def _match_static_target(self, pos, emb, bbox=None, label=""):
        best_id = None
        best_score = -1e9
        for tid, t in self.static_targets.items():
            tpos = self._as_vec3(t.get("position_3d"))
            if tpos is None:
                continue
            d = self._distance3(pos, tpos)
            iou = self._bbox_iou(bbox, t.get("bbox"))
            label_match = bool(label) and str(label).lower() == str(t.get("label", "")).lower()
            if d > self.static_target_match_dist_m and iou < 0.22:
                continue
            sim = cosine_similarity(emb, t.get("embedding", []))
            if sim < self.static_target_embed_sim_min and d > (self.static_target_match_dist_m * 0.55) and iou < 0.30:
                continue
            score = (
                (1.0 - min(1.0, d / max(self.static_target_match_dist_m, 1e-6)))
                + 0.55 * iou
                + 0.30 * sim
                + (0.12 if label_match else 0.0)
                + 0.05 * float(t.get("hits", 0))
            )
            if score > best_score:
                best_score = score
                best_id = tid
        return best_id

    def _update_static_targets(self):
        static_phase = bool(self.world_debug.get("object_surface_static_phase", False))
        if not static_phase:
            return
        now = time.time()
        persistence_source = self._static_target_persistence_source()
        debug = {"considered": 0, "accepted": 0, "locked": 0, "rejects": {}}
        for obj in self.export_objects():
            debug["considered"] += 1
            label = self._normalized_label(obj.get("label", ""))
            area = self._bbox_area(obj.get("bbox"))
            # During static phase, allow unlabeled/stable regions as target candidates too.
            is_unlabeled_region = label in {"", "unknown", "unknown_seg", "unknown seg"} or label.startswith("unknown")
            is_candidate = (label in self.static_target_labels) or is_unlabeled_region
            if label in self.static_target_exclude_labels and label not in self.static_target_labels:
                is_candidate = False
            if not is_candidate:
                debug["rejects"]["label"] = int(debug["rejects"].get("label", 0)) + 1
                continue
            if area < self.static_target_min_area or area > self.static_target_max_area:
                debug["rejects"]["area"] = int(debug["rejects"].get("area", 0)) + 1
                continue
            pos = self._as_vec3(obj.get("position_3d"))
            if pos is None:
                debug["rejects"]["position"] = int(debug["rejects"].get("position", 0)) + 1
                continue
            emb = obj.get("embedding", []) if isinstance(obj.get("embedding", []), list) else []

            matched = self._match_static_target(pos, emb, bbox=obj.get("bbox"), label=label)
            vel3 = obj.get("velocity_3d", [0.0, 0.0, 0.0])
            try:
                speed = math.sqrt(sum(float(v) * float(v) for v in vel3[:3]))
            except Exception:
                speed = 0.0
            matched_iou = 0.0
            if matched is not None:
                matched_iou = self._bbox_iou(obj.get("bbox"), self.static_targets.get(matched, {}).get("bbox"))
            if speed > self.static_target_vel_max and matched_iou < 0.20:
                debug["rejects"]["velocity"] = int(debug["rejects"].get("velocity", 0)) + 1
                continue
            if matched is None:
                matched = f"target_{len(self.static_targets) + 1}"
                self.static_targets[matched] = {
                    "id": matched,
                    "source_object_id": str(obj.get("id", "")),
                    "label": label,
                    "hits": 0,
                    "first_observed_time": now,
                    "last_observed_time": now,
                    "last_observed_frame": self.camera_pose.get("frame_index") if isinstance(self.camera_pose, dict) else None,
                    "position_3d": pos,
                    "position_camera_3d": self._as_vec3(obj.get("position_camera_3d")) or pos,
                    "bbox": obj.get("bbox"),
                    "embedding": emb,
                    "locked": False,
                    "persistent": False,
                    "persistence_source": persistence_source,
                }

            item = self.static_targets[matched]
            item["hits"] = int(item.get("hits", 0)) + 1
            item["last_observed_time"] = now
            item["last_observed_frame"] = self.camera_pose.get("frame_index") if isinstance(self.camera_pose, dict) else None
            item["persistence_source"] = item.get("persistence_source") or persistence_source
            alpha = 0.22
            prev = self._as_vec3(item.get("position_3d")) or pos
            item["position_3d"] = [round((1.0 - alpha) * prev[i] + alpha * pos[i], 4) for i in range(3)]
            cam = self._as_vec3(obj.get("position_camera_3d")) or item["position_3d"]
            item["position_camera_3d"] = [round(float(cam[i]), 4) for i in range(3)]
            item["bbox"] = obj.get("bbox", item.get("bbox"))
            # Keep semantic label if we ever get something more specific than unknown_seg.
            if label and not label.startswith("unknown"):
                item["label"] = label
            elif str(item.get("label", "")).startswith("unknown"):
                area_hint = self._bbox_area(item.get("bbox"))
                luma = obj.get("crop_luma_mean", item.get("crop_luma_mean"))
                if self.static_target_infer_dark_label and luma is not None:
                    try:
                        if float(luma) <= self.static_target_dark_luma_max and area_hint >= 0.003:
                            item["label"] = self.static_target_infer_dark_label
                        elif area_hint >= 0.035 and self.static_target_infer_large_label:
                            item["label"] = self.static_target_infer_large_label
                    except (TypeError, ValueError):
                        pass
                elif area_hint >= 0.035 and self.static_target_infer_large_label:
                    item["label"] = self.static_target_infer_large_label
                elif 0.004 <= area_hint <= 0.026 and self.static_target_infer_small_label:
                    item["label"] = self.static_target_infer_small_label
            if obj.get("crop_luma_mean") is not None:
                item["crop_luma_mean"] = obj.get("crop_luma_mean")
            if emb:
                prev_emb = item.get("embedding", []) if isinstance(item.get("embedding", []), list) else []
                if prev_emb and len(prev_emb) == len(emb):
                    b = 0.2
                    item["embedding"] = [(1.0 - b) * float(prev_emb[i]) + b * float(emb[i]) for i in range(len(emb))]
                else:
                    item["embedding"] = emb
            label_for_lock = str(item.get("label", ""))
            if int(item["hits"]) >= self.static_target_hits_min and (self.static_target_lock_unknown or not label_for_lock.startswith("unknown")):
                item["locked"] = True
                item["persistent"] = bool(self.static_target_persist_locked)
                item.setdefault("locked_time", now)
                item["persistence_source"] = persistence_source
            self.static_targets[matched] = item
            debug["accepted"] += 1
        debug["locked"] = int(sum(1 for t in self.static_targets.values() if bool(t.get("locked", False))))
        debug["total"] = int(len(self.static_targets))
        self.world_debug["static_target_update"] = debug

    def _bbox_center_inside(self, inner, outer):
        if not (isinstance(inner, (list, tuple)) and len(inner) >= 4 and isinstance(outer, (list, tuple)) and len(outer) >= 4):
            return False
        try:
            cx = 0.5 * (float(inner[0]) + float(inner[2]))
            cy = 0.5 * (float(inner[1]) + float(inner[3]))
            x1, y1, x2, y2 = [float(v) for v in outer[:4]]
        except (TypeError, ValueError):
            return False
        return bool(x1 <= cx <= x2 and y1 <= cy <= y2)

    def _find_best_static_target(self, object_camera_pos, object_bbox=None):
        if object_camera_pos is None:
            return None
        candidates = [t for t in self.static_targets.values() if bool(t.get("locked", False))]
        if not candidates:
            return None
        best = None
        for t in candidates:
            projection = self._project_static_target(t)
            tcam = self._as_vec3(projection.get("position_camera_3d_current")) or self._as_vec3(t.get("position_camera_3d"))
            if tcam is None:
                continue
            d3 = self._distance3(object_camera_pos, tcam)
            dx = abs(float(object_camera_pos[0]) - float(tcam[0]))
            dy = abs(float(object_camera_pos[1]) - float(tcam[1]))
            xy = math.sqrt(dx * dx + dy * dy)
            target_bbox = projection.get("projected_bbox") or t.get("bbox")
            overlap = self._bbox_intersection_over_min_area(object_bbox, target_bbox)
            center_inside = self._bbox_center_inside(object_bbox, target_bbox)
            score = (
                3.0 * (1.0 if center_inside else 0.0)
                + 2.2 * min(1.0, float(overlap))
                + 1.4 * (1.0 - min(1.0, xy / 0.55))
                + 0.6 * (1.0 - min(1.0, d3 / 1.25))
                + 0.03 * min(10.0, float(t.get("hits", 0) or 0))
            )
            if best is None or score > best["support_score"]:
                best = {
                    "object_id": t.get("id"),
                    "label": t.get("label", "target"),
                    "distance_m": float(d3),
                    "xy_distance_m": float(xy),
                    "other_cam": tcam,
                    "support_score": float(score),
                    "bbox_overlap": float(overlap),
                    "bbox_center_inside": bool(center_inside),
                    "is_static_target": True,
                }
        return best

    def _decode_temporal_target_from_motion(self, obj, temporal_pred):
        """Decode the learned future motion into a persistent support target.

        The temporal head predicts future dynamics. The semantic destination is
        then grounded by snapping the predicted future 3D point to the persistent
        static support regions built from the COLMAP/depth prior and online SLAM.
        """
        if obj is None or temporal_pred is None:
            return None
        pos = self._as_vec3(obj.get("position_3d"))
        if pos is None:
            return None
        try:
            delta = [
                float(temporal_pred.get("motion_dx", 0.0) or 0.0),
                float(temporal_pred.get("motion_dy", 0.0) or 0.0),
                float(temporal_pred.get("motion_dz", 0.0) or 0.0),
            ]
        except (TypeError, ValueError):
            return None
        future = [float(pos[i]) + float(delta[i]) for i in range(3)]
        candidates = [t for t in self.static_targets.values() if bool(t.get("locked", False))]
        if not candidates:
            return None
        best = None
        for target in candidates:
            tpos = self._as_vec3(target.get("position_3d"))
            if tpos is None:
                continue
            d = self._distance3(future, tpos)
            score = max(0.0, min(1.0, 1.0 - d / 0.85))
            if best is None or score > best["score"]:
                label = self._canonical_support_label(target.get("label") or target.get("id") or "target")
                best = {
                    "target_id": target.get("id"),
                    "target_label": label or str(target.get("id") or "target"),
                    "score": float(score),
                    "distance_m": float(d),
                    "predicted_position_3d": future,
                }
        if best is None:
            return None
        return best

    def _find_object_by_id(self, object_id, include_missing=False):
        if object_id is None:
            return None
        obj = self.objects.get(object_id)
        if obj and (include_missing or obj.get("missing_since") is None):
            return obj
        return None

    def _canonical_support_label(self, label):
        norm = self._normalized_label(label)
        if norm in {"black mat", "table mat", "placemat", "dish", "plate"}:
            return "mat"
        if norm in {"plastic tray", "white tray"}:
            return "tray"
        return norm

    def _configured_transfer_targets(self):
        return [
            self._canonical_support_label(item)
            for item in os.environ.get("DEMO_TRANSFER_TARGETS", "").split(",")
            if item.strip()
        ]

    def _alternate_transfer_target(self, source_label):
        targets = self._configured_transfer_targets()
        source = self._canonical_support_label(source_label)
        if len(targets) < 2 or source not in targets:
            return None
        idx = targets.index(source)
        return targets[(idx + 1) % len(targets)]

    def _persistent_transfer_target_for_object(self, object_id, fallback_target=None):
        obj = self._find_object_by_id(object_id, include_missing=True)
        source_label = None
        if obj is not None:
            source_label = self._canonical_support_label(obj.get("support_target_label"))
            if source_label not in self._configured_transfer_targets():
                pos_cam = self._as_vec3(obj.get("position_camera_3d"))
                relation = self._find_place_relation(object_id, pos_cam, object_bbox=obj.get("bbox"))
                source_label = self._canonical_support_label((relation or {}).get("nearest_object_label"))
        target = self._alternate_transfer_target(source_label)
        if target:
            return target
        fallback = self._canonical_support_label(fallback_target)
        if fallback in self._configured_transfer_targets():
            return fallback
        return fallback_target

    def _relation_with_label(self, relation, label):
        if relation is None or not label:
            return relation
        out = dict(relation)
        out["nearest_object_label"] = label
        out["support_target_label"] = label
        out["support_inferred_from_transfer_memory"] = True
        return out

    def _support_relation_confident(self, relation):
        if relation is None:
            return False
        return bool(
            relation.get("is_on_support", False)
            or relation.get("bbox_center_inside", False)
            or float(relation.get("bbox_overlap", 0.0) or 0.0) >= 0.08
            or float(relation.get("support_score", 0.0) or 0.0) >= 1.65
        )

    def _update_object_support_memory(self, now):
        active_ids = {
            str(active.get("object_id"))
            for active in self.manipulation_active.values()
            if active.get("object_id") is not None
        }
        for obj_id, obj in self.objects.items():
            if obj.get("missing_since") is not None or obj_id in active_ids:
                continue
            label = self._normalized_label(obj.get("label", ""))
            if label in self.static_target_labels or label in self.static_target_exclude_labels:
                continue
            pos = self._as_vec3(obj.get("position_camera_3d"))
            relation = self._find_place_relation(obj_id, pos, object_bbox=obj.get("bbox"))
            if not self._support_relation_confident(relation):
                continue
            support_label = self._canonical_support_label(relation.get("nearest_object_label"))
            if not support_label:
                continue
            obj["support_relation"] = relation
            obj["support_target_id"] = relation.get("nearest_object_id")
            obj["support_target_label"] = support_label
            obj["support_updated_time"] = now

    def _find_place_relation(self, object_id, object_camera_pos, object_bbox=None):
        if object_camera_pos is None:
            return None
        nearest = self._find_best_static_target(object_camera_pos, object_bbox=object_bbox)
        if nearest is None:
            for other in self.export_objects():
                other_id = other.get("id")
                if other_id == object_id:
                    continue
                other_cam = self._as_vec3(other.get("position_camera_3d"))
                if other_cam is None:
                    continue
                d3 = self._distance3(object_camera_pos, other_cam)
                if nearest is None or d3 < nearest["distance_m"]:
                    nearest = {
                        "object_id": other_id,
                        "label": other.get("label", "unknown"),
                        "distance_m": float(d3),
                        "other_cam": other_cam,
                    }
        if nearest is None:
            return None

        dx = abs(float(object_camera_pos[0]) - float(nearest["other_cam"][0]))
        dy = abs(float(object_camera_pos[1]) - float(nearest["other_cam"][1]))
        dz = float(object_camera_pos[2]) - float(nearest["other_cam"][2])
        is_near_xy = dx <= self.manip_relation_near_xy_m and dy <= self.manip_relation_near_xy_m
        is_near_3d = float(nearest["distance_m"]) <= self.manip_relation_near_3d_m
        support_score = float(nearest.get("support_score", 0.0) or 0.0)
        bbox_overlap = float(nearest.get("bbox_overlap", 0.0) or 0.0)
        center_inside = bool(nearest.get("bbox_center_inside", False))
        is_on_support = bool(center_inside or bbox_overlap >= 0.05 or support_score >= 1.35)
        is_behind = bool(is_near_xy and dz >= self.manip_relation_behind_dz_m)
        return {
            "nearest_object_id": nearest["object_id"],
            "nearest_object_label": nearest["label"],
            "nearest_distance_m": round(float(nearest["distance_m"]), 4),
            "support_score": round(float(support_score), 4),
            "bbox_overlap": round(float(bbox_overlap), 4),
            "bbox_center_inside": bool(center_inside),
            "delta_camera_xyz": [round(float(object_camera_pos[0] - nearest["other_cam"][0]), 4),
                                 round(float(object_camera_pos[1] - nearest["other_cam"][1]), 4),
                                 round(dz, 4)],
            "is_near_3d": bool(is_near_3d or is_on_support),
            "is_behind_nearest": bool(is_behind),
            "is_on_support": bool(is_on_support),
            "is_static_target": bool(nearest.get("is_static_target", False)),
        }

    def _update_hand_object_interactions(self, now: float):
        interactions = []
        objects = self.export_objects()
        for hand_id, hand in self.hand_tracks.items():
            side = str(hand.get("side", "unknown")).lower()
            if self.hand_interaction_sides and side not in self.hand_interaction_sides:
                continue
            hand_pos = self._as_vec3(hand.get("center_3d"))
            if hand_pos is None:
                continue
            hand_bbox_norm = self._hand_bbox_norm(hand)
            nearest = None
            intent_candidates = []
            for obj in objects:
                label = str(obj.get("label", "")).lower()
                if label in self.interaction_label_denylist:
                    continue
                obj_pos = self._as_vec3(obj.get("position_3d"))
                if obj_pos is None:
                    continue
                capsule_dist = self._hand_capsule_to_surface_points(hand, obj.get("surface_points_3d"))
                if capsule_dist is None:
                    capsule_dist = self._hand_object_capsule_distance(hand, obj)
                if capsule_dist is not None:
                    dist, penetration = capsule_dist
                else:
                    dist, penetration = self._distance3(hand_pos, obj_pos), 0.0
                overlap_2d = self._bbox_intersection_over_min_area(hand_bbox_norm, obj.get("bbox"))
                jepa = jepa_interaction_score(
                    hand.get("jepa_temporal_embedding", hand.get("jepa_embedding", [])),
                    obj.get("jepa_temporal_embedding", obj.get("jepa_embedding", [])),
                    float(dist),
                    near_distance_m=float(self.hand_near_distance_m),
                )
                score = float(jepa.get("jepa_interaction_score", 0.0))
                sim = float(jepa.get("jepa_similarity", 0.0))
                effective_dist = float(dist)
                if overlap_2d >= self.hand_contact_2d_overlap_min:
                    effective_dist = min(effective_dist, self.hand_contact_2d_effective_distance_m)
                if self.jepa_use_for_contact and sim > 0.0:
                    effective_dist = max(0.0, float(effective_dist) * (1.0 - self.jepa_distance_reweight * min(1.0, sim)))
                hand_speed = math.sqrt(sum(float(v) * float(v) for v in hand.get("velocity_3d", [0.0, 0.0, 0.0])[:3]))
                obj_speed = math.sqrt(sum(float(v) * float(v) for v in obj.get("velocity_3d", [0.0, 0.0, 0.0])[:3]))
                feat = build_feature_vector(
                    hand.get("jepa_temporal_embedding", hand.get("jepa_embedding", [])),
                    obj.get("jepa_temporal_embedding", obj.get("jepa_embedding", [])),
                    float(dist),
                    float(effective_dist),
                    float(sim),
                    float(hand_speed),
                    float(obj_speed),
                    int(hand.get("contact_streak", 0)),
                )
                temporal_pred = self.temporal_head.predict(feat)
                decoded_target = self._decode_temporal_target_from_motion(obj, temporal_pred)
                temporal_target_label = str(temporal_pred.get("target_label", "target"))
                decoded_target_score = float((decoded_target or {}).get("score", 0.0) or 0.0)
                decoded_target_raw_label = str((decoded_target or {}).get("target_label") or "")
                decoded_target_label = (
                    decoded_target_raw_label
                    if decoded_target is not None
                    and decoded_target_score >= 0.55
                    and (
                        self._canonical_support_label(decoded_target_raw_label) == self._canonical_support_label(temporal_target_label)
                        or self._canonical_support_label(temporal_target_label) not in self._configured_transfer_targets()
                    )
                    else temporal_target_label
                )
                pred_future_latent = temporal_pred.get("future_latent", [])
                obj_latent = obj.get("jepa_temporal_embedding", obj.get("jepa_embedding", []))
                latent_conf = 0.0
                if isinstance(pred_future_latent, list) and isinstance(obj_latent, list) and pred_future_latent and obj_latent:
                    try:
                        a = np.asarray(pred_future_latent[:64], dtype=np.float32)
                        b = np.asarray(obj_latent[:64], dtype=np.float32)
                        if a.shape == b.shape and a.size > 0:
                            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                            if denom > 1e-6:
                                latent_conf = max(0.0, min(1.0, 0.5 + 0.5 * float(np.dot(a, b) / denom)))
                    except Exception:
                        latent_conf = 0.0
                intent_candidates.append({
                    "object_id": obj.get("id"),
                    "label": self._interaction_display_label(obj, obj.get("label", "object")),
                    "visual_identity_class": obj.get("visual_identity_class", ""),
                    "distance_m": float(dist),
                    "effective_distance_m": float(effective_dist),
                    "jepa_similarity": float(sim),
                    "jepa_interaction_score": float(score),
                    "pred_contact_prob": float(temporal_pred.get("contact_prob", 0.0)),
                    "pred_placement_prob": float(temporal_pred.get("placement_prob", 0.0)),
                    "pred_release_prob": float(temporal_pred.get("release_prob", 0.0)),
                    "pred_target_tray_prob": float(temporal_pred.get("target_tray_prob", 0.5)),
                    "pred_target_label": str(decoded_target_label),
                    "pred_target_id": str((decoded_target or {}).get("target_id") or ""),
                    "pred_target_motion_score": float((decoded_target or {}).get("score", 0.0) or 0.0),
                    "pred_motion_delta": [
                        float(temporal_pred.get("motion_dx", 0.0)),
                        float(temporal_pred.get("motion_dy", 0.0)),
                        float(temporal_pred.get("motion_dz", 0.0)),
                    ],
                    "pred_future_latent": pred_future_latent[:64] if isinstance(pred_future_latent, list) else [],
                    "pred_future_latent_confidence": float(latent_conf),
                    "hand_object_overlap_2d": float(overlap_2d),
                })
                if nearest is None or effective_dist < nearest["effective_distance_m"]:
                    nearest = {
                        "object_id": obj.get("id"),
                        "label": obj.get("label"),
                        "distance_m": float(dist),
                        "effective_distance_m": float(effective_dist),
                        "penetration_m": float(abs(min(0.0, penetration))),
                        "object_pos_3d": obj_pos,
                        "jepa_similarity": sim,
                        "jepa_interaction_score": score,
                        "hand_object_overlap_2d": float(overlap_2d),
                        "temporal_pred": temporal_pred,
                        "decoded_target": decoded_target,
                    }

            if nearest is None:
                continue

            def learned_candidate_sort_key(candidate):
                return (
                    float(candidate.get("pred_contact_prob", 0.0) or 0.0),
                    float(candidate.get("hand_object_overlap_2d", 0.0) or 0.0),
                    -float(candidate.get("effective_distance_m", 9.9) or 9.9),
                )

            movable_temporal_candidates = [
                c for c in intent_candidates
                if self._is_trackable_object_label(c.get("label", ""))
                and float(c.get("effective_distance_m", 9.9) or 9.9) <= max(0.55, self.hand_near_distance_m * 2.4)
            ]
            movable_temporal_candidates.sort(key=learned_candidate_sort_key, reverse=True)
            learned_candidate = movable_temporal_candidates[0] if movable_temporal_candidates else None

            was_contacting = bool(hand.get("is_contacting", False))
            contact_streak = int(hand.get("contact_streak", 0))
            non_contact_streak = int(hand.get("non_contact_streak", 0))
            distance_m = nearest["distance_m"]
            nearest_label_norm = self._normalized_label(nearest.get("label", ""))
            if was_contacting:
                active_obj_id = hand.get("active_object_id")
                active_obj = self._find_object_by_id(active_obj_id, include_missing=True)
                if active_obj is not None:
                    active_label_norm = self._normalized_label(active_obj.get("label", nearest.get("label", "")))
                    active_pos = self._as_vec3(active_obj.get("position_3d"))
                    active_dist = None
                    active_penetration = 0.0
                    capsule_dist = self._hand_capsule_to_surface_points(hand, active_obj.get("surface_points_3d"))
                    if capsule_dist is None:
                        capsule_dist = self._hand_object_capsule_distance(hand, active_obj)
                    if capsule_dist is not None:
                        active_dist, active_penetration = capsule_dist
                    elif active_pos is not None:
                        active_dist = self._distance3(hand_pos, active_pos)
                    if active_dist is not None and math.isfinite(float(active_dist)):
                        distance_m = float(active_dist)
                        nearest_label_norm = active_label_norm
                        nearest["distance_m"] = float(active_dist)
                        nearest["effective_distance_m"] = min(
                            float(nearest.get("effective_distance_m", active_dist)),
                            float(active_dist),
                        )
                        nearest["penetration_m"] = float(abs(min(0.0, active_penetration)))
                        nearest["object_id"] = active_obj.get("id", active_obj_id)
                        nearest["label"] = active_obj.get("label", nearest.get("label"))
                        nearest["object_pos_3d"] = active_pos or nearest.get("object_pos_3d")
            touch_distance_m = self._touch_threshold(nearest_label_norm, "distance")
            touch_start_distance_m = self._touch_threshold(nearest_label_norm, "start")
            touch_end_distance_m = self._touch_threshold(nearest_label_norm, "end")
            contact_enter_frames = self._contact_enter_frames(nearest_label_norm)

            penetration_m = float(nearest.get("penetration_m", 0.0) or 0.0)
            in_contact_zone = (
                distance_m <= (self.hand_contact_distance_m + self.hand_capsule_contact_padding_m)
                or penetration_m > 0.003
            )
            jepa_score = float(nearest.get("jepa_interaction_score", 0.0) or 0.0)
            jepa_sim = float(nearest.get("jepa_similarity", 0.0) or 0.0)
            overlap_2d = float(nearest.get("hand_object_overlap_2d", 0.0) or 0.0)
            hand_norm = hand.get("image_norm_center", [None, None])
            hx = hand_norm[0] if isinstance(hand_norm, (list, tuple)) and len(hand_norm) >= 2 else None
            hy = hand_norm[1] if isinstance(hand_norm, (list, tuple)) and len(hand_norm) >= 2 else None
            obj_x = None
            obj_y = None
            obj_now = self._find_object_by_id(nearest["object_id"])
            if obj_now is not None:
                try:
                    obj_x = float(obj_now.get("x"))
                    obj_y = float(obj_now.get("y"))
                except (TypeError, ValueError):
                    obj_x = None
                    obj_y = None
            in_contact_zone_2d = False
            if hx is not None and hy is not None and obj_x is not None and obj_y is not None:
                try:
                    d2d = math.sqrt((float(hx) - float(obj_x)) ** 2 + (float(hy) - float(obj_y)) ** 2)
                    in_contact_zone_2d = d2d <= self.hand_contact_2d_distance_norm
                except (TypeError, ValueError):
                    in_contact_zone_2d = False
            # Strict physical touch signal (3D only): used for contact state transitions.
            overlap_touch_now = bool(
                overlap_2d >= self.hand_contact_2d_overlap_min
                and distance_m <= max(
                    touch_start_distance_m + self.hand_capsule_contact_padding_m,
                    self.hand_near_distance_m,
                )
            )
            strict_touch_now = bool(
                (distance_m <= (touch_start_distance_m + self.hand_capsule_contact_padding_m))
                or penetration_m > 0.003
                or overlap_touch_now
            )
            # Near-intent signal can still use 2D/JEPA, but does not start contact by itself.
            near_or_intent = bool(in_contact_zone or overlap_2d >= self.hand_contact_2d_overlap_min or (in_contact_zone_2d and distance_m <= self.hand_near_distance_m))
            if self.jepa_use_for_contact and jepa_score >= self.jepa_contact_score_threshold and distance_m <= (self.hand_near_distance_m * 1.35):
                near_or_intent = True

            if strict_touch_now:
                contact_streak += 1
                non_contact_streak = 0
            else:
                non_contact_streak += 1
                contact_streak = 0

            if was_contacting:
                strict_keep = bool(
                    (distance_m <= (touch_end_distance_m + self.hand_capsule_contact_padding_m))
                    or penetration_m > 0.002
                )
                is_contacting = strict_keep and (non_contact_streak < self.hand_contact_exit_frames)
            else:
                if self.require_3d_for_contact_start:
                    is_contacting = strict_touch_now and (contact_streak >= contact_enter_frames)
                else:
                    is_contacting = (strict_touch_now or near_or_intent) and (contact_streak >= contact_enter_frames)

            if is_contacting and not was_contacting:
                active_object = self._find_object_by_id(nearest["object_id"])
                start_pos_3d = self._as_vec3(active_object.get("position_3d")) if active_object else nearest["object_pos_3d"]
                start_cam_pos = self._as_vec3(active_object.get("position_camera_3d")) if active_object else None
                source_relation = self._find_place_relation(
                    nearest["object_id"],
                    start_cam_pos,
                    object_bbox=active_object.get("bbox") if active_object else None,
                )
                memory_source_label = self._canonical_support_label(active_object.get("support_target_label")) if active_object else None
                if memory_source_label in self._configured_transfer_targets():
                    source_relation = self._relation_with_label(source_relation or {}, memory_source_label)
                self.manipulation_active[hand_id] = {
                    "hand_id": hand_id,
                    "object_id": nearest["object_id"],
                    "label": nearest["label"],
                    "start_time": now,
                    "start_3d": [round(v, 4) for v in (start_pos_3d or [0.0, 0.0, 0.0])],
                    "last_3d": [round(v, 4) for v in (start_pos_3d or [0.0, 0.0, 0.0])],
                    "last_seen_time": now,
                    "source_relation": source_relation,
                    "source_support_label": self._canonical_support_label(source_relation.get("nearest_object_label")) if source_relation else memory_source_label,
                }
                self.hand_contact_events.append({
                    "time": now,
                    "event": "contact_start",
                    "hand_id": hand_id,
                    "object_id": nearest["object_id"],
                    "label": nearest["label"],
                    "distance_m": round(distance_m, 4),
                })

            learned_state = self.learned_manipulation_active.get(hand_id)
            if learned_state is not None:
                active_learned_id = str(learned_state.get("object_id") or "")
                same_object_candidate = next(
                    (c for c in movable_temporal_candidates if str(c.get("object_id") or "") == active_learned_id),
                    None,
                )
                if same_object_candidate is not None:
                    learned_candidate = same_object_candidate
            grounded_temporal_candidate = next(
                (c for c in intent_candidates if str(c.get("object_id") or "") == str(nearest.get("object_id") or "")),
                None,
            )
            # Once a learned manipulation episode has started, keep decoding its
            # original object. The hand can pass near the other object or a static
            # target during the transfer; those nearby rows should not steal the
            # episode label, target, or release state.
            learned_source_candidate = learned_candidate if learned_state is not None else grounded_temporal_candidate
            learned_source = learned_source_candidate or grounded_temporal_candidate or {
                "object_id": nearest.get("object_id"),
                "label": self._interaction_display_label(self._find_object_by_id(nearest.get("object_id"), include_missing=True) or nearest, nearest.get("label", "object")),
                "pred_contact_prob": float(nearest.get("temporal_pred", {}).get("contact_prob", 0.0) or 0.0),
                "pred_release_prob": float(nearest.get("temporal_pred", {}).get("release_prob", 0.0) or 0.0),
                "pred_target_label": str(
                    (nearest.get("decoded_target") or {}).get("target_label")
                    or nearest.get("temporal_pred", {}).get("target_label", "target")
                ),
                "pred_target_motion_score": float((nearest.get("decoded_target") or {}).get("score", 0.0) or 0.0),
            }
            learned_now_held = False
            learned_now_releasing = False
            learned_contact_prob = float(learned_source.get("pred_contact_prob", 0.0) or 0.0)
            learned_release_prob = float(learned_source.get("pred_release_prob", 0.0) or 0.0)
            learned_target = str(
                learned_source.get("pred_target_label")
                or "target"
            )
            learned_obj_id = str(learned_source.get("object_id") or "")
            learned_obj = self._find_object_by_id(learned_obj_id, include_missing=True)
            learned_label = self._interaction_display_label(learned_obj or learned_source, learned_source.get("label", "object"))
            persistent_target = str(self._persistent_transfer_target_for_object(learned_obj_id, learned_target) or "")
            if self._canonical_support_label(learned_target) not in self._configured_transfer_targets() and persistent_target:
                learned_target = persistent_target
            if learned_state is None and is_contacting and learned_contact_prob >= self.learned_contact_threshold:
                learned_state = {
                    "hand_id": hand_id,
                    "object_id": learned_obj_id,
                    "label": learned_label,
                    "target_label": learned_target,
                    "start_time": now,
                    "last_update": now,
                    "release_time": None,
                    "contact_prob": learned_contact_prob,
                    "release_prob": learned_release_prob,
                }
                self.learned_manipulation_active[hand_id] = learned_state
                self.hand_contact_events.append({
                    "time": now,
                    "event": "learned_contact_start",
                    "hand_id": hand_id,
                    "object_id": learned_obj_id,
                    "label": learned_label,
                    "target_label": learned_target,
                    "pred_contact_prob": round(learned_contact_prob, 4),
                    "learned": True,
                })
            if learned_state is not None:
                elapsed = max(0.0, now - float(learned_state.get("start_time", now)))
                active_id = str(learned_state.get("object_id") or "")
                same_object = bool(not active_id or not learned_obj_id or active_id == learned_obj_id)
                if same_object and learned_target:
                    learned_state["target_label"] = learned_target
                learned_state["last_update"] = now
                learned_state["contact_prob"] = max(float(learned_state.get("contact_prob", 0.0) or 0.0), learned_contact_prob)
                learned_state["release_prob"] = max(float(learned_state.get("release_prob", 0.0) or 0.0), learned_release_prob)
                if (
                    elapsed >= self.learned_min_hold_s
                    and learned_release_prob >= self.learned_release_ui_threshold
                    and learned_state.get("release_time") is None
                ):
                    learned_state["release_time"] = now
                    event = {
                        "time": now,
                        "event": "learned_release",
                        "hand_id": hand_id,
                        "object_id": learned_state.get("object_id"),
                        "label": learned_state.get("label", learned_label),
                        "target_label": learned_state.get("target_label", learned_target),
                        "pred_release_prob": round(learned_release_prob, 4),
                        "pred_contact_prob": round(learned_contact_prob, 4),
                        "learned": True,
                    }
                    self.learned_manipulation_events.append(event)
                    self.hand_contact_events.append(event)
                release_time = learned_state.get("release_time")
                release_age = max(0.0, now - float(release_time)) if release_time is not None else None
                learned_now_held = release_time is None or (release_age is not None and release_age <= self.learned_release_linger_s)
                learned_now_releasing = bool(release_time is not None and release_age is not None and release_age <= self.learned_release_linger_s)
                if (
                    release_age is not None
                    and release_age > self.learned_release_linger_s
                    and (not is_contacting or learned_contact_prob < self.learned_contact_threshold)
                ):
                    self.learned_manipulation_active.pop(hand_id, None)
                    learned_state = None
                    learned_now_held = False
                    learned_now_releasing = False
            if was_contacting and not is_contacting:
                active = self.manipulation_active.get(hand_id)
                active_obj_id = hand.get("active_object_id")
                active_obj = self._find_object_by_id(active_obj_id, include_missing=True)
                end_pos_3d = self._as_vec3(active_obj.get("position_3d")) if active_obj else nearest["object_pos_3d"]
                if active is not None and end_pos_3d is not None:
                    start_pos = self._as_vec3(active.get("start_3d"))
                    if start_pos is None:
                        start_pos = end_pos_3d
                    moved_dist = self._distance3(start_pos, end_pos_3d)
                    end_cam_pos = self._as_vec3(active_obj.get("position_camera_3d")) if active_obj else None
                    place_relation = self._find_place_relation(
                        active.get("object_id"),
                        end_cam_pos,
                        object_bbox=active_obj.get("bbox") if active_obj else None,
                    )
                    source_label = active.get("source_support_label")
                    alternate_label = self._alternate_transfer_target(source_label)
                    if alternate_label and moved_dist >= self.manipulation_min_move_distance_m:
                        raw_label = self._canonical_support_label((place_relation or {}).get("nearest_object_label"))
                        raw_score = float((place_relation or {}).get("support_score", 0.0) or 0.0)
                        if place_relation is None or raw_label == source_label or raw_label != alternate_label or raw_score < 2.2:
                            place_relation = self._relation_with_label(place_relation or {}, alternate_label)
                    event = {
                        "time": now,
                        "event": "pick_place",
                        "hand_id": hand_id,
                        "object_id": active.get("object_id"),
                        "label": active.get("label", nearest.get("label")),
                        "from_3d": [round(v, 4) for v in start_pos],
                        "to_3d": [round(v, 4) for v in end_pos_3d],
                        "move_distance_m": round(float(moved_dist), 4),
                        "picked_up": True,
                        "moved": bool(moved_dist >= self.manipulation_min_move_distance_m),
                        "place_relation": place_relation,
                        "confidence": round(
                            max(
                                0.0,
                                min(
                                    1.0,
                                    0.4
                                    + 0.4 * min(1.0, moved_dist / max(self.manipulation_min_move_distance_m, 1e-6))
                                    + 0.2 * float(hand.get("confidence", 0.0)),
                                ),
                            ),
                            3,
                        ),
                    }
                    self.manipulation_events.append(event)
                    memory_obj = self.objects.get(str(active.get("object_id")))
                    if memory_obj is not None and place_relation is not None:
                        memory_obj["support_relation"] = place_relation
                        memory_obj["support_target_id"] = place_relation.get("nearest_object_id")
                        memory_obj["support_target_label"] = place_relation.get("nearest_object_label")
                        memory_obj["support_updated_time"] = now
                if hand_id in self.manipulation_active:
                    del self.manipulation_active[hand_id]
                self.hand_contact_events.append({
                    "time": now,
                    "event": "contact_end",
                    "hand_id": hand_id,
                    "object_id": hand.get("active_object_id"),
                    "distance_m": round(distance_m, 4),
                })

            hand["contact_streak"] = contact_streak
            hand["non_contact_streak"] = non_contact_streak
            hand["is_contacting"] = is_contacting
            hand["active_object_id"] = nearest["object_id"] if is_contacting else None
            if is_contacting:
                active = self.manipulation_active.get(hand_id)
                if active is not None:
                    active_obj = self._find_object_by_id(active.get("object_id"), include_missing=True)
                    obj_pos = self._as_vec3(active_obj.get("position_3d")) if active_obj else nearest["object_pos_3d"]
                    if obj_pos is not None:
                        active["last_3d"] = [round(v, 4) for v in obj_pos]
                        active["last_seen_time"] = now

            intent_candidates.sort(
                key=lambda c: (
                    float(c.get("pred_contact_prob", 0.0)),
                    float(c.get("jepa_interaction_score", 0.0)),
                    -float(c.get("effective_distance_m", 9.9)),
                ),
                reverse=True,
            )
            top_grab = intent_candidates[:4]
            static_place_candidates = []
            for target in self.static_targets.values():
                if not bool(target.get("locked", False)):
                    continue
                tpos = self._as_vec3(target.get("position_3d"))
                if tpos is None:
                    continue
                d_target = self._distance3(hand_pos, tpos)
                score = max(0.05, min(0.95, 1.0 - (d_target / max(0.75, self.manip_relation_near_3d_m * 4.0))))
                static_place_candidates.append({
                    "object_id": target.get("id"),
                    "label": target.get("label", "target"),
                    "distance_m": float(d_target),
                    "effective_distance_m": float(d_target),
                    "pred_placement_prob": float(score),
                    "jepa_interaction_score": 0.0,
                    "is_static_target": True,
                })
            placement_candidates = sorted(
                intent_candidates + static_place_candidates,
                key=lambda c: (
                    float(c.get("pred_placement_prob", 0.0)),
                    -float(c.get("effective_distance_m", 9.9)),
                ),
                reverse=True,
            )[:4]
            held_object_id = str(
                (hand.get("active_object_id") if is_contacting else "")
                or (learned_state or {}).get("object_id")
                or ""
            )
            held_object = self._find_object_by_id(held_object_id, include_missing=True) if held_object_id else None
            held_object_label = self._interaction_display_label(
                held_object or (learned_state or {}),
                (learned_state or {}).get("label", nearest.get("label", "object")),
            ) if held_object_id else ""

            interactions.append({
                "hand_id": hand_id,
                "side": hand.get("side", "unknown"),
                "hand_confidence": round(float(hand.get("confidence", 0.0)), 3),
                "hand_center_3d": [round(v, 4) for v in hand_pos],
                "nearest_object_id": nearest["object_id"],
                "nearest_object_label": nearest["label"],
                "distance_m": round(distance_m, 4),
                "effective_distance_m": round(float(nearest.get("effective_distance_m", distance_m)), 4),
                "is_near": bool(distance_m <= self.hand_near_distance_m),
                "is_touching_strict": bool(distance_m <= touch_distance_m or penetration_m > 0.003),
                "touch_threshold_m": round(float(touch_distance_m), 4),
                "touch_start_threshold_m": round(float(touch_start_distance_m), 4),
                "contact_enter_frames": int(contact_enter_frames),
                "is_contacting": bool(is_contacting),
                "penetration_m": round(float(nearest.get("penetration_m", 0.0) or 0.0), 4),
                "jepa_similarity": round(jepa_sim, 4),
                "jepa_interaction_score": round(jepa_score, 4),
                "pred_contact_prob": round(float(nearest.get("temporal_pred", {}).get("contact_prob", 0.0)), 4),
                "pred_placement_prob": round(float(nearest.get("temporal_pred", {}).get("placement_prob", 0.0)), 4),
                "pred_release_prob": round(float(nearest.get("temporal_pred", {}).get("release_prob", 0.0)), 4),
                "held_object_id": held_object_id,
                "held_object_label": held_object_label,
                "learned_is_held": bool(learned_now_held),
                "learned_releasing": bool(learned_now_releasing),
                "learned_event_state": "releasing" if learned_now_releasing else ("held" if learned_now_held else "idle"),
                "learned_object_id": str((learned_state or {}).get("object_id") or ""),
                "learned_target_label": str((learned_state or {}).get("target_label") or ""),
                "learned_contact_age_s": round(max(0.0, now - float((learned_state or {}).get("start_time", now))), 3) if learned_state is not None else None,
                "learned_release_age_s": (
                    round(max(0.0, now - float((learned_state or {}).get("release_time"))), 3)
                    if learned_state is not None and (learned_state or {}).get("release_time") is not None
                    else None
                ),
                "pred_target_tray_prob": round(float(nearest.get("temporal_pred", {}).get("target_tray_prob", 0.5)), 4),
                "pred_target_label": str(
                    (nearest.get("decoded_target") or {}).get("target_label")
                    or nearest.get("temporal_pred", {}).get("target_label", "target")
                ),
                "pred_target_id": str((nearest.get("decoded_target") or {}).get("target_id") or ""),
                "pred_target_motion_score": round(float((nearest.get("decoded_target") or {}).get("score", 0.0) or 0.0), 4),
                "pred_motion_delta": [
                    round(float(nearest.get("temporal_pred", {}).get("motion_dx", 0.0)), 4),
                    round(float(nearest.get("temporal_pred", {}).get("motion_dy", 0.0)), 4),
                    round(float(nearest.get("temporal_pred", {}).get("motion_dz", 0.0)), 4),
                ],
                "pred_future_latent_confidence": round(
                    float(
                        next(
                            (
                                c.get("pred_future_latent_confidence", 0.0)
                                for c in intent_candidates
                                if c.get("object_id") == nearest.get("object_id")
                            ),
                            0.0,
                        )
                    ),
                    4,
                ),
                "intent_grab_candidates": [
                    {
                        "object_id": str(c.get("object_id") or ""),
                        "label": str(c.get("label") or "object"),
                        "distance_m": round(float(c.get("distance_m", 0.0)), 4),
                        "effective_distance_m": round(float(c.get("effective_distance_m", 0.0)), 4),
                        "pred_contact_prob": round(float(c.get("pred_contact_prob", 0.0)), 4),
                        "pred_release_prob": round(float(c.get("pred_release_prob", 0.0)), 4),
                        "pred_target_tray_prob": round(float(c.get("pred_target_tray_prob", 0.5)), 4),
                        "pred_target_label": str(c.get("pred_target_label") or "target"),
                        "pred_target_id": str(c.get("pred_target_id") or ""),
                        "pred_target_motion_score": round(float(c.get("pred_target_motion_score", 0.0)), 4),
                        "pred_future_latent_confidence": round(float(c.get("pred_future_latent_confidence", 0.0)), 4),
                        "jepa_interaction_score": round(float(c.get("jepa_interaction_score", 0.0)), 4),
                    }
                    for c in top_grab
                ],
                "intent_place_candidates": [
                    {
                        "object_id": str(c.get("object_id") or ""),
                        "label": str(c.get("label") or "object"),
                        "distance_m": round(float(c.get("distance_m", 0.0)), 4),
                        "effective_distance_m": round(float(c.get("effective_distance_m", 0.0)), 4),
                        "pred_placement_prob": round(float(c.get("pred_placement_prob", 0.0)), 4),
                        "pred_future_latent_confidence": round(float(c.get("pred_future_latent_confidence", 0.0)), 4),
                        "jepa_interaction_score": round(float(c.get("jepa_interaction_score", 0.0)), 4),
                        "is_static_target": bool(c.get("is_static_target", False)),
                    }
                    for c in placement_candidates
                ],
            })
        self.hand_object_interactions = interactions

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
        objects = []
        for obj in self.objects.values():
            if obj.get("missing_since") is not None:
                continue
            item = obj.copy()
            support_label = self._canonical_support_label(item.get("support_target_label"))
            if support_label:
                item["region_membership"] = {
                    "type": "support_surface",
                    "label": support_label,
                    "target_id": item.get("support_target_id"),
                    "confidence": round(float((item.get("support_relation") or {}).get("support_score", 0.0) or 0.0), 4),
                }
            objects.append(item)
        objects.sort(
            key=lambda x: (
                x.get("label") == "cup",
                x.get("last_seen", 0.0),
                x.get("confidence", 0.0),
            ),
            reverse=True,
        )
        return objects

    def export_object_memory(self):
        now = time.time()
        objects = []
        for obj in self.objects.values():
            missing_since = obj.get("missing_since")
            visible = missing_since is None
            age_missing_s = 0.0
            if missing_since is not None:
                try:
                    age_missing_s = max(0.0, now - float(missing_since))
                except (TypeError, ValueError):
                    age_missing_s = 0.0
            item = obj.copy()
            item["visible"] = bool(visible)
            item["missing_seconds"] = round(float(age_missing_s), 3)
            item["last_known_position_3d"] = obj.get("position_3d", [0.0, 0.0, 0.0])
            item["last_known_bbox"] = obj.get("bbox", [])
            item["memory_ttl_seconds"] = round(float(max(0.0, self.object_memory_seconds - age_missing_s)), 3)
            support_label = self._canonical_support_label(item.get("support_target_label"))
            if support_label:
                item["region_membership"] = {
                    "type": "support_surface",
                    "label": support_label,
                    "target_id": item.get("support_target_id"),
                    "confidence": round(float((item.get("support_relation") or {}).get("support_score", 0.0) or 0.0), 4),
                }
            objects.append(item)
        objects.sort(
            key=lambda x: (
                not bool(x.get("visible", False)),
                -int(x.get("observation_count", 0) or 0),
                str(x.get("label", "")),
            )
        )
        return objects

    def export_objects_3d(self):
        return self.objects_3d

    def export_world_state(self):
        return {
            "camera_pose": self.camera_pose,
            "objects": self.export_objects(),
            "object_memory": self.export_object_memory(),
            "objects_3d": self.export_objects_3d(),
            "hands": self.hands,
            "hand_object_interactions": self.hand_object_interactions,
            "manipulation_events": self.manipulation_events[-20:],
            "learned_manipulation_events": self.learned_manipulation_events[-20:],
            "hand_trajectories": self._export_hand_trajectories(),
            "world_debug": self.world_debug,
            "static_targets": self._export_static_targets(),
            "sparse_map": self.sparse_map,
        }
