"""Optional hand detection + lightweight temporal tracking for embodied interaction.

If MediaPipe is available, this module returns smoothed hand centers in both
camera and world coordinates. If unavailable, it safely returns no hands.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Any

import numpy as np

# MediaPipe's native runtime can emit repeated Clearcut telemetry uploader
# errors on offline/desktop runs. Keep those logs quiet by default while still
# allowing MEDIAPIPE_LOG_LEVEL=0/1/2 for deeper debugging.
_mediapipe_log_level = os.environ.get("MEDIAPIPE_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", _mediapipe_log_level)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", _mediapipe_log_level)

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

try:
    import cv2
    import mediapipe as mp
except Exception:
    cv2 = None
    mp = None


def _clamp_int(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(value)))))


HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandTracker:
    def __init__(self):
        self.enabled = os.environ.get("HAND_TRACKING_ENABLED", "1").lower() not in {"0", "false", "no"}
        self.max_hands = int(os.environ.get("HAND_MAX_HANDS", "2"))
        self.model_complexity = int(os.environ.get("HAND_MODEL_COMPLEXITY", "1"))
        self.min_det_conf = float(os.environ.get("HAND_MIN_DET_CONF", "0.35"))
        self.min_track_conf = float(os.environ.get("HAND_MIN_TRACK_CONF", "0.35"))
        self.ema_alpha = float(os.environ.get("HAND_EMA_ALPHA", "0.55"))
        self.depth_scale_enabled = os.environ.get("HAND_DEPTH_SCALE_ENABLED", "1").lower() not in {"0", "false", "no"}
        self.depth_scale_radius_px = float(os.environ.get("HAND_DEPTH_SCALE_RADIUS_PX", "190"))
        self.depth_scale_min_anchors = int(os.environ.get("HAND_DEPTH_SCALE_MIN_ANCHORS", "4"))
        self.depth_scale_clamp_min = float(os.environ.get("HAND_DEPTH_SCALE_CLAMP_MIN", "0.65"))
        self.depth_scale_clamp_max = float(os.environ.get("HAND_DEPTH_SCALE_CLAMP_MAX", "1.50"))
        self.depth_sample_radius_px = int(os.environ.get("HAND_DEPTH_SAMPLE_RADIUS_PX", "4"))
        self.persist_missing_frames = int(os.environ.get("HAND_PERSIST_MISSING_FRAMES", "3"))
        self.prediction_max_step_m = float(os.environ.get("HAND_PREDICTION_MAX_STEP_M", "0.02"))
        self.prediction_conf_decay = float(os.environ.get("HAND_PREDICTION_CONF_DECAY", "0.85"))
        self.min_emit_confidence = float(os.environ.get("HAND_MIN_EMIT_CONFIDENCE", "0.15"))
        self.metric_prior_enabled = os.environ.get("HAND_METRIC_PRIOR_ENABLED", "1").lower() not in {"0", "false", "no"}
        self.metric_prior_palm_width_m = float(os.environ.get("HAND_METRIC_PRIOR_PALM_WIDTH_M", "0.085"))
        self.metric_prior_scale_clamp_min = float(os.environ.get("HAND_METRIC_PRIOR_SCALE_MIN", "0.92"))
        self.metric_prior_scale_clamp_max = float(os.environ.get("HAND_METRIC_PRIOR_SCALE_MAX", "1.10"))
        self.metric_prior_min_anchor_support = int(os.environ.get("HAND_METRIC_PRIOR_MIN_ANCHORS", "1"))
        self.force_side = os.environ.get("HAND_FORCE_SIDE", "").strip().lower()
        self.finger_radius_m = float(os.environ.get("HAND_FINGER_RADIUS_M", "0.009"))
        self.thumb_radius_m = float(os.environ.get("HAND_THUMB_RADIUS_M", "0.010"))
        self.palm_capsule_radius_m = float(os.environ.get("HAND_PALM_CAPSULE_RADIUS_M", "0.018"))
        self.depth_scale_temporal_fallback_enabled = os.environ.get("HAND_DEPTH_SCALE_TEMPORAL_FALLBACK", "1").lower() not in {"0", "false", "no"}
        self._last_global_depth_scale = 1.0
        self._smoothed_by_side: dict[str, np.ndarray] = {}
        self._tracks_by_side: dict[str, dict[str, Any]] = {}
        self._frame_index = 0
        self._mp_hands = None
        self._task_landmarker = None
        self.backend = "disabled"
        self.ready = False

        if not self.enabled:
            self.backend = "disabled"
            return
        if mp is None or cv2 is None:
            self.backend = "unavailable"
            return

        # Prefer legacy solutions API when available.
        if hasattr(mp, "solutions") and getattr(mp, "solutions", None) is not None:
            self._mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max(1, self.max_hands),
                min_detection_confidence=self.min_det_conf,
                min_tracking_confidence=self.min_track_conf,
                model_complexity=max(0, min(2, int(self.model_complexity))),
            )
            self.backend = "mediapipe-solutions"
            self.ready = True
            return

        # Fallback for newer MediaPipe builds (tasks API).
        self._task_landmarker = self._build_task_landmarker()
        if self._task_landmarker is not None:
            self.backend = "mediapipe-tasks"
            self.ready = True
            return

        self.backend = "unavailable"
        self.ready = False

    def _build_task_landmarker(self):
        try:
            from mediapipe.tasks.python import vision
            base_options_cls = mp.tasks.BaseOptions
            model_path = os.environ.get(
                "HAND_TASK_MODEL_PATH",
                os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task"),
            )
            model_path = os.path.abspath(model_path)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if not os.path.isfile(model_path):
                model_url = os.environ.get(
                    "HAND_TASK_MODEL_URL",
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                )
                urllib.request.urlretrieve(model_url, model_path)

            options = vision.HandLandmarkerOptions(
                base_options=base_options_cls(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max(1, self.max_hands),
                min_hand_detection_confidence=self.min_det_conf,
                min_hand_presence_confidence=self.min_track_conf,
                min_tracking_confidence=self.min_track_conf,
            )
            return vision.HandLandmarker.create_from_options(options)
        except Exception:
            return None

    def reset(self):
        self._smoothed_by_side = {}
        self._tracks_by_side = {}
        self._frame_index = 0

    def close(self):
        for handle in (self._mp_hands, self._task_landmarker):
            close_fn = getattr(handle, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    def _sample_depth(self, depth_map: np.ndarray, u: float, v: float) -> float:
        if depth_map is None or depth_map.size == 0:
            return 0.0
        h, w = depth_map.shape[:2]
        x = _clamp_int(u, 0, w - 1)
        y = _clamp_int(v, 0, h - 1)
        value = float(depth_map[y, x])
        if not np.isfinite(value):
            return 0.0
        return value

    def _sample_depth_window(self, depth_map: np.ndarray, u: float, v: float, radius: int = 0) -> float:
        if depth_map is None or depth_map.size == 0:
            return 0.0
        h, w = depth_map.shape[:2]
        r = max(0, int(radius))
        x = _clamp_int(u, 0, w - 1)
        y = _clamp_int(v, 0, h - 1)
        if r <= 0:
            return self._sample_depth(depth_map, u, v)
        x1 = max(0, x - r)
        x2 = min(w, x + r + 1)
        y1 = max(0, y - r)
        y2 = min(h, y + r + 1)
        crop = depth_map[y1:y2, x1:x2]
        if crop.size == 0:
            return self._sample_depth(depth_map, u, v)
        vals = crop[np.isfinite(crop)]
        if vals.size == 0:
            return 0.0
        return float(np.median(vals))

    def _lift_to_world(self, u: float, v: float, depth: float, intrinsics: dict, camera_pose: dict):
        fx = float(intrinsics.get("fx", 1.0))
        fy = float(intrinsics.get("fy", 1.0))
        cx = float(intrinsics.get("cx", 0.0))
        cy = float(intrinsics.get("cy", 0.0))
        x_cam = ((u - cx) / max(fx, 1e-6)) * depth
        y_cam = ((v - cy) / max(fy, 1e-6)) * depth
        z_cam = depth
        camera_point = np.asarray([x_cam, y_cam, z_cam], dtype=np.float32)

        camera_position_world = np.asarray(
            camera_pose.get("camera_position_world", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        rotation_wc = np.asarray(camera_pose.get("rotation_wc", np.eye(3, dtype=np.float32)), dtype=np.float32)
        if rotation_wc.shape != (3, 3):
            rotation_wc = np.eye(3, dtype=np.float32)
        world_point = rotation_wc @ camera_point + camera_position_world
        return camera_point, world_point

    def _build_depth_anchors(self, depth_map: np.ndarray, camera_pose: dict):
        anchors = []
        if depth_map is None or depth_map.size == 0:
            return anchors
        h, w = depth_map.shape[:2]
        sparse_points = camera_pose.get("local_sparse_map") or camera_pose.get("sparse_map") or camera_pose.get("persistent_map") or []
        camera_position_world = np.asarray(
            camera_pose.get("camera_position_world", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        rotation_wc = np.asarray(camera_pose.get("rotation_wc", np.eye(3, dtype=np.float32)), dtype=np.float32)
        if rotation_wc.shape != (3, 3):
            rotation_wc = np.eye(3, dtype=np.float32)
        rotation_cw = rotation_wc.T

        for point in sparse_points:
            image_xy = point.get("image_xy")
            pos_world = (
                point.get("position_world")
                or point.get("triangulated_position_world")
                or point.get("position_world_depth_prior")
            )
            if not (
                isinstance(image_xy, (list, tuple))
                and len(image_xy) >= 2
                and isinstance(pos_world, (list, tuple))
                and len(pos_world) >= 3
            ):
                continue
            u = float(image_xy[0])
            v = float(image_xy[1])
            if not (np.isfinite(u) and np.isfinite(v) and 0 <= u < w and 0 <= v < h):
                continue
            world = np.asarray([float(pos_world[0]), float(pos_world[1]), float(pos_world[2])], dtype=np.float32)
            camera = rotation_cw @ (world - camera_position_world)
            z_true = float(camera[2])
            if not np.isfinite(z_true) or z_true <= 0.0:
                continue
            z_depth = self._sample_depth(depth_map, u, v)
            if z_depth <= 0.0:
                continue
            anchors.append((u, v, z_depth, z_true))
        return anchors

    def _estimate_local_depth_scale(self, anchors, u: float, v: float) -> tuple[float, int, str]:
        if not self.depth_scale_enabled or not anchors:
            return 1.0, 0, "disabled"
        radius = max(10.0, float(self.depth_scale_radius_px))
        ratios = []
        global_ratios = []
        for au, av, z_depth, z_true in anchors:
            r = z_true / z_depth if z_depth > 0.0 else 0.0
            if np.isfinite(r) and r > 0:
                global_ratios.append(float(r))
            if ((au - u) ** 2 + (av - v) ** 2) ** 0.5 > radius:
                continue
            if z_depth <= 0.0:
                continue
            if np.isfinite(r) and r > 0:
                ratios.append(float(r))

        local_min = max(1, int(self.depth_scale_min_anchors))
        if len(ratios) >= local_min:
            scale = float(np.median(np.asarray(ratios, dtype=np.float32)))
            mode = "local"
            support = len(ratios)
        elif len(ratios) >= 1:
            scale = float(np.median(np.asarray(ratios, dtype=np.float32)))
            mode = "local-weak"
            support = len(ratios)
        elif len(global_ratios) >= local_min * 2:
            scale = float(np.median(np.asarray(global_ratios, dtype=np.float32)))
            mode = "global-fallback"
            support = len(global_ratios)
        elif len(global_ratios) >= max(2, local_min):
            scale = float(np.median(np.asarray(global_ratios, dtype=np.float32)))
            mode = "global-weak"
            support = len(global_ratios)
        elif self.depth_scale_temporal_fallback_enabled and np.isfinite(self._last_global_depth_scale):
            scale = float(self._last_global_depth_scale)
            mode = "temporal-fallback"
            support = 0
        else:
            return 1.0, len(ratios), "insufficient"

        scale = min(max(scale, float(self.depth_scale_clamp_min)), float(self.depth_scale_clamp_max))
        if mode in {"local", "local-weak", "global-fallback", "global-weak"}:
            self._last_global_depth_scale = float(0.85 * float(self._last_global_depth_scale) + 0.15 * scale)
        return scale, int(support), mode

    def _compute_hand_volume(self, landmarks_3d):
        points = []
        for pt in landmarks_3d:
            if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                xyz = np.asarray([float(pt[0]), float(pt[1]), float(pt[2])], dtype=np.float32)
                if np.all(np.isfinite(xyz)):
                    points.append(xyz)
        if len(points) < 4:
            return {"capsules": [], "aabb": None, "palm_radius": 0.0}

        pts = np.stack(points, axis=0)
        mn = np.min(pts, axis=0)
        mx = np.max(pts, axis=0)
        palm_idx = [0, 1, 5, 9, 13, 17]
        palm_points = []
        for idx in palm_idx:
            if idx < len(landmarks_3d):
                pt = landmarks_3d[idx]
                if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                    palm_points.append(np.asarray(pt[:3], dtype=np.float32))
        if len(palm_points) >= 2:
            palm = np.stack(palm_points, axis=0)
            centroid = np.mean(palm, axis=0)
            palm_radius = float(np.median(np.linalg.norm(palm - centroid, axis=1)))
        else:
            palm_radius = float(np.linalg.norm(mx - mn) * 0.06)

        capsules = []
        for a, b in HAND_BONES:
            if a >= len(landmarks_3d) or b >= len(landmarks_3d):
                continue
            p1 = landmarks_3d[a]
            p2 = landmarks_3d[b]
            if not (
                isinstance(p1, (list, tuple))
                and isinstance(p2, (list, tuple))
                and len(p1) >= 3
                and len(p2) >= 3
            ):
                continue
            v1 = np.asarray(p1[:3], dtype=np.float32)
            v2 = np.asarray(p2[:3], dtype=np.float32)
            seg_len = float(np.linalg.norm(v2 - v1))
            if not np.isfinite(seg_len) or seg_len <= 1e-5:
                continue
            if a == 0 or b == 0:
                radius = self.palm_capsule_radius_m
            elif a in {1, 2, 3, 4} or b in {1, 2, 3, 4}:
                radius = self.thumb_radius_m
            elif a in {5, 9, 13, 17} and b in {5, 9, 13, 17}:
                radius = self.palm_capsule_radius_m
            else:
                radius = self.finger_radius_m
            radius = float(np.clip(radius, 0.004, 0.026))
            capsules.append(
                {
                    "a": [round(float(v1[0]), 4), round(float(v1[1]), 4), round(float(v1[2]), 4)],
                    "b": [round(float(v2[0]), 4), round(float(v2[1]), 4), round(float(v2[2]), 4)],
                    "r": round(radius, 4),
                }
            )

        return {
            "capsules": capsules,
            "aabb": {
                "min": [round(float(mn[0]), 4), round(float(mn[1]), 4), round(float(mn[2]), 4)],
                "max": [round(float(mx[0]), 4), round(float(mx[1]), 4), round(float(mx[2]), 4)],
            },
            "palm_radius": round(max(0.0, palm_radius), 4),
        }

    def _apply_metric_hand_prior(self, center_world: np.ndarray, landmarks_3d, anchor_support: int = 0):
        if not self.metric_prior_enabled:
            return center_world, landmarks_3d, 1.0
        if int(anchor_support) < max(1, int(self.metric_prior_min_anchor_support)):
            return center_world, landmarks_3d, 1.0
        if not isinstance(landmarks_3d, list) or len(landmarks_3d) < 18:
            return center_world, landmarks_3d, 1.0
        p5 = landmarks_3d[5] if 5 < len(landmarks_3d) else None
        p17 = landmarks_3d[17] if 17 < len(landmarks_3d) else None
        p0 = landmarks_3d[0] if 0 < len(landmarks_3d) else None
        if not (
            isinstance(p5, (list, tuple)) and len(p5) >= 3 and
            isinstance(p17, (list, tuple)) and len(p17) >= 3 and
            isinstance(p0, (list, tuple)) and len(p0) >= 3
        ):
            return center_world, landmarks_3d, 1.0
        a = np.asarray(p5[:3], dtype=np.float32)
        b = np.asarray(p17[:3], dtype=np.float32)
        wrist = np.asarray(p0[:3], dtype=np.float32)
        width = float(np.linalg.norm(a - b))
        if not np.isfinite(width) or width <= 1e-4:
            return center_world, landmarks_3d, 1.0
        target = max(0.04, float(self.metric_prior_palm_width_m))
        scale = target / width
        scale = float(np.clip(scale, self.metric_prior_scale_clamp_min, self.metric_prior_scale_clamp_max))
        if abs(scale - 1.0) < 0.02:
            return center_world, landmarks_3d, 1.0
        scaled = []
        for pt in landmarks_3d:
            if not (isinstance(pt, (list, tuple)) and len(pt) >= 3):
                scaled.append(pt)
                continue
            p = np.asarray(pt[:3], dtype=np.float32)
            q = wrist + (p - wrist) * scale
            scaled.append([round(float(q[0]), 4), round(float(q[1]), 4), round(float(q[2]), 4)])
        center_scaled = wrist + (center_world - wrist) * scale
        return center_scaled, scaled, scale

    def _update_track(self, hand: dict, side: str):
        now = self._frame_index
        center = np.asarray(hand["center_3d"], dtype=np.float32)
        track = self._tracks_by_side.get(side)
        velocity = np.zeros(3, dtype=np.float32)
        if track is not None:
            prev_center = np.asarray(track.get("center_3d", center), dtype=np.float32)
            dt = max(1.0, float(now - int(track.get("frame_idx", now - 1))))
            velocity = (center - prev_center) / dt
            step = float(np.linalg.norm(velocity))
            vmax = max(0.001, float(self.prediction_max_step_m))
            if step > vmax:
                velocity = velocity * (vmax / step)
        self._tracks_by_side[side] = {
            "frame_idx": now,
            "center_3d": hand["center_3d"],
            "landmarks_3d": hand.get("landmarks_3d", []),
            "landmarks_px": hand.get("landmarks_px", []),
            "pixel_center": hand.get("pixel_center", [0.0, 0.0]),
            "image_norm_center": hand.get("image_norm_center", [0.0, 0.0]),
            "confidence": float(hand.get("confidence", 0.0) or 0.0),
            "velocity_3d": [float(velocity[0]), float(velocity[1]), float(velocity[2])],
            "depth": float(hand.get("depth", 0.0) or 0.0),
            "depth_scale": float(hand.get("depth_scale", 1.0) or 1.0),
            "depth_scale_support": int(hand.get("depth_scale_support", 0) or 0),
        }

    def _depth_evidence_for_track(self, track: dict, depth_map: np.ndarray):
        if depth_map is None or depth_map.size == 0:
            return {"score": 0.0, "supported": False, "reason": "no-depth"}
        pts = track.get("landmarks_px", [])
        if not isinstance(pts, list) or len(pts) < 6:
            return {"score": 0.0, "supported": False, "reason": "no-landmarks"}
        h, w = depth_map.shape[:2]
        xs, ys = [], []
        for pt in pts:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x = float(pt[0])
                y = float(pt[1])
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x)
                    ys.append(y)
        if len(xs) < 6:
            return {"score": 0.0, "supported": False, "reason": "invalid-landmarks"}
        x1 = _clamp_int(min(xs), 0, w - 1)
        y1 = _clamp_int(min(ys), 0, h - 1)
        x2 = _clamp_int(max(xs), 0, w - 1)
        y2 = _clamp_int(max(ys), 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            return {"score": 0.0, "supported": False, "reason": "empty-bbox"}
        pad = max(4, int(0.35 * max(x2 - x1, y2 - y1)))
        ix1, iy1, ix2, iy2 = x1, y1, x2, y2
        ox1 = _clamp_int(x1 - pad, 0, w - 1)
        oy1 = _clamp_int(y1 - pad, 0, h - 1)
        ox2 = _clamp_int(x2 + pad, 0, w - 1)
        oy2 = _clamp_int(y2 + pad, 0, h - 1)
        inside = np.asarray(depth_map[iy1:iy2 + 1, ix1:ix2 + 1], dtype=np.float32).reshape(-1)
        outer = np.asarray(depth_map[oy1:oy2 + 1, ox1:ox2 + 1], dtype=np.float32)
        ring = outer.copy()
        rx1, ry1 = ix1 - ox1, iy1 - oy1
        rx2, ry2 = rx1 + (ix2 - ix1) + 1, ry1 + (iy2 - iy1) + 1
        if 0 <= ry1 < ry2 <= ring.shape[0] and 0 <= rx1 < rx2 <= ring.shape[1]:
            ring[ry1:ry2, rx1:rx2] = np.nan
        ring = ring.reshape(-1)
        inside = inside[np.isfinite(inside) & (inside > 0.0)]
        ring = ring[np.isfinite(ring) & (ring > 0.0)]
        if inside.size < 12 or ring.size < 12:
            return {"score": 0.0, "supported": False, "reason": "insufficient-depth"}
        inside_med = float(np.median(inside))
        ring_med = float(np.median(ring))
        contrast = abs(ring_med - inside_med)
        expected = float(track.get("depth", 0.0) or 0.0)
        depth_consistency = 0.0
        if expected > 0.0:
            depth_consistency = max(0.0, 1.0 - abs(inside_med - expected) / max(0.25, expected * 0.20))
        contrast_score = min(1.0, contrast / 0.08)
        score = max(0.0, min(1.0, 0.55 * depth_consistency + 0.45 * contrast_score))
        return {
            "score": round(float(score), 3),
            "supported": bool(score >= 0.35),
            "inside_depth": round(float(inside_med), 4),
            "ring_depth": round(float(ring_med), 4),
            "contrast_m": round(float(contrast), 4),
        }

    def _predict_missing_hands(self, detected_sides: set[str], depth_map: np.ndarray | None = None):
        now = self._frame_index
        predicted = []
        for side, track in self._tracks_by_side.items():
            if side in detected_sides:
                continue
            missing = int(now - int(track.get("frame_idx", now)))
            if missing <= 0 or missing > max(0, int(self.persist_missing_frames)):
                continue
            base_conf = float(track.get("confidence", 0.0) or 0.0)
            conf = base_conf * (float(self.prediction_conf_decay) ** missing)
            if conf < float(self.min_emit_confidence):
                continue

            vel = np.asarray(track.get("velocity_3d", [0.0, 0.0, 0.0]), dtype=np.float32)
            center = np.asarray(track.get("center_3d", [0.0, 0.0, 0.0]), dtype=np.float32) + vel * float(missing)
            lm3d = track.get("landmarks_3d", [])
            lm3d_pred = []
            for pt in lm3d:
                if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                    p = np.asarray(pt[:3], dtype=np.float32) + vel * float(missing)
                    lm3d_pred.append([round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)])
                else:
                    lm3d_pred.append(pt)

            hand = {
                "id": f"hand_{side}_pred",
                "side": side,
                "confidence": round(conf, 3),
                "pixel_center": track.get("pixel_center", [0.0, 0.0]),
                "image_norm_center": track.get("image_norm_center", [0.0, 0.0]),
                "depth": round(float(track.get("depth", 0.0) or 0.0), 4),
                "depth_scale": round(float(track.get("depth_scale", 1.0) or 1.0), 4),
                "depth_scale_support": int(track.get("depth_scale_support", 0) or 0),
                "position_camera_3d": [0.0, 0.0, 0.0],
                "center_3d": [round(float(center[0]), 4), round(float(center[1]), 4), round(float(center[2]), 4)],
                "landmarks_px": track.get("landmarks_px", []),
                "landmarks_3d": lm3d_pred,
                "predicted": True,
                "missing_frames": int(missing),
                "track_age": int(now - int(track.get("frame_idx", now))),
            }
            evidence = self._depth_evidence_for_track(track, depth_map)
            hand["depth_evidence"] = evidence
            if evidence.get("supported"):
                hand["confidence"] = round(min(0.95, float(hand["confidence"]) + 0.12 * float(evidence.get("score", 0.0))), 3)
            hand["volume_3d"] = self._compute_hand_volume(lm3d_pred)
            predicted.append(hand)
        return predicted

    def detect(self, frame_bgr: np.ndarray, depth_map: np.ndarray, intrinsics: dict, camera_pose: dict):
        self._frame_index += 1
        debug = {
            "backend": self.backend,
            "ready": bool(self.ready),
            "hands_detected": 0,
        }
        if not self.ready or frame_bgr is None or frame_bgr.size == 0:
            return [], debug

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        h, w = frame_bgr.shape[:2]
        depth_anchors = self._build_depth_anchors(depth_map, camera_pose)
        debug["depth_anchor_count"] = int(len(depth_anchors))
        debug["depth_scale_enabled"] = bool(self.depth_scale_enabled)
        debug["prediction_enabled"] = bool(self.persist_missing_frames > 0)
        scale_modes = {
            "local": 0,
            "local-weak": 0,
            "global-fallback": 0,
            "global-weak": 0,
            "temporal-fallback": 0,
            "insufficient": 0,
            "disabled": 0,
        }
        hands = []
        if self.backend == "mediapipe-solutions":
            result = self._mp_hands.process(rgb)
            if not result or not result.multi_hand_landmarks:
                predicted_hands = self._predict_missing_hands(set(), depth_map=depth_map)
                debug["hands_detected"] = int(len(predicted_hands))
                debug["hands_predicted"] = int(len(predicted_hands))
                debug["depth_scale_modes"] = scale_modes
                return predicted_hands, debug
            handedness_list = result.multi_handedness or []
            landmarks_iter = []
            for idx, landmarks in enumerate(result.multi_hand_landmarks):
                side = "unknown"
                side_score = 0.0
                if idx < len(handedness_list):
                    cls = handedness_list[idx].classification[0]
                    side = str(cls.label).lower()
                    side_score = float(cls.score)
                if self.force_side in {"left", "right"}:
                    side = self.force_side
                lm = [(float(item.x), float(item.y)) for item in landmarks.landmark]
                landmarks_iter.append((idx, side, side_score, lm))
        else:
            # mediapipe-tasks backend
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._task_landmarker.detect(mp_image)
            if not result or not getattr(result, "hand_landmarks", None):
                predicted_hands = self._predict_missing_hands(set(), depth_map=depth_map)
                debug["hands_detected"] = int(len(predicted_hands))
                debug["hands_predicted"] = int(len(predicted_hands))
                debug["depth_scale_modes"] = scale_modes
                return predicted_hands, debug
            landmarks_iter = []
            handedness_list = getattr(result, "handedness", []) or []
            for idx, lm_list in enumerate(result.hand_landmarks):
                side = "unknown"
                side_score = 0.0
                if idx < len(handedness_list) and handedness_list[idx]:
                    cat = handedness_list[idx][0]
                    side = str(getattr(cat, "category_name", "unknown")).lower()
                    side_score = float(getattr(cat, "score", 0.0) or 0.0)
                if self.force_side in {"left", "right"}:
                    side = self.force_side
                lm = [(float(item.x), float(item.y)) for item in lm_list]
                landmarks_iter.append((idx, side, side_score, lm))

        for idx, side, side_score, lm in landmarks_iter:
            palm_idx = [0, 1, 5, 9, 13, 17]
            palm = [lm[i] for i in palm_idx if i < len(lm)]
            if not palm:
                continue
            xs = [p[0] * w for p in palm]
            ys = [p[1] * h for p in palm]
            u = float(np.median(xs))
            v = float(np.median(ys))
            depth_scale, scale_support, scale_mode = self._estimate_local_depth_scale(depth_anchors, u, v)
            depth = self._sample_depth_window(depth_map, u, v, radius=self.depth_sample_radius_px) * depth_scale
            if depth <= 0.0:
                continue

            camera_point, world_point = self._lift_to_world(u, v, depth, intrinsics, camera_pose)

            landmarks_3d = []
            for px, py in lm:
                lu = float(px * w)
                lv = float(py * h)
                lscale, lsupport, _ = self._estimate_local_depth_scale(depth_anchors, lu, lv)
                if lsupport < max(1, int(self.depth_scale_min_anchors // 2)):
                    lscale = depth_scale
                ldepth = self._sample_depth_window(depth_map, lu, lv, radius=max(1, self.depth_sample_radius_px // 2)) * lscale
                # Keep per-joint lift close to palm depth to avoid extreme 3D spikes.
                if depth > 0.0 and ldepth > 0.0:
                    ldepth = float(np.clip(ldepth, depth * 0.6, depth * 1.65))
                if ldepth <= 0.0:
                    landmarks_3d.append(None)
                    continue
                _, wpt = self._lift_to_world(lu, lv, ldepth, intrinsics, camera_pose)
                landmarks_3d.append(np.round(wpt, 4).tolist())

            valid_landmarks = []
            for pt in landmarks_3d:
                if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                    arr = np.asarray(pt[:3], dtype=np.float32)
                    if np.isfinite(arr).all():
                        valid_landmarks.append(arr)
            if len(valid_landmarks) >= 6:
                lm_center = np.median(np.stack(valid_landmarks, axis=0), axis=0)
                # Bias center toward lifted landmarks so 3D hand and bones stay in the same frame.
                world_point = 0.35 * world_point + 0.65 * lm_center
            world_point, landmarks_3d, metric_scale = self._apply_metric_hand_prior(
                world_point,
                landmarks_3d,
                anchor_support=scale_support,
            )
            previous = self._smoothed_by_side.get(side)
            if previous is None:
                smoothed = world_point
            else:
                alpha = np.clip(self.ema_alpha, 0.0, 1.0)
                jump = float(np.linalg.norm(world_point - previous))
                weak_support = scale_support < max(2, int(self.depth_scale_min_anchors // 2))
                # If current scale support is weak and jump is large, avoid dragging stale pose.
                if weak_support and jump > 0.55:
                    smoothed = world_point
                else:
                    smoothed = (1.0 - alpha) * previous + alpha * world_point
            self._smoothed_by_side[side] = smoothed

            hands.append(
                {
                    "id": f"hand_{side}_{idx}",
                    "side": side,
                    "confidence": round(side_score, 3),
                    "pixel_center": [round(u, 1), round(v, 1)],
                    "image_norm_center": [round(float(u / max(w, 1)), 4), round(float(v / max(h, 1)), 4)],
                    "image_size": [int(w), int(h)],
                    "depth": round(float(depth), 4),
                    "depth_scale": round(float(depth_scale), 4),
                    "depth_scale_support": int(scale_support),
                    "depth_scale_mode": scale_mode,
                    "metric_prior_scale": round(float(metric_scale), 3),
                    "position_camera_3d": np.round(camera_point, 4).tolist(),
                    "center_3d": np.round(smoothed, 4).tolist(),
                    "landmarks_px": [[round(float(px * w), 1), round(float(py * h), 1)] for px, py in lm],
                    "landmarks_3d": landmarks_3d,
                    "predicted": False,
                    "missing_frames": 0,
                    "track_age": 0,
                }
            )
            scale_modes[scale_mode] = int(scale_modes.get(scale_mode, 0) + 1)

        detected_sides: set[str] = set()
        for hand in hands:
            side = str(hand.get("side", "unknown")).lower()
            detected_sides.add(side)
            self._update_track(hand, side)
            hand["volume_3d"] = self._compute_hand_volume(hand.get("landmarks_3d", []))

        predicted_hands = self._predict_missing_hands(detected_sides, depth_map=depth_map)
        if predicted_hands:
            hands.extend(predicted_hands)

        debug["hands_detected"] = len(hands)
        debug["hands_predicted"] = int(len(predicted_hands))
        debug["depth_scale_modes"] = scale_modes
        return hands, debug
