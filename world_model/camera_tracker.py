"""Sparse egomotion and landmark tracking for the embodied-agent prototype.

This tracker uses sparse optical flow to estimate frame-to-frame motion
and maintain a lightweight persistent landmark map in a shared world
frame. It is not full SLAM, but it provides a clean first step toward a
SLAM-like embodied memory: sparse tracked features, accumulated camera
translation, and a persistent 3D point set built from multi-view depth.
"""

from __future__ import annotations

import cv2
import numpy as np
import os
import sys
from pathlib import Path

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None

MAX_PNP_REPROJECTION_ERROR = float(os.environ.get("SLAM_MAX_PNP_REPROJECTION_ERROR", "10.0"))
MAX_PNP_POSITION_JUMP = float(os.environ.get("SLAM_MAX_PNP_POSITION_JUMP", "0.75"))
MAX_MISSING_FRAMES = int(os.environ.get("SLAM_MAX_MISSING_FRAMES", "220"))
MAX_LANDMARKS = int(os.environ.get("SLAM_MAX_LANDMARKS", "1800"))
MIN_QUALITY_FOR_PERSISTENCE = float(os.environ.get("SLAM_MIN_QUALITY_FOR_PERSISTENCE", "0.2"))
MIN_HITS_FOR_PERSISTENCE = int(os.environ.get("SLAM_MIN_HITS_FOR_PERSISTENCE", "2"))
XFEAT_TOP_K = int(os.environ.get("SLAM_XFEAT_TOP_K", "768"))
XFEAT_UPDATE_EVERY = int(os.environ.get("SLAM_XFEAT_UPDATE_EVERY", "4"))
XFEAT_MATCH_RADIUS_PX = float(os.environ.get("SLAM_XFEAT_MATCH_RADIUS_PX", "12.0"))
FEATURE_BACKEND = os.environ.get("FEATURE_BACKEND", "hybrid").lower()
MIN_REOBSERVATION_SIMILARITY = 0.78
MAX_REOBSERVATION_DISTANCE_PX = 90.0
GEOMETRY_REOBSERVATION_SIMILARITY = float(os.environ.get("SLAM_GEOMETRY_REOBSERVATION_SIMILARITY", "0.72"))
GEOMETRY_REOBSERVATION_DISTANCE_PX = float(os.environ.get("SLAM_GEOMETRY_REOBSERVATION_DISTANCE_PX", "140.0"))
PROTECTED_GEOMETRY_MIN_HITS = int(os.environ.get("SLAM_PROTECTED_GEOMETRY_MIN_HITS", "6"))
PROTECTED_GEOMETRY_MISSING_MULTIPLIER = float(os.environ.get("SLAM_PROTECTED_GEOMETRY_MISSING_MULTIPLIER", "2.0"))
FEATURE_QUALITY_LEVEL = float(os.environ.get("SLAM_FEATURE_QUALITY_LEVEL", "0.004"))
FEATURE_MIN_DISTANCE = int(os.environ.get("SLAM_FEATURE_MIN_DISTANCE", "6"))
FEATURE_BLOCK_SIZE = 5
FEATURE_GRID_COLS = 4
FEATURE_GRID_ROWS = 3
KEYFRAME_MIN_INTERVAL = int(os.environ.get("SLAM_KEYFRAME_MIN_INTERVAL", "8"))
KEYFRAME_MIN_TRANSLATION = float(os.environ.get("SLAM_KEYFRAME_MIN_TRANSLATION", "0.018"))
KEYFRAME_MIN_VISIBLE = int(os.environ.get("SLAM_KEYFRAME_MIN_VISIBLE", "28"))
MAX_KEYFRAMES = int(os.environ.get("SLAM_MAX_KEYFRAMES", "32"))
MAX_OBSERVATIONS_PER_LANDMARK = int(os.environ.get("SLAM_MAX_OBSERVATIONS_PER_LANDMARK", "12"))
STABLE_MIN_OBSERVATIONS = 2
STABLE_MAX_REPROJECTION_ERROR = 8.0
STABLE_2D_MIN_HITS = 8
GEOMETRY_VERIFIED_MIN_OBSERVATIONS = 2
GEOMETRY_VERIFIED_MAX_REPROJECTION_ERROR = 10.0
GEOMETRY_VERIFIED_MIN_RAY_ANGLE_DEG = 0.5
TRIANGULATED_MIN_RAY_ANGLE_DEG = 0.8
TRIANGULATED_MIN_BASELINE = float(os.environ.get("SLAM_TRIANGULATED_MIN_BASELINE", "0.02"))
TRIANGULATED_MAX_REPROJECTION_ERROR = 10.0
TRIANGULATED_MAX_DEPTH_ERROR = 0.45
TRIANGULATION_POSITION_BLEND = float(os.environ.get("SLAM_TRIANGULATION_POSITION_BLEND", "0.40"))
BA_LITE_MIN_OBSERVATIONS = 2
BA_LITE_MAX_INITIAL_ERROR = 25.0
BA_LITE_BLEND = 0.35
BA_LITE_MAX_UPDATES_PER_KEYFRAME = int(os.environ.get("SLAM_BA_LITE_MAX_UPDATES_PER_KEYFRAME", "120"))
BA_LITE_MIN_BASELINE = 0.03
BA_LITE_MIN_RAY_ANGLE_DEG = 1.0
SLIDING_BA_MAX_KEYFRAMES = int(os.environ.get("SLAM_SLIDING_BA_MAX_KEYFRAMES", "5"))
SLIDING_BA_MAX_LANDMARKS = int(os.environ.get("SLAM_SLIDING_BA_MAX_LANDMARKS", "90"))
SLIDING_BA_MAX_RESIDUAL_OBS = int(os.environ.get("SLAM_SLIDING_BA_MAX_RESIDUAL_OBS", "320"))
SLIDING_BA_MIN_OBSERVATIONS = 2
SLIDING_BA_MIN_KEYFRAMES = 2
SLIDING_BA_MAX_NFEV = 35
SLIDING_BA_RUN_EVERY_N_KEYFRAMES = 2
SLIDING_BA_ROTATION_PRIOR_WEIGHT = 2.0
SLIDING_BA_TRANSLATION_PRIOR_WEIGHT = 20.0
SLIDING_BA_POINT_PRIOR_WEIGHT = 2.0
SLIDING_BA_DEPTH_PRIOR_WEIGHT = 0.0
SLIDING_BA_MAX_LANDMARK_ERROR = 12.0
SLIDING_BA_MIN_LANDMARK_HITS = 4
COVISIBILITY_MIN_SHARED = int(os.environ.get("SLAM_COVISIBILITY_MIN_SHARED", "6"))
LOCAL_MAP_MAX_KEYFRAMES = int(os.environ.get("SLAM_LOCAL_MAP_MAX_KEYFRAMES", "6"))
LOCAL_MAP_MAX_LANDMARKS = int(os.environ.get("SLAM_LOCAL_MAP_MAX_LANDMARKS", "320"))
MIN_LOCAL_PNP_ANCHORS = int(os.environ.get("SLAM_MIN_LOCAL_PNP_ANCHORS", "4"))
VISIBLE_MAP_EXPORT_LIMIT = int(os.environ.get("SLAM_VISIBLE_MAP_EXPORT_LIMIT", "480"))
GEOMETRY_EXPORT_MIN_OBSERVATIONS = 2
GEOMETRY_EXPORT_MAX_REPROJECTION_ERROR = 9.0
RELATIVE_POSE_MIN_INLIERS = 20
RELATIVE_POSE_MAX_TRANSLATION = 0.08
LK_FORWARD_BACKWARD_MAX_ERROR_PX = 2.0
PNP_MIN_INLIERS = int(os.environ.get("SLAM_PNP_MIN_INLIERS", "6"))
PNP_LOCK_MIN_INLIERS = int(os.environ.get("SLAM_PNP_LOCK_MIN_INLIERS", "12"))
PNP_LOCK_MAX_REPROJECTION_ERROR = float(os.environ.get("SLAM_PNP_LOCK_MAX_REPROJECTION_ERROR", "6.0"))
ESSENTIAL_FALLBACK_SCALE_DAMPING = float(os.environ.get("SLAM_ESSENTIAL_FALLBACK_SCALE_DAMPING", "0.55"))
ESSENTIAL_FALLBACK_MAX_TRANSLATION = float(os.environ.get("SLAM_ESSENTIAL_FALLBACK_MAX_TRANSLATION", "0.03"))
ESSENTIAL_ROTATION_ONLY_AFTER_MISSED_PNP = int(
    os.environ.get("SLAM_ESSENTIAL_ROTATION_ONLY_AFTER_MISSED_PNP", "10")
)
ESSENTIAL_KEYFRAME_MAX_FRAMES_SINCE_PNP = int(
    os.environ.get("SLAM_ESSENTIAL_KEYFRAME_MAX_FRAMES_SINCE_PNP", "999")
)
ESSENTIAL_KEYFRAME_MIN_INLIERS = int(os.environ.get("SLAM_ESSENTIAL_KEYFRAME_MIN_INLIERS", "0"))
PERSISTENT_MAP_EXPORT_LIMIT = int(os.environ.get("SLAM_PERSISTENT_MAP_EXPORT_LIMIT", "800"))
PNP_MAX_ANCHORS = int(os.environ.get("SLAM_PNP_MAX_ANCHORS", "260"))
SEMANTIC_DYNAMIC_ON = 0.58
SEMANTIC_DYNAMIC_OFF = 0.42
SEMANTIC_DYNAMIC_EMA = 0.2
MOTION_OUTLIER_Z_ON = float(os.environ.get("SLAM_MOTION_OUTLIER_Z_ON", "3.5"))
MOTION_OUTLIER_Z_FULL = float(os.environ.get("SLAM_MOTION_OUTLIER_Z_FULL", "8.0"))
DYNAMIC_EVIDENCE_GATE = float(os.environ.get("SLAM_DYNAMIC_EVIDENCE_GATE", "0.90"))
HIGH_DYNAMIC_SCORE = float(os.environ.get("SLAM_HIGH_DYNAMIC_SCORE", "0.85"))
DYNAMIC_EVIDENCE_SEMANTIC_WEIGHT = float(os.environ.get("SLAM_DYNAMIC_SEMANTIC_WEIGHT", "0.25"))
DYNAMIC_EVIDENCE_MOTION_WEIGHT = float(os.environ.get("SLAM_DYNAMIC_MOTION_WEIGHT", "1.00"))


class XFeatDescriptorBackend:
    """Lazy XFeat descriptor adapter for landmark re-observation."""

    def __init__(self, top_k: int = XFEAT_TOP_K):
        self.top_k = top_k
        self.available = False
        self.status = "not-loaded"
        self.last_error = None
        self.device = "cpu"
        self._model = None
        self._torch = None
        self._cache_id = None
        self._cache_keypoints = None
        self._cache_descriptors = None
        self._load()

    def _load(self):
        xfeat_repo = self._resolve_xfeat_repo()
        if xfeat_repo and os.path.isdir(xfeat_repo) and xfeat_repo not in sys.path:
            sys.path.insert(0, xfeat_repo)

        try:
            import torch
            from modules.xfeat import XFeat

            self._torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = XFeat()
            if hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
            if hasattr(self._model, "eval"):
                self._model.eval()
            self.available = True
            self.status = "ready"
        except Exception as exc:
            self.available = False
            self.status = "unavailable"
            self.last_error = str(exc)

    def _resolve_xfeat_repo(self) -> str | None:
        configured = os.environ.get("XFEAT_REPO", "").strip()
        if configured and os.path.isdir(configured):
            return configured

        here = Path(__file__).resolve()
        repo_root = here.parent.parent
        candidates = [
            repo_root / "third_party" / "xfeat" / "accelerated_features",
            repo_root / "xfeat" / "accelerated_features",
            repo_root.parent / "xfeat" / "accelerated_features",
            Path("C:/code/xfeat/accelerated_features"),
            Path("C:/tonyt-ai/xfeat/accelerated_features"),
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate)
        return None

    def _compute(self, gray: np.ndarray):
        if not self.available or self._model is None or self._torch is None:
            return None, None

        cache_id = id(gray)
        if self._cache_id == cache_id:
            return self._cache_keypoints, self._cache_descriptors

        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
        tensor = tensor.to(self.device)

        with self._torch.no_grad():
            output = self._model.detectAndCompute(tensor, top_k=self.top_k)[0]

        keypoints = output.get("keypoints")
        descriptors = output.get("descriptors")
        if keypoints is None or descriptors is None:
            self._cache_id = cache_id
            self._cache_keypoints = None
            self._cache_descriptors = None
            return None, None

        keypoints = keypoints.detach().float().cpu().numpy()
        descriptors = descriptors.detach().float().cpu().numpy()
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / np.maximum(norms, 1e-6)

        self._cache_id = cache_id
        self._cache_keypoints = keypoints
        self._cache_descriptors = descriptors
        return keypoints, descriptors

    def describe_at(self, gray: np.ndarray, u: float, v: float) -> np.ndarray | None:
        keypoints, descriptors = self._compute(gray)
        if keypoints is None or descriptors is None or len(keypoints) == 0:
            return None

        deltas = keypoints[:, :2] - np.array([u, v], dtype=np.float32)
        distances = np.linalg.norm(deltas, axis=1)
        idx = int(np.argmin(distances))
        if float(distances[idx]) > XFEAT_MATCH_RADIUS_PX:
            return None
        return descriptors[idx].copy()

    def keypoints(self, gray: np.ndarray, limit: int) -> np.ndarray | None:
        keypoints, _ = self._compute(gray)
        if keypoints is None or len(keypoints) == 0 or limit <= 0:
            return None
        return keypoints[:limit, :2].astype(np.float32)

    def debug_info(self) -> dict:
        return {
            "backend": "xfeat",
            "status": self.status,
            "device": self.device,
            "top_k": self.top_k,
            "update_every": XFEAT_UPDATE_EVERY,
            "last_error": self.last_error,
        }


class OrbFeatureBackend:
    """Fast CPU ORB feature adapter used for keypoint seeding/debug A-B tests."""

    def __init__(self, top_k: int = XFEAT_TOP_K):
        self.top_k = top_k
        self.status = "ready"
        self.last_error = None
        self._orb = cv2.ORB_create(
            nfeatures=max(top_k, 300),
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=12,
            fastThreshold=8,
        )

    def keypoints(self, gray: np.ndarray, limit: int) -> np.ndarray | None:
        if limit <= 0:
            return None
        try:
            keypoints = self._orb.detect(gray, None)
        except cv2.error as exc:
            self.last_error = str(exc)
            self.status = "error"
            return None
        if not keypoints:
            return None
        keypoints = sorted(keypoints, key=lambda item: item.response, reverse=True)[:limit]
        return np.asarray([kp.pt for kp in keypoints], dtype=np.float32)

    def debug_info(self) -> dict:
        return {
            "backend": "orb",
            "status": self.status,
            "top_k": self.top_k,
            "last_error": self.last_error,
        }


class CameraTracker:
    """Estimate camera motion and maintain a sparse landmark map."""

    def __init__(self, max_points: int = 320, min_points: int = 100):
        max_points = int(os.environ.get("SLAM_MAX_TRACK_POINTS", str(max_points)))
        min_points = int(os.environ.get("SLAM_MIN_TRACK_POINTS", str(min_points)))
        self.max_points = max_points
        self.min_points = min_points
        self.prev_gray = None
        self.prev_points = None
        self.prev_track_ids = []
        self.frame_index = 0
        self.next_track_id = 1
        self.camera_position_world = np.zeros(3, dtype=np.float32)
        self.rotation_wc = np.eye(3, dtype=np.float32)
        self.landmarks = {}
        self.descriptor_backend = XFeatDescriptorBackend()
        self.orb_backend = OrbFeatureBackend()
        self.keyframes = []
        self.next_keyframe_id = 1
        self.last_keyframe_frame = 0
        self.last_keyframe_position = None
        self.covisibility_graph = {}
        self.local_keyframe_ids = []
        self.local_landmark_ids = set()
        self.last_pnp_anchor_scope = "none"
        self.frames_since_pnp_lock = 0
        self.essential_translation_scale = 0.06
        self.ba_lite_stats = {
            "runs": 0,
            "landmarks_refined": 0,
            "last_refined": 0,
            "last_skipped_low_parallax": 0,
            "last_mean_error_before": None,
            "last_mean_error_after": None,
        }
        self.sliding_ba_stats = {
            "available": least_squares is not None,
            "runs": 0,
            "last_status": "not-run",
            "last_keyframes": 0,
            "last_landmarks": 0,
            "last_observations": 0,
            "last_candidates": 0,
            "last_rejected": 0,
            "last_cost_before": None,
            "last_cost_after": None,
        }
        self.triangulation_stats = {
            "candidates": 0,
            "accepted": 0,
            "rejected_observations": 0,
            "rejected_baseline": 0,
            "rejected_angle": 0,
            "rejected_solver": 0,
            "rejected_reprojection": 0,
            "rejected_depth": 0,
            "depth_disagreement": 0,
        }
        self.lifecycle_stats = {
            "created": 0,
            "updated": 0,
            "reobserved": 0,
            "descriptor_reassociated": 0,
            "marked_missing": 0,
            "pruned": 0,
        }

    def _triangulated_landmark_count(self) -> int:
        return sum(1 for item in self.landmarks.values() if item.get("is_triangulated"))

    def _stable_2d_landmark_count(self) -> int:
        return sum(1 for item in self.landmarks.values() if item.get("is_2d_stable"))

    def _geometry_verified_landmark_count(self) -> int:
        return sum(1 for item in self.landmarks.values() if item.get("is_geometry_verified"))

    def _dynamic_landmark_count(self) -> int:
        return sum(1 for item in self.landmarks.values() if item.get("is_dynamic"))

    def _is_geometry_owned_landmark(self, landmark: dict) -> bool:
        """True when a landmark's world position should be owned by geometry, not per-frame depth."""
        return bool(
            landmark.get("is_triangulated")
            or landmark.get("is_geometry_verified")
            or landmark.get("ba_lite_refined")
            or landmark.get("sliding_ba_refined")
        )

    def _is_geometry_exportable_landmark(self, landmark: dict) -> bool:
        if not self._is_geometry_owned_landmark(landmark):
            return False
        if int(landmark.get("observation_count", 0)) < GEOMETRY_EXPORT_MIN_OBSERVATIONS:
            return False
        reproj = landmark.get("mean_reprojection_error")
        if reproj is None:
            return False
        return float(reproj) <= GEOMETRY_EXPORT_MAX_REPROJECTION_ERROR

    def _is_protected_geometry_landmark(self, landmark: dict) -> bool:
        return bool(
            self._is_geometry_owned_landmark(landmark)
            and int(landmark.get("hits", 0)) >= PROTECTED_GEOMETRY_MIN_HITS
            and (
                bool(landmark.get("is_triangulated"))
                or bool(landmark.get("is_geometry_verified"))
            )
        )

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.prev_track_ids = []
        self.frame_index = 0
        self.next_track_id = 1
        self.camera_position_world = np.zeros(3, dtype=np.float32)
        self.rotation_wc = np.eye(3, dtype=np.float32)
        self.landmarks = {}
        self.keyframes = []
        self.next_keyframe_id = 1
        self.last_keyframe_frame = 0
        self.last_keyframe_position = None
        self.covisibility_graph = {}
        self.local_keyframe_ids = []
        self.local_landmark_ids = set()
        self.last_pnp_anchor_scope = "none"
        self.frames_since_pnp_lock = 0
        self.essential_translation_scale = 0.06
        self.ba_lite_stats = {
            "runs": 0,
            "landmarks_refined": 0,
            "last_refined": 0,
            "last_skipped_low_parallax": 0,
            "last_mean_error_before": None,
            "last_mean_error_after": None,
        }
        self.sliding_ba_stats = {
            "available": least_squares is not None,
            "runs": 0,
            "last_status": "not-run",
            "last_keyframes": 0,
            "last_landmarks": 0,
            "last_observations": 0,
            "last_candidates": 0,
            "last_rejected": 0,
            "last_cost_before": None,
            "last_cost_after": None,
        }
        self.triangulation_stats = {
            "candidates": 0,
            "accepted": 0,
            "rejected_observations": 0,
            "rejected_baseline": 0,
            "rejected_angle": 0,
            "rejected_solver": 0,
            "rejected_reprojection": 0,
            "rejected_depth": 0,
            "depth_disagreement": 0,
        }
        self.lifecycle_stats = {
            "created": 0,
            "updated": 0,
            "reobserved": 0,
            "descriptor_reassociated": 0,
            "marked_missing": 0,
            "pruned": 0,
        }

    def _pose_dict(
        self,
        *,
        status: str,
        tracking_quality: float,
        image_shift_px,
        delta_translation_world,
        sparse_map,
        pose_source: str = "bootstrap",
        pnp_inliers: int = 0,
        pnp_reprojection_error: float | None = None,
    ) -> dict:
        rotation_cw = self.rotation_wc.T
        return {
            "frame_index": self.frame_index,
            "status": status,
            "tracking_quality": round(float(tracking_quality), 3),
            "pose_source": pose_source,
            "image_shift_px": [round(float(image_shift_px[0]), 3), round(float(image_shift_px[1]), 3)],
            "delta_translation_world": np.round(np.array(delta_translation_world, dtype=np.float32), 4).tolist(),
            "translation_world": np.round(self.camera_position_world, 4).tolist(),
            "camera_position_world": np.round(self.camera_position_world, 4).tolist(),
            "rotation_wc": np.round(self.rotation_wc, 4).tolist(),
            "rotation_cw": np.round(rotation_cw, 4).tolist(),
            "active_tracks": len(self.prev_track_ids),
            "sparse_landmark_count": len(self.landmarks),
            "visible_landmark_count": len(sparse_map),
            "persistent_landmark_count": self._count_landmarks("persistent"),
            "missing_landmark_count": self._count_landmarks("missing"),
            "landmark_lifecycle": dict(self.lifecycle_stats),
            "descriptor_backend": self.descriptor_backend.debug_info(),
            "feature_backend": {
                "mode": FEATURE_BACKEND,
                "xfeat": self.descriptor_backend.debug_info(),
                "orb": self.orb_backend.debug_info(),
            },
            "keyframes": len(self.keyframes),
            "stable_landmark_count": self._stable_landmark_count(),
            "stable_2d_landmark_count": self._stable_2d_landmark_count(),
            "geometry_verified_landmark_count": self._geometry_verified_landmark_count(),
            "triangulated_landmark_count": self._triangulated_landmark_count(),
            "dynamic_landmark_count": self._dynamic_landmark_count(),
            "triangulation": dict(self.triangulation_stats),
            "mean_stable_reprojection_error": self._mean_stable_reprojection_error(),
            "latest_keyframe": self.keyframes[-1] if self.keyframes else None,
            "covisibility_edges": self._covisibility_edge_count(),
            "latest_covisible_keyframes": self._latest_covisible_keyframes(),
            "local_keyframes": list(self.local_keyframe_ids),
            "local_landmark_count": len(self.local_landmark_ids),
            "local_visible_landmark_count": len(self._export_local_visible_map()),
            "local_keyframe_baseline": self._local_keyframe_baseline(),
            "pnp_anchor_scope": self.last_pnp_anchor_scope,
            "frames_since_pnp_lock": int(self.frames_since_pnp_lock),
            "ba_lite": dict(self.ba_lite_stats),
            "sliding_ba": dict(self.sliding_ba_stats),
            "pnp_inliers": int(pnp_inliers),
            "pnp_reprojection_error": None if pnp_reprojection_error is None else round(float(pnp_reprojection_error), 4),
            "sparse_map": sparse_map,
            "local_sparse_map": self._export_local_visible_map(),
            "persistent_map": self._export_persistent_map(),
        }

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _feature_mask(self, gray: np.ndarray) -> np.ndarray:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if self.prev_points is None:
            return mask

        for point in self.prev_points.reshape(-1, 2):
            cv2.circle(
                mask,
                (int(round(float(point[0]))), int(round(float(point[1])))),
                FEATURE_MIN_DISTANCE,
                0,
                thickness=-1,
            )
        return mask

    def _detect_grid_features(self, gray: np.ndarray, max_corners: int, mask=None) -> np.ndarray | None:
        if max_corners <= 0:
            return None

        height, width = gray.shape[:2]
        per_cell = max(4, int(np.ceil(max_corners / max(FEATURE_GRID_COLS * FEATURE_GRID_ROWS, 1))) + 2)
        candidates = []
        occupied = []

        for row in range(FEATURE_GRID_ROWS):
            y1 = int(round(row * height / FEATURE_GRID_ROWS))
            y2 = int(round((row + 1) * height / FEATURE_GRID_ROWS))
            for col in range(FEATURE_GRID_COLS):
                x1 = int(round(col * width / FEATURE_GRID_COLS))
                x2 = int(round((col + 1) * width / FEATURE_GRID_COLS))
                if x2 <= x1 or y2 <= y1:
                    continue

                cell = gray[y1:y2, x1:x2]
                cell_mask = None if mask is None else mask[y1:y2, x1:x2]
                points = cv2.goodFeaturesToTrack(
                    cell,
                    maxCorners=per_cell,
                    qualityLevel=FEATURE_QUALITY_LEVEL,
                    minDistance=FEATURE_MIN_DISTANCE,
                    blockSize=FEATURE_BLOCK_SIZE,
                    mask=cell_mask,
                )
                if points is None:
                    continue

                for point in points.reshape(-1, 2):
                    u = float(point[0] + x1)
                    v = float(point[1] + y1)
                    candidates.append((u, v))

        learned_candidates = []
        if FEATURE_BACKEND in {"xfeat", "hybrid"} and self.descriptor_backend.available:
            xfeat_points = self.descriptor_backend.keypoints(gray, max_corners)
            if xfeat_points is not None:
                learned_candidates.extend((float(point[0]), float(point[1])) for point in xfeat_points.reshape(-1, 2))
        if FEATURE_BACKEND in {"orb", "hybrid"}:
            orb_points = self.orb_backend.keypoints(gray, max_corners)
            if orb_points is not None:
                learned_candidates.extend((float(point[0]), float(point[1])) for point in orb_points.reshape(-1, 2))

        filtered_learned_candidates = []
        for u, v in learned_candidates:
            if mask is not None:
                x = int(np.clip(round(u), 0, width - 1))
                y = int(np.clip(round(v), 0, height - 1))
                if int(mask[y, x]) == 0:
                    continue
            filtered_learned_candidates.append((u, v))

        candidates = filtered_learned_candidates + candidates

        if not candidates:
            return None

        if self.prev_points is not None:
            occupied.extend((float(point[0]), float(point[1])) for point in self.prev_points.reshape(-1, 2))

        selected = []
        min_dist_sq = float(FEATURE_MIN_DISTANCE * FEATURE_MIN_DISTANCE)
        for u, v in candidates:
            if len(selected) >= max_corners:
                break
            too_close = any((u - ox) ** 2 + (v - oy) ** 2 < min_dist_sq for ox, oy in occupied)
            if too_close:
                continue
            selected.append([u, v])
            occupied.append((u, v))

        if not selected:
            return None
        return np.asarray(selected, dtype=np.float32).reshape(-1, 1, 2)

    def _detect_features(self, gray: np.ndarray) -> np.ndarray | None:
        return self._detect_grid_features(gray, self.max_points)

    def _append_new_tracks(self, gray: np.ndarray):
        existing = 0 if self.prev_points is None else len(self.prev_points)
        needed = self.max_points - existing
        if needed <= 0:
            return

        new_points = self._detect_grid_features(gray, needed, mask=self._feature_mask(gray))
        if new_points is None or len(new_points) == 0:
            return

        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = new_points.astype(np.float32)
            self.prev_track_ids = []
        else:
            self.prev_points = np.concatenate([self.prev_points, new_points.astype(np.float32)], axis=0)

        used_reobserved_ids = set(self.prev_track_ids)
        for point in new_points.reshape(-1, 2):
            matched_id, descriptor = self._match_missing_landmark(
                gray,
                float(point[0]),
                float(point[1]),
                used_reobserved_ids,
            )
            if matched_id is not None:
                self.prev_track_ids.append(matched_id)
                used_reobserved_ids.add(matched_id)
                landmark = self.landmarks[matched_id]
                landmark["status"] = "visible"
                landmark["image_xy"] = [round(float(point[0]), 1), round(float(point[1]), 1)]
                landmark["missed_frames"] = 0
                landmark["last_seen"] = self.frame_index
                if descriptor is not None:
                    landmark["descriptor"] = descriptor
                self.lifecycle_stats["descriptor_reassociated"] += 1
            else:
                self.prev_track_ids.append(self.next_track_id)
                self.next_track_id += 1

    def _sample_depth(self, depth_map: np.ndarray, u: float, v: float) -> float:
        h, w = depth_map.shape[:2]
        x = int(np.clip(round(u), 0, w - 1))
        y = int(np.clip(round(v), 0, h - 1))
        return float(depth_map[y, x])

    def _sample_semantic_dynamic(self, semantic_mask: np.ndarray | None, u: float, v: float) -> float:
        if semantic_mask is None or semantic_mask.size == 0:
            return 0.0
        h, w = semantic_mask.shape[:2]
        x = int(np.clip(round(u), 0, w - 1))
        y = int(np.clip(round(v), 0, h - 1))
        return 1.0 if float(semantic_mask[y, x]) > 0.5 else 0.0

    def _track_dynamic_evidence(self, points_new, points_old, track_ids, semantic_mask=None) -> dict[int, float]:
        if points_new is None or points_old is None or len(points_new) == 0 or not track_ids:
            return {}

        shifts = points_new - points_old
        median_shift = np.median(shifts, axis=0)
        residuals = np.linalg.norm(shifts - median_shift.reshape(1, 2), axis=1)
        residual_median = float(np.median(residuals)) if len(residuals) else 0.0
        residual_mad = float(np.median(np.abs(residuals - residual_median))) if len(residuals) else 0.0
        robust_sigma = max(1.4826 * residual_mad, 1e-3)

        evidence = {}
        for track_id, pt, residual in zip(track_ids, points_new, residuals.tolist()):
            semantic_value = self._sample_semantic_dynamic(semantic_mask, float(pt[0]), float(pt[1]))
            z = float(residual) / robust_sigma
            motion_value = float(
                np.clip(
                    (z - MOTION_OUTLIER_Z_ON) / max(MOTION_OUTLIER_Z_FULL - MOTION_OUTLIER_Z_ON, 1e-6),
                    0.0,
                    1.0,
                )
            )
            total_weight = max(DYNAMIC_EVIDENCE_SEMANTIC_WEIGHT + DYNAMIC_EVIDENCE_MOTION_WEIGHT, 1e-6)
            combined = float(
                np.clip(
                    (
                        DYNAMIC_EVIDENCE_SEMANTIC_WEIGHT * semantic_value
                        + DYNAMIC_EVIDENCE_MOTION_WEIGHT * motion_value
                    )
                    / total_weight,
                    0.0,
                    1.0,
                )
            )
            evidence[int(track_id)] = combined
        return evidence

    def _semantic_static_track_mask(self, semantic_mask: np.ndarray | None, points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.zeros((0,), dtype=bool)
        if semantic_mask is None or semantic_mask.size == 0:
            return np.ones((len(points),), dtype=bool)
        keep = np.ones((len(points),), dtype=bool)
        for idx, pt in enumerate(points):
            if self._sample_semantic_dynamic(semantic_mask, float(pt[0]), float(pt[1])) > 0.5:
                keep[idx] = False
        return keep

    def _should_update_descriptors(self) -> bool:
        return self.frame_index % XFEAT_UPDATE_EVERY == 0

    def _extract_descriptor(self, gray: np.ndarray, u: float, v: float) -> np.ndarray | None:
        return self.descriptor_backend.describe_at(gray, u, v)

    def _descriptor_similarity(self, a, b) -> float:
        if a is None or b is None:
            return -1.0
        return float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))

    def _match_missing_landmark(self, gray: np.ndarray, u: float, v: float, used_ids: set[int]) -> tuple[int | None, np.ndarray | None]:
        descriptor = self._extract_descriptor(gray, u, v)
        if descriptor is None:
            return None, None

        best_id = None
        best_score = MIN_REOBSERVATION_SIMILARITY
        for track_id, landmark in self.landmarks.items():
            if track_id in used_ids:
                continue
            if landmark.get("status") != "missing":
                continue
            protected_geometry = self._is_protected_geometry_landmark(landmark)
            max_missed = int(
                MAX_MISSING_FRAMES * (PROTECTED_GEOMETRY_MISSING_MULTIPLIER if protected_geometry else 1.0)
            )
            if int(landmark.get("missed_frames", 0)) > max_missed:
                continue
            if "descriptor" not in landmark:
                continue

            image_xy = landmark.get("image_xy")
            if image_xy and len(image_xy) >= 2:
                distance = float(np.hypot(u - float(image_xy[0]), v - float(image_xy[1])))
                distance_limit = (
                    GEOMETRY_REOBSERVATION_DISTANCE_PX
                    if protected_geometry
                    else MAX_REOBSERVATION_DISTANCE_PX
                )
                if distance > distance_limit:
                    continue

            similarity = self._descriptor_similarity(descriptor, landmark.get("descriptor"))
            min_similarity = (
                GEOMETRY_REOBSERVATION_SIMILARITY
                if protected_geometry
                else MIN_REOBSERVATION_SIMILARITY
            )
            if similarity < min_similarity:
                continue
            recency_bonus = max(0.0, 1.0 - int(landmark.get("missed_frames", 0)) / max(MAX_MISSING_FRAMES, 1))
            geometry_bonus = 0.04 if protected_geometry else 0.0
            score = similarity + 0.05 * recency_bonus + geometry_bonus
            if score > best_score:
                best_score = score
                best_id = track_id

        return best_id, descriptor

    def _public_landmark(self, landmark: dict) -> dict:
        return {
            key: value
            for key, value in landmark.items()
            if key not in {"descriptor", "observations"}
        }

    def _finite_world_position(self, landmark: dict):
        for key in ("position_world", "triangulated_position_world", "position_world_depth_prior"):
            value = landmark.get(key)
            if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
                continue
            point = np.asarray(value[:3], dtype=np.float32)
            if np.isfinite(point).all():
                return point
        return None

    def _covisibility_edge_count(self) -> int:
        return sum(len(neighbors) for neighbors in self.covisibility_graph.values()) // 2

    def _latest_covisible_keyframes(self):
        if not self.keyframes:
            return []
        latest_id = int(self.keyframes[-1]["id"])
        neighbors = self.covisibility_graph.get(latest_id, {})
        return [
            {"id": int(keyframe_id), "shared_landmarks": int(shared)}
            for keyframe_id, shared in sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[:8]
        ]

    def _local_keyframe_baseline(self):
        positions = []
        local_ids = set(self.local_keyframe_ids)
        for keyframe in self.keyframes:
            if int(keyframe["id"]) not in local_ids:
                continue
            position = keyframe.get("camera_position_world")
            if not position or len(position) < 3:
                continue
            positions.append(np.asarray(position, dtype=np.float32))

        if len(positions) < 2:
            return 0.0

        max_distance = 0.0
        for idx, position in enumerate(positions):
            for other in positions[idx + 1:]:
                max_distance = max(max_distance, float(np.linalg.norm(position - other)))
        return round(max_distance, 4)

    def _rebuild_covisibility_graph(self):
        live_keyframe_ids = {int(item["id"]) for item in self.keyframes}
        pair_counts = {}

        for landmark in self.landmarks.values():
            observations = landmark.get("observations", [])
            if len(observations) < 2:
                continue

            keyframe_ids = sorted(
                {
                    int(obs["keyframe_id"])
                    for obs in observations
                    if int(obs.get("keyframe_id", -1)) in live_keyframe_ids
                }
            )
            for idx, keyframe_id in enumerate(keyframe_ids):
                for other_id in keyframe_ids[idx + 1:]:
                    pair = (keyframe_id, other_id)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        graph = {keyframe_id: {} for keyframe_id in live_keyframe_ids}
        for (left_id, right_id), shared in pair_counts.items():
            if shared < COVISIBILITY_MIN_SHARED:
                continue
            graph.setdefault(left_id, {})[right_id] = shared
            graph.setdefault(right_id, {})[left_id] = shared
        self.covisibility_graph = graph

    def _select_local_map(self, latest_keyframe_id: int):
        neighbors = self.covisibility_graph.get(latest_keyframe_id, {})
        local_keyframes = [latest_keyframe_id]
        local_keyframes.extend(
            keyframe_id
            for keyframe_id, _ in sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[
                : LOCAL_MAP_MAX_KEYFRAMES - 1
            ]
        )
        local_keyframe_set = set(local_keyframes)

        scored_landmarks = []
        for track_id, landmark in self.landmarks.items():
            observations = landmark.get("observations", [])
            if not observations or int(landmark.get("hits", 0)) < MIN_HITS_FOR_PERSISTENCE:
                continue

            observed_keyframes = {
                int(obs["keyframe_id"])
                for obs in observations
                if int(obs.get("keyframe_id", -1)) in local_keyframe_set
            }
            if not observed_keyframes:
                continue

            score = (
                int(bool(landmark.get("is_triangulated"))),
                int(bool(landmark.get("is_geometry_verified"))),
                len(observed_keyframes),
                int(bool(landmark.get("is_2d_stable"))),
                float(landmark.get("quality", 0.0)),
                int(landmark.get("hits", 0)),
                int(landmark.get("last_seen", 0)),
            )
            scored_landmarks.append((score, track_id))

        scored_landmarks.sort(reverse=True)
        self.local_keyframe_ids = local_keyframes
        self.local_landmark_ids = {
            track_id for _, track_id in scored_landmarks[:LOCAL_MAP_MAX_LANDMARKS]
        }

        for track_id, landmark in self.landmarks.items():
            landmark["is_local_map"] = track_id in self.local_landmark_ids

    def _count_landmarks(self, status: str) -> int:
        if status == "persistent":
            return sum(
                1 for item in self.landmarks.values()
                if item.get("status") in {"visible", "missing"}
                and int(item.get("hits", 0)) >= MIN_HITS_FOR_PERSISTENCE
                and self._is_geometry_owned_landmark(item)
            )
        return sum(1 for item in self.landmarks.values() if item.get("status") == status)

    def _mean_stable_reprojection_error(self):
        errors = [
            float(item["mean_reprojection_error"])
            for item in self.landmarks.values()
            if item.get("is_stable") and item.get("mean_reprojection_error") is not None
        ]
        if not errors:
            return None
        return round(float(np.mean(errors)), 3)

    def _landmark_quality(self, hits: int, age: int, missed_frames: int) -> float:
        hit_score = min(1.0, hits / 12.0)
        age_penalty = min(0.35, age / 600.0)
        miss_penalty = min(0.75, missed_frames / max(MAX_MISSING_FRAMES, 1))
        return float(np.clip(hit_score - age_penalty - miss_penalty, 0.0, 1.0))

    def _mark_unseen_landmarks_missing(self, visible_ids: set[int]):
        for track_id, landmark in self.landmarks.items():
            if track_id in visible_ids:
                continue
            if landmark.get("status") == "visible":
                self.lifecycle_stats["marked_missing"] += 1
            landmark["status"] = "missing"
            landmark["missed_frames"] = self.frame_index - int(landmark.get("last_seen", self.frame_index))
            age = self.frame_index - int(landmark.get("first_seen", self.frame_index))
            landmark["age"] = age
            missed_for_quality = int(landmark.get("missed_frames", 0))
            if self._is_protected_geometry_landmark(landmark):
                missed_for_quality = int(np.ceil(missed_for_quality / max(PROTECTED_GEOMETRY_MISSING_MULTIPLIER, 1e-6)))
            landmark["quality"] = round(
                self._landmark_quality(
                    int(landmark.get("hits", 0)),
                    age,
                    missed_for_quality,
                ),
                3,
            )

    def _prune_landmarks(self):
        stale_ids = []
        for track_id, landmark in self.landmarks.items():
            missed = int(landmark.get("missed_frames", 0))
            max_missed = int(
                MAX_MISSING_FRAMES
                * (PROTECTED_GEOMETRY_MISSING_MULTIPLIER if self._is_protected_geometry_landmark(landmark) else 1.0)
            )
            if missed > max_missed:
                stale_ids.append(track_id)

        if len(self.landmarks) - len(stale_ids) > MAX_LANDMARKS:
            candidates = sorted(
                (
                    (
                        int(
                            self._is_geometry_owned_landmark(item)
                            or bool(item.get("is_triangulated"))
                            or bool(item.get("is_geometry_verified"))
                        ),
                        int(track_id in self.local_landmark_ids),
                        int(item.get("status") == "visible"),
                        float(item.get("quality", 0.0)),
                        int(item.get("last_seen", 0)),
                        track_id,
                    )
                    for track_id, item in self.landmarks.items()
                    if track_id not in stale_ids
                ),
                key=lambda item: (item[0], item[1], item[2], item[3], item[4]),
            )
            overflow = len(self.landmarks) - len(stale_ids) - MAX_LANDMARKS
            stale_ids.extend(track_id for *_, track_id in candidates[:overflow])

        for track_id in stale_ids:
            if track_id in self.landmarks:
                del self.landmarks[track_id]
                self.lifecycle_stats["pruned"] += 1
        if stale_ids:
            self.local_landmark_ids.difference_update(stale_ids)

    def _export_visible_map(self):
        visible = [
            self._public_landmark(item) for item in self.landmarks.values()
            if item.get("status") == "visible"
        ]
        visible.sort(
            key=lambda item: (
                item.get("is_local_map", False),
                item.get("is_triangulated", False),
                item.get("is_geometry_verified", False),
                item.get("is_2d_stable", False),
                item.get("quality", 0.0),
                item.get("hits", 0),
                item.get("last_seen", 0),
            ),
            reverse=True,
        )
        return visible[:VISIBLE_MAP_EXPORT_LIMIT]

    def _export_local_visible_map(self):
        visible_geometry = [
            self._public_landmark(item) for track_id, item in self.landmarks.items()
            if (
                track_id in self.local_landmark_ids
                and item.get("status") == "visible"
                and self._is_geometry_exportable_landmark(item)
            )
        ]
        visible_fallback = [
            self._public_landmark(item) for track_id, item in self.landmarks.items()
            if track_id in self.local_landmark_ids and item.get("status") == "visible"
        ]
        visible = visible_geometry if visible_geometry else visible_fallback
        visible.sort(
            key=lambda item: (
                item.get("is_triangulated", False),
                item.get("is_geometry_verified", False),
                item.get("is_2d_stable", False),
                item.get("quality", 0.0),
                item.get("hits", 0),
                item.get("last_seen", 0),
            ),
            reverse=True,
        )
        return visible[:VISIBLE_MAP_EXPORT_LIMIT]

    def _export_persistent_map(self):
        persistent = [
            self._public_landmark(item) for item in self.landmarks.values()
            if item.get("status") in {"visible", "missing"}
            and int(item.get("hits", 0)) >= MIN_HITS_FOR_PERSISTENCE
            and self._is_geometry_owned_landmark(item)
        ]
        persistent.sort(
            key=lambda item: (
                item.get("is_local_map", False),
                item.get("is_triangulated", False),
                item.get("is_geometry_verified", False),
                item.get("is_2d_stable", False),
                item.get("status") == "visible",
                item.get("quality", 0.0),
                item.get("hits", 0),
                item.get("last_seen", 0),
            ),
            reverse=True,
        )
        return persistent[:max(120, int(PERSISTENT_MAP_EXPORT_LIMIT))]

    def _lift_pixel(self, u: float, v: float, depth_value: float, intrinsics: dict) -> np.ndarray:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])

        x_cam = ((u - cx) / max(fx, 1e-6)) * depth_value
        y_cam = ((v - cy) / max(fy, 1e-6)) * depth_value
        z_cam = depth_value
        camera_point = np.array([x_cam, y_cam, z_cam], dtype=np.float32)
        return self.rotation_wc @ camera_point + self.camera_position_world

    def _camera_matrix(self, intrinsics: dict) -> np.ndarray:
        return np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _distortion_coefficients(self, intrinsics: dict) -> np.ndarray | None:
        values = intrinsics.get("distortion_coefficients", []) if intrinsics else []
        if values is None:
            values = []
        if not isinstance(values, (list, tuple, np.ndarray)):
            values = []
        try:
            coeffs = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        except Exception:
            return None
        if coeffs.size == 0:
            return None
        return coeffs

    def _undistort_points(self, points: np.ndarray, intrinsics: dict) -> np.ndarray:
        if points is None or len(points) == 0 or intrinsics is None:
            return points
        camera_matrix = self._camera_matrix(intrinsics)
        dist_coeffs = self._distortion_coefficients(intrinsics)
        if dist_coeffs is None:
            return points
        try:
            undistorted = cv2.undistortPoints(
                points.reshape(-1, 1, 2).astype(np.float32),
                camera_matrix,
                dist_coeffs,
                P=camera_matrix,
            )
            return undistorted.reshape(-1, 2).astype(np.float32)
        except cv2.error:
            return points

    def _estimate_relative_pose_from_flow(self, old_points, new_points, intrinsics):
        if intrinsics is None or old_points is None or new_points is None or len(new_points) < 8:
            return None

        old_points = self._undistort_points(old_points, intrinsics)
        new_points = self._undistort_points(new_points, intrinsics)

        try:
            camera_matrix = self._camera_matrix(intrinsics)
            essential, inlier_mask = cv2.findEssentialMat(
                old_points.astype(np.float32),
                new_points.astype(np.float32),
                camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.5,
            )
            if essential is None or inlier_mask is None:
                return None

            _, rotation_21, translation_21, pose_mask = cv2.recoverPose(
                essential,
                old_points.astype(np.float32),
                new_points.astype(np.float32),
                camera_matrix,
            )
        except cv2.error:
            return None

        if pose_mask is not None:
            pose_inlier_mask = pose_mask.reshape(-1) > 0
        else:
            pose_inlier_mask = inlier_mask.reshape(-1) > 0
        inliers = int(pose_inlier_mask.sum())
        if inliers < RELATIVE_POSE_MIN_INLIERS:
            return None

        flow = np.linalg.norm(new_points - old_points, axis=1)
        median_flow = float(np.median(flow)) if len(flow) else 0.0
        fx = max(float(intrinsics.get("fx", 1.0)), 1.0)
        # Keep monocular scale independent from per-frame depth.
        flow_ratio = median_flow / fx
        scale = self.essential_translation_scale * float(np.clip(0.5 + 10.0 * flow_ratio, 0.5, 2.0))
        scale = float(np.clip(scale, 0.005, RELATIVE_POSE_MAX_TRANSLATION))
        translation_dir = translation_21.reshape(3).astype(np.float32)
        norm = float(np.linalg.norm(translation_dir))
        if norm <= 1e-6:
            return None
        translation_dir = translation_dir / norm
        return {
            "rotation_21": rotation_21.astype(np.float32),
            "translation_21": translation_dir,
            "scale": scale,
            "inliers": inliers,
            "inlier_mask": pose_inlier_mask,
        }

    def _project_world_point(self, world_point, rotation_cw, camera_position_world, intrinsics) -> np.ndarray | None:
        world_point = np.asarray(world_point, dtype=np.float32)
        camera_point = rotation_cw @ (world_point - camera_position_world)
        if float(camera_point[2]) <= 1e-6:
            return None

        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        u = fx * float(camera_point[0]) / float(camera_point[2]) + cx
        v = fy * float(camera_point[1]) / float(camera_point[2]) + cy
        return np.array([u, v], dtype=np.float32)

    def _ray_from_observation(self, observation):
        intrinsics = observation["intrinsics"]
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        u, v = observation["image_xy"]
        ray_camera = np.array(
            [
                (float(u) - cx) / max(fx, 1e-6),
                (float(v) - cy) / max(fy, 1e-6),
                1.0,
            ],
            dtype=np.float32,
        )
        ray_camera = ray_camera / max(float(np.linalg.norm(ray_camera)), 1e-6)
        rotation_cw = np.asarray(observation["rotation_cw"], dtype=np.float32)
        rotation_wc = rotation_cw.T
        origin = np.asarray(observation["camera_position_world"], dtype=np.float32)
        direction = rotation_wc @ ray_camera
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        return origin, direction

    def _triangulate_from_observations(self, observations):
        if len(observations) < 2:
            return None

        lhs = np.zeros((3, 3), dtype=np.float32)
        rhs = np.zeros(3, dtype=np.float32)
        eye = np.eye(3, dtype=np.float32)
        for observation in observations:
            origin, direction = self._ray_from_observation(observation)
            projector = eye - np.outer(direction, direction).astype(np.float32)
            lhs += projector
            rhs += projector @ origin

        try:
            point = np.linalg.solve(lhs + 1e-4 * eye, rhs)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(point)):
            return None
        return point.astype(np.float32)

    def _observation_geometry(self, observations):
        if len(observations) < 2:
            return 0.0, 0.0

        rays = [self._ray_from_observation(obs) for obs in observations]
        max_baseline = 0.0
        max_angle = 0.0
        for idx, (origin, direction) in enumerate(rays):
            for other_origin, other_direction in rays[idx + 1:]:
                max_baseline = max(max_baseline, float(np.linalg.norm(origin - other_origin)))
                dot = float(np.clip(np.dot(direction, other_direction), -1.0, 1.0))
                max_angle = max(max_angle, float(np.degrees(np.arccos(dot))))
        return max_baseline, max_angle

    def _mean_reprojection_error_for_point(self, world_point, observations):
        errors = []
        for obs in observations:
            projected = self._project_world_point(
                world_point,
                np.asarray(obs["rotation_cw"], dtype=np.float32),
                np.asarray(obs["camera_position_world"], dtype=np.float32),
                obs["intrinsics"],
            )
            if projected is None:
                continue
            image_xy = np.asarray(obs["image_xy"], dtype=np.float32)
            errors.append(float(np.linalg.norm(projected - image_xy)))
        if not errors:
            return None
        return float(np.mean(errors))

    def _mean_depth_error_for_point(self, world_point, observations):
        errors = []
        world_point = np.asarray(world_point, dtype=np.float32)
        for obs in observations:
            expected_depth = obs.get("depth")
            if expected_depth is None:
                continue
            rotation_cw = np.asarray(obs["rotation_cw"], dtype=np.float32)
            camera_position_world = np.asarray(obs["camera_position_world"], dtype=np.float32)
            camera_point = rotation_cw @ (world_point - camera_position_world)
            if float(camera_point[2]) <= 1e-6:
                continue
            errors.append(abs(float(camera_point[2]) - float(expected_depth)))
        if not errors:
            return None
        return float(np.mean(errors))

    def _refresh_triangulated_landmark(self, track_id):
        landmark = self.landmarks.get(track_id)
        if not landmark:
            return
        observations = landmark.get("observations", [])
        if len(observations) < BA_LITE_MIN_OBSERVATIONS:
            landmark["is_triangulated"] = False
            self.triangulation_stats["rejected_observations"] += 1
            return

        self.triangulation_stats["candidates"] += 1
        baseline, ray_angle = self._observation_geometry(observations)
        landmark["triangulation_baseline"] = round(baseline, 4)
        landmark["triangulation_angle_deg"] = round(ray_angle, 3)
        if baseline < TRIANGULATED_MIN_BASELINE:
            landmark["is_triangulated"] = False
            self.triangulation_stats["rejected_baseline"] += 1
            return
        if ray_angle < TRIANGULATED_MIN_RAY_ANGLE_DEG:
            landmark["is_triangulated"] = False
            self.triangulation_stats["rejected_angle"] += 1
            return

        triangulated = self._triangulate_from_observations(observations)
        if triangulated is None:
            landmark["is_triangulated"] = False
            self.triangulation_stats["rejected_solver"] += 1
            return

        reprojection_error = self._mean_reprojection_error_for_point(triangulated, observations)
        depth_error = self._mean_depth_error_for_point(triangulated, observations)
        observed_depths = [
            float(obs["depth"])
            for obs in observations
            if obs.get("depth") is not None and np.isfinite(float(obs["depth"]))
        ]
        mean_depth = float(np.mean(observed_depths)) if observed_depths else None
        max_depth_error = (
            max(TRIANGULATED_MAX_DEPTH_ERROR, 0.35 * mean_depth)
            if mean_depth is not None
            else TRIANGULATED_MAX_DEPTH_ERROR
        )
        landmark["triangulated_position_world"] = np.round(triangulated, 4).tolist()
        landmark["triangulated_reprojection_error"] = (
            round(reprojection_error, 3) if reprojection_error is not None else None
        )
        landmark["triangulated_depth_error"] = (
            round(depth_error, 3) if depth_error is not None else None
        )
        landmark["triangulated_depth_error_limit"] = round(max_depth_error, 3)

        is_good = (
            reprojection_error is not None
            and reprojection_error <= TRIANGULATED_MAX_REPROJECTION_ERROR
        )
        landmark["is_triangulated"] = bool(is_good)
        if landmark["is_triangulated"]:
            landmark["is_geometry_verified"] = True
        if is_good:
            self.triangulation_stats["accepted"] += 1
            if depth_error is not None and depth_error > max_depth_error:
                self.triangulation_stats["depth_disagreement"] += 1
            current_point = self._finite_world_position(landmark)
            if current_point is None:
                current_point = triangulated
            blend = float(np.clip(TRIANGULATION_POSITION_BLEND, 0.05, 1.0))
            landmark["position_world"] = np.round((1.0 - blend) * current_point + blend * triangulated, 4).tolist()
            landmark["position_world_source"] = "triangulated"
        elif reprojection_error is None or reprojection_error > TRIANGULATED_MAX_REPROJECTION_ERROR:
            self.triangulation_stats["rejected_reprojection"] += 1
        else:
            self.triangulation_stats["rejected_depth"] += 1

    def _sliding_ba_window(self):
        if least_squares is None:
            self.sliding_ba_stats["last_status"] = "scipy-unavailable"
            return None

        if len(self.keyframes) < SLIDING_BA_MIN_KEYFRAMES:
            self.sliding_ba_stats["last_status"] = "too-few-keyframes"
            return None

        local_ids = list(self.local_keyframe_ids)
        if len(local_ids) < SLIDING_BA_MIN_KEYFRAMES:
            local_ids = [int(item["id"]) for item in self.keyframes[-SLIDING_BA_MAX_KEYFRAMES:]]
        local_ids = local_ids[-SLIDING_BA_MAX_KEYFRAMES:]
        keyframe_by_id = {int(item["id"]): item for item in self.keyframes if int(item["id"]) in set(local_ids)}
        keyframe_ids = [keyframe_id for keyframe_id in local_ids if keyframe_id in keyframe_by_id]
        if len(keyframe_ids) < SLIDING_BA_MIN_KEYFRAMES:
            self.sliding_ba_stats["last_status"] = "too-few-local-keyframes"
            return None

        scored_landmarks = []
        candidates_seen = 0
        candidates_rejected = 0
        for track_id, landmark in self.landmarks.items():
            world = self._finite_world_position(landmark)
            if world is None:
                continue
            if not self._is_geometry_owned_landmark(landmark):
                continue
            observations = [
                obs for obs in landmark.get("observations", [])
                if int(obs.get("keyframe_id", -1)) in keyframe_by_id
            ]
            observed_keyframes = {int(obs["keyframe_id"]) for obs in observations}
            if len(observed_keyframes) < SLIDING_BA_MIN_OBSERVATIONS:
                continue
            candidates_seen += 1
            if int(landmark.get("hits", 0)) < SLIDING_BA_MIN_LANDMARK_HITS:
                candidates_rejected += 1
                continue
            mean_error = landmark.get("mean_reprojection_error")
            if mean_error is not None and float(mean_error) > SLIDING_BA_MAX_LANDMARK_ERROR:
                candidates_rejected += 1
                continue
            baseline, ray_angle = self._observation_geometry(observations)
            if baseline < BA_LITE_MIN_BASELINE or ray_angle < BA_LITE_MIN_RAY_ANGLE_DEG:
                candidates_rejected += 1
                continue
            observed_depths = [
                float(depth)
                for depth in (
                    obs.get("depth", landmark.get("depth"))
                    for obs in observations
                )
                if depth is not None
            ]
            depth_variance = float(np.var(observed_depths)) if len(observed_depths) >= 2 else 0.0
            score = (
                int(track_id in self.local_landmark_ids),
                int(bool(landmark.get("is_triangulated"))),
                int(bool(landmark.get("is_geometry_verified"))),
                int(bool(landmark.get("is_2d_stable"))),
                len(observations),
                -depth_variance,
                float(landmark.get("quality", 0.0)),
                int(landmark.get("hits", 0)),
            )
            scored_landmarks.append((score, track_id, observations))

        scored_landmarks.sort(reverse=True)
        selected = scored_landmarks[:SLIDING_BA_MAX_LANDMARKS]
        self.sliding_ba_stats["last_candidates"] = candidates_seen
        self.sliding_ba_stats["last_rejected"] = candidates_rejected
        if not selected:
            self.sliding_ba_stats["last_status"] = "no-parallax-landmarks"
            return None

        observations = []
        for _, track_id, landmark_observations in selected:
            for obs in landmark_observations:
                observations.append((track_id, int(obs["keyframe_id"]), obs))
                if len(observations) >= SLIDING_BA_MAX_RESIDUAL_OBS:
                    break
            if len(observations) >= SLIDING_BA_MAX_RESIDUAL_OBS:
                break

        landmark_ids = sorted({track_id for track_id, _, _ in observations})
        if len(landmark_ids) < 6 or len(observations) < 12:
            self.sliding_ba_stats["last_status"] = "too-few-observations"
            return None

        return keyframe_ids, keyframe_by_id, landmark_ids, observations

    def _pack_sliding_ba_parameters(self, keyframe_ids, keyframe_by_id, landmark_ids):
        fixed_keyframe_id = keyframe_ids[0]
        variable_keyframe_ids = keyframe_ids[1:]
        params = []
        pose_priors = {}
        point_priors = {}

        for keyframe_id in variable_keyframe_ids:
            keyframe = keyframe_by_id[keyframe_id]
            rotation_cw = np.asarray(keyframe["rotation_cw"], dtype=np.float64)
            rvec, _ = cv2.Rodrigues(rotation_cw)
            position = np.asarray(keyframe["camera_position_world"], dtype=np.float64)
            if not np.isfinite(rvec).all() or not np.isfinite(position).all():
                continue
            pose_priors[keyframe_id] = (rvec.reshape(3).copy(), position.reshape(3).copy())
            params.extend(rvec.reshape(3).tolist())
            params.extend(position.reshape(3).tolist())

        kept_landmark_ids = []
        for track_id in landmark_ids:
            point = self._finite_world_position(self.landmarks[track_id])
            if point is None:
                continue
            point = point.astype(np.float64)
            if point.shape[0] < 3 or not np.isfinite(point).all():
                continue
            kept_landmark_ids.append(track_id)
            point_priors[track_id] = point.reshape(3).copy()
            params.extend(point.reshape(3).tolist())

        return (
            np.asarray(params, dtype=np.float64),
            fixed_keyframe_id,
            variable_keyframe_ids,
            kept_landmark_ids,
            pose_priors,
            point_priors,
        )

    def _unpack_sliding_ba_parameters(self, params, keyframe_ids, keyframe_by_id, landmark_ids, fixed_keyframe_id, variable_keyframe_ids):
        pose_params = {}
        offset = 0
        fixed_keyframe = keyframe_by_id[fixed_keyframe_id]
        pose_params[fixed_keyframe_id] = (
            np.asarray(fixed_keyframe["rotation_cw"], dtype=np.float64),
            np.asarray(fixed_keyframe["camera_position_world"], dtype=np.float64),
        )

        for keyframe_id in variable_keyframe_ids:
            rvec = params[offset:offset + 3].reshape(3, 1)
            offset += 3
            position = params[offset:offset + 3]
            offset += 3
            rotation_cw, _ = cv2.Rodrigues(rvec)
            pose_params[keyframe_id] = (rotation_cw, position)

        point_params = {}
        for track_id in landmark_ids:
            point_params[track_id] = params[offset:offset + 3]
            offset += 3
        return pose_params, point_params

    def _sliding_ba_residuals(
        self,
        params,
        keyframe_ids,
        keyframe_by_id,
        landmark_ids,
        observations,
        fixed_keyframe_id,
        variable_keyframe_ids,
        pose_priors=None,
        point_priors=None,
        include_priors=True,
    ):
        pose_params, point_params = self._unpack_sliding_ba_parameters(
            params,
            keyframe_ids,
            keyframe_by_id,
            landmark_ids,
            fixed_keyframe_id,
            variable_keyframe_ids,
        )
        residuals = []
        for track_id, keyframe_id, obs in observations:
            rotation_cw, camera_position_world = pose_params[keyframe_id]
            projected = self._project_world_point(
                point_params[track_id],
                rotation_cw,
                camera_position_world,
                obs["intrinsics"],
            )
            if projected is None:
                residuals.extend([50.0, 50.0])
                continue
            image_xy = np.asarray(obs["image_xy"], dtype=np.float64)
            residuals.extend((projected.astype(np.float64) - image_xy).tolist())

        if include_priors:
            pose_priors = pose_priors or {}
            point_priors = point_priors or {}
            offset = 0
            for keyframe_id in variable_keyframe_ids:
                rvec = params[offset:offset + 3]
                offset += 3
                position = params[offset:offset + 3]
                offset += 3
                prior = pose_priors.get(keyframe_id)
                if prior is not None:
                    prior_rvec, prior_position = prior
                    residuals.extend(((rvec - prior_rvec) * SLIDING_BA_ROTATION_PRIOR_WEIGHT).tolist())
                    residuals.extend(((position - prior_position) * SLIDING_BA_TRANSLATION_PRIOR_WEIGHT).tolist())

            for track_id in landmark_ids:
                point = params[offset:offset + 3]
                offset += 3
                prior_point = point_priors.get(track_id)
                if prior_point is not None:
                    residuals.extend(((point - prior_point) * SLIDING_BA_POINT_PRIOR_WEIGHT).tolist())

            if SLIDING_BA_DEPTH_PRIOR_WEIGHT > 0.0:
                for track_id, keyframe_id, obs in observations:
                    expected_depth = obs.get("depth")
                    if expected_depth is None:
                        residuals.append(0.0)
                        continue
                    rotation_cw, camera_position_world = pose_params[keyframe_id]
                    world_point = point_params[track_id]
                    camera_point = rotation_cw @ (world_point - camera_position_world)
                    if float(camera_point[2]) <= 1e-6:
                        residuals.append(50.0)
                        continue
                    residuals.append((float(camera_point[2]) - float(expected_depth)) * SLIDING_BA_DEPTH_PRIOR_WEIGHT)
        return np.asarray(residuals, dtype=np.float64)

    def _update_observation_poses(self, optimized_keyframe_ids: set[int], pose_params):
        for landmark in self.landmarks.values():
            for obs in landmark.get("observations", []):
                keyframe_id = int(obs.get("keyframe_id", -1))
                if keyframe_id not in optimized_keyframe_ids:
                    continue
                rotation_cw, camera_position_world = pose_params[keyframe_id]
                obs["rotation_cw"] = np.round(rotation_cw, 6).tolist()
                obs["camera_position_world"] = np.round(camera_position_world, 4).tolist()

    def _run_sliding_window_ba(self):
        window = self._sliding_ba_window()
        if window is None:
            return

        keyframe_ids, keyframe_by_id, landmark_ids, observations = window
        if len(self.keyframes) % SLIDING_BA_RUN_EVERY_N_KEYFRAMES != 0:
            self.sliding_ba_stats["last_status"] = "skipped-throttle"
            return

        params0, fixed_keyframe_id, variable_keyframe_ids, landmark_ids, pose_priors, point_priors = self._pack_sliding_ba_parameters(
            keyframe_ids,
            keyframe_by_id,
            landmark_ids,
        )
        if len(landmark_ids) < 6 or params0.size == 0 or not np.isfinite(params0).all():
            self.sliding_ba_stats["last_status"] = "invalid-initial-params"
            return
        landmark_id_set = set(landmark_ids)
        observations = [obs for obs in observations if obs[0] in landmark_id_set]
        if len(observations) < 12:
            self.sliding_ba_stats["last_status"] = "too-few-valid-observations"
            return
        residuals0 = self._sliding_ba_residuals(
            params0,
            keyframe_ids,
            keyframe_by_id,
            landmark_ids,
            observations,
            fixed_keyframe_id,
            variable_keyframe_ids,
            pose_priors,
            point_priors,
            False,
        )
        if residuals0.size == 0:
            self.sliding_ba_stats["last_status"] = "empty-residuals"
            return

        try:
            result = least_squares(
                self._sliding_ba_residuals,
                params0,
                args=(
                    keyframe_ids,
                    keyframe_by_id,
                    landmark_ids,
                    observations,
                    fixed_keyframe_id,
                    variable_keyframe_ids,
                    pose_priors,
                    point_priors,
                    True,
                ),
                loss="huber",
                f_scale=4.0,
                max_nfev=SLIDING_BA_MAX_NFEV,
                x_scale="jac",
                verbose=0,
            )
        except Exception as exc:
            self.sliding_ba_stats["last_status"] = f"failed: {str(exc)[:80]}"
            return

        pose_params, point_params = self._unpack_sliding_ba_parameters(
            result.x,
            keyframe_ids,
            keyframe_by_id,
            landmark_ids,
            fixed_keyframe_id,
            variable_keyframe_ids,
        )
        optimized_keyframe_ids = set(keyframe_ids)
        for keyframe_id, keyframe in keyframe_by_id.items():
            rotation_cw, camera_position_world = pose_params[keyframe_id]
            keyframe["rotation_cw"] = np.round(rotation_cw, 6).tolist()
            keyframe["camera_position_world"] = np.round(camera_position_world, 4).tolist()

        for track_id, point in point_params.items():
            landmark = self.landmarks.get(track_id)
            if not landmark:
                continue
            landmark["position_world"] = np.round(point, 4).tolist()
            landmark["sliding_ba_refined"] = True
            landmark["position_world_source"] = "sliding-ba"

        self._update_observation_poses(optimized_keyframe_ids, pose_params)
        latest_keyframe_id = int(self.keyframes[-1]["id"]) if self.keyframes else None
        if latest_keyframe_id in pose_params:
            rotation_cw, camera_position_world = pose_params[latest_keyframe_id]
            self.rotation_wc = rotation_cw.T.astype(np.float32)
            self.camera_position_world = camera_position_world.astype(np.float32)

        for track_id in landmark_ids:
            self._update_landmark_reprojection_stats(track_id)
            self._refresh_triangulated_landmark(track_id)

        residuals1 = self._sliding_ba_residuals(
            result.x,
            keyframe_ids,
            keyframe_by_id,
            landmark_ids,
            observations,
            fixed_keyframe_id,
            variable_keyframe_ids,
            pose_priors,
            point_priors,
            False,
        )
        cost_before = float(np.mean(np.abs(residuals0)))
        cost_after = float(np.mean(np.abs(residuals1)))
        if result.success:
            status = "optimized"
        elif cost_after < cost_before * 0.98:
            status = "improved-max-iter"
        else:
            status = "max-iter"
        self.sliding_ba_stats.update({
            "available": True,
            "runs": int(self.sliding_ba_stats.get("runs", 0)) + 1,
            "last_status": status,
            "last_keyframes": len(keyframe_ids),
            "last_landmarks": len(landmark_ids),
            "last_observations": len(observations),
            "last_cost_before": round(cost_before, 3),
            "last_cost_after": round(cost_after, 3),
        })

    def _run_ba_lite(self, candidate_ids):
        before_errors = []
        after_errors = []
        refined = 0
        skipped_low_parallax = 0

        for track_id in list(candidate_ids)[:BA_LITE_MAX_UPDATES_PER_KEYFRAME]:
            landmark = self.landmarks.get(track_id)
            if not landmark:
                continue
            observations = landmark.get("observations", [])
            if len(observations) < BA_LITE_MIN_OBSERVATIONS:
                continue

            baseline, ray_angle = self._observation_geometry(observations)
            landmark["triangulation_baseline"] = round(baseline, 4)
            landmark["triangulation_angle_deg"] = round(ray_angle, 3)
            if baseline < BA_LITE_MIN_BASELINE or ray_angle < BA_LITE_MIN_RAY_ANGLE_DEG:
                skipped_low_parallax += 1
                self._update_landmark_reprojection_stats(track_id)
                continue

            geometry_owned = self._is_geometry_owned_landmark(landmark)
            before_error = None
            current_point = None
            if geometry_owned:
                world = self._finite_world_position(landmark)
                if world is None:
                    self._update_landmark_reprojection_stats(track_id)
                    continue
                current_point = world
                before_error = self._mean_reprojection_error_for_point(current_point, observations)
                if before_error is None or before_error > BA_LITE_MAX_INITIAL_ERROR:
                    self._update_landmark_reprojection_stats(track_id)
                    continue

            triangulated = self._triangulate_from_observations(observations)
            if triangulated is None:
                self._update_landmark_reprojection_stats(track_id)
                continue

            if geometry_owned:
                candidate = (1.0 - BA_LITE_BLEND) * current_point + BA_LITE_BLEND * triangulated
            else:
                candidate = triangulated
            after_error = self._mean_reprojection_error_for_point(candidate, observations)
            if after_error is None:
                self._update_landmark_reprojection_stats(track_id)
                continue
            if geometry_owned and after_error > before_error:
                self._update_landmark_reprojection_stats(track_id)
                continue
            if not geometry_owned and after_error > STABLE_MAX_REPROJECTION_ERROR:
                self._update_landmark_reprojection_stats(track_id)
                continue

            landmark["position_world"] = np.round(candidate, 4).tolist()
            landmark["ba_lite_refined"] = True
            landmark["position_world_source"] = "ba-lite"
            landmark["ba_lite_error_before"] = round(before_error, 3) if before_error is not None else None
            landmark["ba_lite_error_after"] = round(after_error, 3)
            self._update_landmark_reprojection_stats(track_id)
            self._refresh_triangulated_landmark(track_id)
            before_errors.append(before_error)
            after_errors.append(after_error)
            refined += 1

        self.ba_lite_stats["runs"] += 1
        self.ba_lite_stats["last_refined"] = refined
        self.ba_lite_stats["last_skipped_low_parallax"] = skipped_low_parallax
        self.ba_lite_stats["landmarks_refined"] += refined
        before_values = [float(value) for value in before_errors if value is not None]
        after_values = [float(value) for value in after_errors if value is not None]
        self.ba_lite_stats["last_mean_error_before"] = (
            round(float(np.mean(before_values)), 3) if before_values else None
        )
        self.ba_lite_stats["last_mean_error_after"] = (
            round(float(np.mean(after_values)), 3) if after_values else None
        )

    def _update_landmark_reprojection_stats(self, track_id):
        landmark = self.landmarks.get(track_id)
        if not landmark:
            return
        observations = landmark.get("observations", [])
        if not observations:
            landmark["observation_count"] = 0
            landmark["mean_reprojection_error"] = None
            landmark["is_stable"] = False
            landmark["is_geometry_verified"] = False
            landmark["is_2d_stable"] = int(landmark.get("hits", 0)) >= STABLE_2D_MIN_HITS
            return

        landmark["is_2d_stable"] = (
            int(landmark.get("hits", 0)) >= STABLE_2D_MIN_HITS
            and int(landmark.get("missed_frames", 0)) == 0
            and float(landmark.get("quality", 0.0)) >= 0.45
        )
        errors = []
        world_point = self._finite_world_position(landmark)
        if world_point is None:
            landmark["observation_count"] = len(observations)
            landmark["mean_reprojection_error"] = None
            landmark["is_stable"] = False
            landmark["is_geometry_verified"] = False
            return

        for obs in observations:
            projected = self._project_world_point(
                world_point,
                np.asarray(obs["rotation_cw"], dtype=np.float32),
                np.asarray(obs["camera_position_world"], dtype=np.float32),
                obs["intrinsics"],
            )
            if projected is None:
                continue
            image_xy = np.asarray(obs["image_xy"], dtype=np.float32)
            errors.append(float(np.linalg.norm(projected - image_xy)))

        landmark["observation_count"] = len(observations)
        if errors:
            mean_error = float(np.mean(errors))
            baseline, ray_angle = self._observation_geometry(observations)
            landmark["mean_reprojection_error"] = round(mean_error, 3)
            landmark["geometry_baseline"] = round(baseline, 4)
            landmark["geometry_ray_angle_deg"] = round(ray_angle, 3)
            geometry_owned = self._is_geometry_owned_landmark(landmark) or bool(landmark.get("is_triangulated"))
            landmark["is_geometry_verified"] = geometry_owned and (
                len(observations) >= GEOMETRY_VERIFIED_MIN_OBSERVATIONS
                and mean_error <= GEOMETRY_VERIFIED_MAX_REPROJECTION_ERROR
                and ray_angle >= GEOMETRY_VERIFIED_MIN_RAY_ANGLE_DEG
            )
            landmark["is_stable"] = (
                len(observations) >= STABLE_MIN_OBSERVATIONS
                and mean_error <= STABLE_MAX_REPROJECTION_ERROR
            )
            if landmark["is_geometry_verified"] and not landmark.get("position_world_source"):
                landmark["position_world_source"] = "geometry"
        else:
            landmark["mean_reprojection_error"] = None
            landmark["is_stable"] = False
            landmark["is_geometry_verified"] = False

    def _stable_landmark_count(self) -> int:
        return sum(1 for item in self.landmarks.values() if item.get("is_stable"))

    def _should_create_keyframe(self, visible_ids: set[int], pnp_inliers: int, pose_source: str) -> bool:
        if pose_source not in {"pnp", "essential"}:
            return False
        if pose_source == "essential":
            if (
                ESSENTIAL_KEYFRAME_MAX_FRAMES_SINCE_PNP >= 0
                and self.frames_since_pnp_lock > ESSENTIAL_KEYFRAME_MAX_FRAMES_SINCE_PNP
            ):
                return False
            if ESSENTIAL_KEYFRAME_MIN_INLIERS > 0 and pnp_inliers < ESSENTIAL_KEYFRAME_MIN_INLIERS:
                return False
        if len(visible_ids) < KEYFRAME_MIN_VISIBLE or pnp_inliers < KEYFRAME_MIN_VISIBLE:
            return False
        if not self.keyframes:
            return True
        if self.frame_index - self.last_keyframe_frame < KEYFRAME_MIN_INTERVAL:
            return False
        if self.last_keyframe_position is None:
            return True
        translation = float(np.linalg.norm(self.camera_position_world - self.last_keyframe_position))
        return translation >= KEYFRAME_MIN_TRANSLATION

    def _maybe_add_keyframe(self, visible_ids: set[int], pnp_inliers: int, pose_source: str, intrinsics):
        if intrinsics is None or not self._should_create_keyframe(visible_ids, pnp_inliers, pose_source):
            return

        rotation_cw = self.rotation_wc.T.astype(np.float32)
        keyframe = {
            "id": self.next_keyframe_id,
            "frame_index": self.frame_index,
            "camera_position_world": np.round(self.camera_position_world, 4).tolist(),
            "rotation_cw": np.round(rotation_cw, 6).tolist(),
            "visible_landmarks": len(visible_ids),
        }
        self.next_keyframe_id += 1
        self.keyframes.append(keyframe)
        self.keyframes = self.keyframes[-MAX_KEYFRAMES:]
        self.last_keyframe_frame = self.frame_index
        self.last_keyframe_position = self.camera_position_world.copy()

        intrinsics_snapshot = {
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
        }

        for track_id in visible_ids:
            landmark = self.landmarks.get(track_id)
            if not landmark:
                continue
            observation = {
                "keyframe_id": keyframe["id"],
                "frame_index": self.frame_index,
                "image_xy": landmark.get("image_xy", [0.0, 0.0]),
                "depth": landmark.get("depth"),
                "camera_position_world": keyframe["camera_position_world"],
                "rotation_cw": keyframe["rotation_cw"],
                "intrinsics": intrinsics_snapshot,
            }
            observations = landmark.setdefault("observations", [])
            observations.append(observation)
            landmark["observations"] = observations[-MAX_OBSERVATIONS_PER_LANDMARK:]
            self._update_landmark_reprojection_stats(track_id)
            self._refresh_triangulated_landmark(track_id)

        self._rebuild_covisibility_graph()
        self._select_local_map(keyframe["id"])
        self._run_ba_lite(self.local_landmark_ids or visible_ids)

    def consolidate_map(self):
        """Run heavier local map optimization outside the frame hot path."""
        self._run_sliding_window_ba()

    def _estimate_pose_from_anchors(self, points, track_ids, intrinsics):
        if intrinsics is None or points is None or len(points) < 6:
            return None

        candidates = []
        for track_id, pt in zip(track_ids, points):
            landmark = self.landmarks.get(track_id)
            if not landmark:
                continue
            world_point = self._finite_world_position(landmark)
            if world_point is None:
                continue
            dynamic_score = float(landmark.get("dynamic_score", 0.0))
            candidates.append({
                "track_id": track_id,
                "object_point": world_point.tolist(),
                "image_point": [float(pt[0]), float(pt[1])],
                "is_local": track_id in self.local_landmark_ids,
                "is_stable": bool(landmark.get("is_stable")),
                "is_2d_stable": bool(landmark.get("is_2d_stable")),
                "is_geometry_verified": bool(landmark.get("is_geometry_verified")),
                "is_triangulated": bool(landmark.get("is_triangulated")),
                "dynamic_score": dynamic_score,
                "is_dynamic": bool(landmark.get("is_dynamic")),
            })

        if len(candidates) < 6:
            return None

        def prefer_static(items):
            return [item for item in items if float(item.get("dynamic_score", 0.0)) < HIGH_DYNAMIC_SCORE]

        local_candidates = [item for item in candidates if item["is_local"]]
        triangulated_candidates = [item for item in candidates if item["is_triangulated"]]
        geometry_candidates = [item for item in candidates if item["is_geometry_verified"]]
        local_verified_candidates = [
            item for item in local_candidates
            if item["is_triangulated"] or item["is_geometry_verified"]
        ]
        triangulated_static = prefer_static(triangulated_candidates)
        local_verified_static = prefer_static(local_verified_candidates)
        geometry_static = prefer_static(geometry_candidates)

        if len(triangulated_static) >= MIN_LOCAL_PNP_ANCHORS:
            candidates = triangulated_static
            anchor_scope = "triangulated-static"
        elif len(triangulated_candidates) >= MIN_LOCAL_PNP_ANCHORS:
            candidates = triangulated_candidates
            anchor_scope = "triangulated"
        elif len(local_verified_static) >= MIN_LOCAL_PNP_ANCHORS:
            candidates = local_verified_static
            anchor_scope = "local-verified-static"
        elif len(local_verified_candidates) >= MIN_LOCAL_PNP_ANCHORS:
            candidates = local_verified_candidates
            anchor_scope = "local-verified"
        elif len(geometry_static) >= MIN_LOCAL_PNP_ANCHORS:
            candidates = geometry_static
            anchor_scope = "geometry-verified-static"
        elif len(geometry_candidates) >= MIN_LOCAL_PNP_ANCHORS:
            candidates = geometry_candidates
            anchor_scope = "geometry-verified"
        else:
            return None

        candidates = sorted(
            candidates,
            key=lambda item: (
                float(item.get("dynamic_score", 0.0)),
                int(not item.get("is_triangulated")),
                int(not item.get("is_geometry_verified")),
                int(not item.get("is_local")),
            ),
        )[:max(60, int(PNP_MAX_ANCHORS))]

        object_points = np.asarray([item["object_point"] for item in candidates], dtype=np.float32)
        image_points = np.asarray([item["image_point"] for item in candidates], dtype=np.float32)
        camera_matrix = self._camera_matrix(intrinsics)
        dist_coeffs = self._distortion_coefficients(intrinsics)

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=8.0,
                confidence=0.98,
                iterationsCount=100,
            )
        except cv2.error:
            return None

        if not success or rvec is None or tvec is None:
            return None

        if inliers is not None and len(inliers) >= 6:
            inlier_ids = inliers.reshape(-1).astype(np.int32)
            try:
                refine_success, rvec_refined, tvec_refined = cv2.solvePnP(
                    object_points[inlier_ids],
                    image_points[inlier_ids],
                    camera_matrix,
                    dist_coeffs,
                    rvec=rvec,
                    tvec=tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if (
                    refine_success
                    and rvec_refined is not None
                    and tvec_refined is not None
                    and np.isfinite(rvec_refined).all()
                    and np.isfinite(tvec_refined).all()
                ):
                    rvec = rvec_refined
                    tvec = tvec_refined
            except cv2.error:
                pass

        rotation_cw, _ = cv2.Rodrigues(rvec)
        rotation_wc = rotation_cw.T.astype(np.float32)
        camera_position_world = (-rotation_wc @ tvec.reshape(3)).astype(np.float32)

        reprojection_error = None
        inlier_track_ids = []
        if inliers is not None and len(inliers) > 0:
            inlier_ids = inliers.reshape(-1)
            inlier_track_ids = [int(candidates[int(idx)]["track_id"]) for idx in inlier_ids]
            reprojected, _ = cv2.projectPoints(
                object_points[inlier_ids],
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            reprojection_error = float(
                np.mean(
                    np.linalg.norm(
                        reprojected.reshape(-1, 2) - image_points[inlier_ids],
                        axis=1,
                    )
                )
            )

        return {
            "rotation_wc": rotation_wc,
            "camera_position_world": camera_position_world,
            "inliers": 0 if inliers is None else int(len(inliers)),
            "reprojection_error": reprojection_error,
            "anchor_scope": anchor_scope,
            "anchors_considered": len(candidates),
            "inlier_track_ids": inlier_track_ids,
        }

    def _update_landmarks(self, points, track_ids, depth_map, intrinsics, gray=None, semantic_mask=None, dynamic_evidence_by_track=None):
        if intrinsics is None or points is None or len(points) == 0:
            self._mark_unseen_landmarks_missing(set())
            self._prune_landmarks()
            return self._export_visible_map()

        visible_ids = set()
        for track_id, pt in zip(track_ids, points):
            u = float(pt[0])
            v = float(pt[1])
            depth_value = self._sample_depth(depth_map, u, v) if depth_map is not None else None
            should_update_descriptor = gray is not None and (
                track_id not in self.landmarks
                or self.landmarks[track_id].get("status") == "missing"
                or self._should_update_descriptors()
            )
            descriptor = self._extract_descriptor(gray, u, v) if should_update_descriptor else None
            dynamic_hit = self._sample_semantic_dynamic(semantic_mask, u, v)
            dynamic_evidence = dynamic_hit
            if dynamic_evidence_by_track and track_id in dynamic_evidence_by_track:
                dynamic_evidence = max(dynamic_hit, float(dynamic_evidence_by_track[track_id]))

            previous_landmark = self.landmarks.get(track_id, {})
            if previous_landmark:
                was_missing = previous_landmark.get("status") == "missing"
                if self._is_geometry_owned_landmark(previous_landmark):
                    world_point = np.array(previous_landmark["position_world"], dtype=np.float32)
                    position_source = previous_landmark.get("position_world_source", "geometry")
                else:
                    world_point = None
                    position_source = "uninitialized"
                hits = previous_landmark["hits"] + 1
                first_seen = previous_landmark.get("first_seen", self.frame_index)
                if was_missing:
                    self.lifecycle_stats["reobserved"] += 1
                else:
                    self.lifecycle_stats["updated"] += 1
            else:
                world_point = None
                position_source = "uninitialized"
                hits = 1
                first_seen = self.frame_index
                self.lifecycle_stats["created"] += 1

            previous_dynamic_score = float(previous_landmark.get("dynamic_score", 0.0))
            dynamic_score = (1.0 - SEMANTIC_DYNAMIC_EMA) * previous_dynamic_score + SEMANTIC_DYNAMIC_EMA * dynamic_evidence
            if previous_landmark.get("is_dynamic"):
                is_dynamic = dynamic_score >= SEMANTIC_DYNAMIC_OFF
            else:
                is_dynamic = dynamic_score >= SEMANTIC_DYNAMIC_ON

            age = self.frame_index - int(first_seen)
            quality = self._landmark_quality(hits, age, missed_frames=0)

            preserved = {
                key: previous_landmark[key]
                for key in (
                    "descriptor",
                    "observations",
                    "observation_count",
                    "mean_reprojection_error",
                    "is_stable",
                    "is_2d_stable",
                    "is_geometry_verified",
                    "geometry_baseline",
                    "geometry_ray_angle_deg",
                    "is_triangulated",
                    "triangulated_position_world",
                    "triangulated_reprojection_error",
                    "triangulated_depth_error",
                    "triangulated_depth_error_limit",
                    "triangulation_baseline",
                    "triangulation_angle_deg",
                    "position_world_depth_prior",
                    "position_world_source",
                    "is_local_map",
                    "ba_lite_refined",
                    "sliding_ba_refined",
                    "ba_lite_error_before",
                    "ba_lite_error_after",
                    "dynamic_score",
                    "is_dynamic",
                )
                if key in previous_landmark
            }
            landmark_update = {
                "id": track_id,
                "position_world": None if world_point is None else np.round(world_point, 4).tolist(),
                "position_world_depth_prior": None,
                "position_world_source": position_source,
                "image_xy": [round(u, 1), round(v, 1)],
                "depth": None if depth_value is None else round(float(depth_value), 4),
                "hits": hits,
                "first_seen": first_seen,
                "last_seen": self.frame_index,
                "age": age,
                "missed_frames": 0,
                "status": "visible",
                "quality": round(quality, 3),
                "dynamic_score": round(float(dynamic_score), 4),
                "is_dynamic": bool(is_dynamic),
            }
            landmark_update.update(preserved)
            if descriptor is not None:
                landmark_update["descriptor"] = descriptor
            self.landmarks[track_id] = landmark_update
            visible_ids.add(track_id)

        self._mark_unseen_landmarks_missing(visible_ids)
        self._prune_landmarks()
        return self._export_visible_map()

    def refine_visible_landmarks(self, depth_map, intrinsics, camera_pose: dict, semantic_mask=None) -> dict:
        """Refresh currently visible landmarks using a corrected depth map."""
        if (
            depth_map is None
            or intrinsics is None
            or self.prev_points is None
            or len(self.prev_points) == 0
            or not self.prev_track_ids
        ):
            return camera_pose

        points = self.prev_points.reshape(-1, 2)
        visible_ids = set()
        for track_id, pt in zip(self.prev_track_ids, points):
            if track_id not in self.landmarks:
                continue

            u = float(pt[0])
            v = float(pt[1])
            depth_value = self._sample_depth(depth_map, u, v) if depth_map is not None else None
            should_update_descriptor = (
                self.prev_gray is not None
                and self._should_update_descriptors()
            )
            descriptor = self._extract_descriptor(self.prev_gray, u, v) if should_update_descriptor else None
            landmark = self.landmarks[track_id]
            dynamic_hit = self._sample_semantic_dynamic(semantic_mask, u, v)
            previous_dynamic_score = float(landmark.get("dynamic_score", 0.0))
            dynamic_score = (1.0 - SEMANTIC_DYNAMIC_EMA) * previous_dynamic_score + SEMANTIC_DYNAMIC_EMA * dynamic_hit
            if landmark.get("is_dynamic"):
                is_dynamic = dynamic_score >= SEMANTIC_DYNAMIC_OFF
            else:
                is_dynamic = dynamic_score >= SEMANTIC_DYNAMIC_ON

            self.landmarks[track_id]["image_xy"] = [round(u, 1), round(v, 1)]
            self.landmarks[track_id]["depth"] = None if depth_value is None else round(float(depth_value), 4)
            if descriptor is not None:
                self.landmarks[track_id]["descriptor"] = descriptor
            self.landmarks[track_id]["last_seen"] = self.frame_index
            self.landmarks[track_id]["status"] = "visible"
            self.landmarks[track_id]["missed_frames"] = 0
            first_seen = int(self.landmarks[track_id].get("first_seen", self.frame_index))
            age = self.frame_index - first_seen
            self.landmarks[track_id]["age"] = age
            self.landmarks[track_id]["quality"] = round(
                self._landmark_quality(int(self.landmarks[track_id].get("hits", 0)), age, 0),
                3,
            )
            self.landmarks[track_id]["dynamic_score"] = round(float(dynamic_score), 4)
            self.landmarks[track_id]["is_dynamic"] = bool(is_dynamic)
            visible_ids.add(track_id)

        self._mark_unseen_landmarks_missing(visible_ids)
        self._prune_landmarks()
        refined_pose = dict(camera_pose)
        geometric_inlier_ids = set(int(item) for item in refined_pose.get("geometric_inlier_ids", []) or [])
        pose_source = str(refined_pose.get("pose_source", "unknown"))
        keyframe_visible_ids = visible_ids
        if pose_source == "essential" and geometric_inlier_ids:
            keyframe_visible_ids = visible_ids.intersection(geometric_inlier_ids)
        self._maybe_add_keyframe(
            keyframe_visible_ids,
            len(keyframe_visible_ids),
            pose_source,
            intrinsics,
        )
        sparse_map = self._export_visible_map()
        refined_pose["sparse_map"] = sparse_map
        refined_pose["local_sparse_map"] = self._export_local_visible_map()
        refined_pose["persistent_map"] = self._export_persistent_map()
        refined_pose["visible_landmark_count"] = len(sparse_map)
        refined_pose["local_visible_landmark_count"] = len(refined_pose["local_sparse_map"])
        refined_pose["sparse_landmark_count"] = len(self.landmarks)
        refined_pose["persistent_landmark_count"] = len(refined_pose["persistent_map"])
        refined_pose["missing_landmark_count"] = self._count_landmarks("missing")
        refined_pose["landmark_lifecycle"] = dict(self.lifecycle_stats)
        refined_pose["keyframes"] = len(self.keyframes)
        refined_pose["stable_landmark_count"] = self._stable_landmark_count()
        refined_pose["stable_2d_landmark_count"] = self._stable_2d_landmark_count()
        refined_pose["geometry_verified_landmark_count"] = self._geometry_verified_landmark_count()
        refined_pose["triangulated_landmark_count"] = self._triangulated_landmark_count()
        refined_pose["dynamic_landmark_count"] = self._dynamic_landmark_count()
        refined_pose["triangulation"] = dict(self.triangulation_stats)
        refined_pose["mean_stable_reprojection_error"] = self._mean_stable_reprojection_error()
        refined_pose["latest_keyframe"] = self.keyframes[-1] if self.keyframes else None
        refined_pose["covisibility_edges"] = self._covisibility_edge_count()
        refined_pose["latest_covisible_keyframes"] = self._latest_covisible_keyframes()
        refined_pose["local_keyframes"] = list(self.local_keyframe_ids)
        refined_pose["local_landmark_count"] = len(self.local_landmark_ids)
        refined_pose["local_keyframe_baseline"] = self._local_keyframe_baseline()
        refined_pose["ba_lite"] = dict(self.ba_lite_stats)
        refined_pose["sliding_ba"] = dict(self.sliding_ba_stats)
        return refined_pose

    def update(self, frame: np.ndarray, depth_map=None, intrinsics=None, semantic_mask=None) -> dict:
        """Return a sparse-motion pose estimate and sparse landmark map."""
        gray = self._prepare(frame)
        self.frame_index += 1

        if self.prev_gray is None:
            self.prev_gray = gray
            self._append_new_tracks(gray)
            return self._pose_dict(
                status="initialized",
                tracking_quality=1.0,
                image_shift_px=[0.0, 0.0],
                delta_translation_world=[0.0, 0.0, 0.0],
                sparse_map=[],
            )

        if self.prev_points is None or len(self.prev_points) < 4:
            self.prev_points = None
            self.prev_track_ids = []
            self._append_new_tracks(self.prev_gray)

        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_gray = gray
            self._append_new_tracks(gray)
            return self._pose_dict(
                status="reseeded",
                tracking_quality=0.0,
                image_shift_px=[0.0, 0.0],
                delta_translation_world=[0.0, 0.0, 0.0],
                sparse_map=[],
            )

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if next_points is None or status is None:
            self.prev_gray = gray
            self.prev_points = None
            self.prev_track_ids = []
            self._append_new_tracks(gray)
            return self._pose_dict(
                status="tracking-lost",
                tracking_quality=0.0,
                image_shift_px=[0.0, 0.0],
                delta_translation_world=[0.0, 0.0, 0.0],
                sparse_map=[],
            )

        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.prev_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if back_points is not None and back_status is not None:
            fb_error = np.linalg.norm(
                back_points.reshape(-1, 2) - self.prev_points.reshape(-1, 2),
                axis=1,
            )
            good_mask = (
                (status.reshape(-1) == 1)
                & (back_status.reshape(-1) == 1)
                & (fb_error <= LK_FORWARD_BACKWARD_MAX_ERROR_PX)
            )
        else:
            fb_error = np.full((len(status.reshape(-1)),), np.inf, dtype=np.float32)
            good_mask = status.reshape(-1) == 1
        good_new = next_points.reshape(-1, 2)[good_mask]
        good_old = self.prev_points.reshape(-1, 2)[good_mask]
        good_ids = [track_id for track_id, ok in zip(self.prev_track_ids, good_mask.tolist()) if ok]

        dynamic_evidence_by_track = self._track_dynamic_evidence(
            good_new,
            good_old,
            good_ids,
            semantic_mask=semantic_mask,
        )
        evidence_values = np.asarray(
            [float(dynamic_evidence_by_track.get(track_id, 0.0)) for track_id in good_ids],
            dtype=np.float32,
        )
        semantic_static_mask = evidence_values < DYNAMIC_EVIDENCE_GATE
        static_good_new = good_new[semantic_static_mask]
        static_good_old = good_old[semantic_static_mask]
        static_good_ids = [track_id for track_id, ok in zip(good_ids, semantic_static_mask.tolist()) if ok]
        geometry_new = static_good_new if len(static_good_new) >= 4 else good_new
        geometry_old = static_good_old if len(static_good_old) >= 4 else good_old
        geometry_ids = static_good_ids if len(static_good_ids) >= 4 else good_ids

        if len(geometry_new) >= 4:
            shifts = geometry_new - geometry_old
            dx_px = float(np.median(shifts[:, 0]))
            dy_px = float(np.median(shifts[:, 1]))
            flow_delta = np.array(
                [-dx_px / max(frame.shape[1], 1), -dy_px / max(frame.shape[0], 1), 0.0],
                dtype=np.float32,
            )
            tracking_quality = float(min(1.0, len(geometry_new) / max(len(self.prev_track_ids), 1)))
            status_text = "tracking"
        else:
            dx_px = 0.0
            dy_px = 0.0
            flow_delta = np.zeros(3, dtype=np.float32)
            tracking_quality = 0.0
            status_text = "low_confidence"

        prev_camera_position = self.camera_position_world.copy()
        prev_rotation_wc = self.rotation_wc.copy()
        pose_source = "flow"
        pnp_inliers = 0
        pnp_error = None
        geometric_inlier_ids = []
        self.last_pnp_anchor_scope = "none"
        pnp_pose = self._estimate_pose_from_anchors(geometry_new, geometry_ids, intrinsics)
        relative_pose = self._estimate_relative_pose_from_flow(geometry_old, geometry_new, intrinsics)
        pnp_position_ok = False
        if pnp_pose is not None:
            pnp_jump = float(np.linalg.norm(pnp_pose["camera_position_world"] - prev_camera_position))
            pnp_error_ok = (
                pnp_pose["reprojection_error"] is None
                or pnp_pose["reprojection_error"] <= MAX_PNP_REPROJECTION_ERROR
            )
            pnp_position_ok = pnp_jump <= MAX_PNP_POSITION_JUMP

        pnp_lock_ready = (
            pnp_pose is not None
            and pnp_pose["inliers"] >= PNP_LOCK_MIN_INLIERS
            and pnp_error_ok
            and pnp_position_ok
            and (
                pnp_pose["reprojection_error"] is None
                or pnp_pose["reprojection_error"] <= PNP_LOCK_MAX_REPROJECTION_ERROR
            )
        )
        pnp_basic_ready = (
            pnp_pose is not None
            and pnp_pose["inliers"] >= PNP_MIN_INLIERS
            and pnp_error_ok
            and pnp_position_ok
        )

        if pnp_lock_ready or pnp_basic_ready:
            self.rotation_wc = pnp_pose["rotation_wc"]
            self.camera_position_world = pnp_pose["camera_position_world"]
            pnp_inliers = pnp_pose["inliers"]
            pnp_error = pnp_pose["reprojection_error"]
            self.last_pnp_anchor_scope = pnp_pose.get("anchor_scope", "unknown")
            geometric_inlier_ids = pnp_pose.get("inlier_track_ids", [])
            pnp_delta = float(np.linalg.norm(self.camera_position_world - prev_camera_position))
            self.essential_translation_scale = float(
                np.clip(0.8 * self.essential_translation_scale + 0.2 * pnp_delta, 0.005, RELATIVE_POSE_MAX_TRANSLATION)
            )
            self.frames_since_pnp_lock = 0
            pose_source = "pnp"
        elif relative_pose is not None:
            rotation_21 = relative_pose["rotation_21"]
            translation_21 = relative_pose["translation_21"]
            self.rotation_wc = (prev_rotation_wc @ rotation_21.T).astype(np.float32)
            camera_delta_world = prev_rotation_wc @ (-rotation_21.T @ translation_21)
            self.frames_since_pnp_lock += 1
            essential_scale = float(relative_pose["scale"]) * ESSENTIAL_FALLBACK_SCALE_DAMPING
            essential_scale = float(np.clip(essential_scale, 0.0, ESSENTIAL_FALLBACK_MAX_TRANSLATION))
            if self.frames_since_pnp_lock >= ESSENTIAL_ROTATION_ONLY_AFTER_MISSED_PNP:
                essential_scale = float(min(essential_scale, 0.006))
            self.camera_position_world = (
                prev_camera_position + camera_delta_world.astype(np.float32) * essential_scale
            )
            pnp_inliers = int(relative_pose["inliers"])
            relative_inlier_mask = relative_pose.get("inlier_mask")
            if relative_inlier_mask is not None:
                geometric_inlier_ids = [
                    int(track_id)
                    for track_id, ok in zip(geometry_ids, relative_inlier_mask.tolist())
                    if ok
                ]
            pose_source = "essential"
        else:
            self.frames_since_pnp_lock += 1
            self.camera_position_world += flow_delta
            self.rotation_wc = prev_rotation_wc

        delta = self.camera_position_world - prev_camera_position

        sparse_map = self._update_landmarks(
            good_new,
            good_ids,
            depth_map,
            intrinsics,
            gray=gray,
            semantic_mask=semantic_mask,
            dynamic_evidence_by_track=dynamic_evidence_by_track,
        )

        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2).astype(np.float32) if len(good_new) else None
        self.prev_track_ids = good_ids
        self._append_new_tracks(gray)

        pose = self._pose_dict(
            status=status_text,
            tracking_quality=tracking_quality,
            image_shift_px=[dx_px, dy_px],
            delta_translation_world=delta,
            sparse_map=sparse_map,
            pose_source=pose_source,
            pnp_inliers=pnp_inliers,
            pnp_reprojection_error=pnp_error,
        )
        pose["geometric_inlier_ids"] = geometric_inlier_ids
        pose["geometric_inlier_count"] = len(geometric_inlier_ids)
        return pose
