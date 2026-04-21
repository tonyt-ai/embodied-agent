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

MAX_PNP_REPROJECTION_ERROR = 10.0
MAX_PNP_POSITION_JUMP = 0.75
MAX_MISSING_FRAMES = 180
MAX_LANDMARKS = 600
MIN_QUALITY_FOR_PERSISTENCE = 0.2
MIN_HITS_FOR_PERSISTENCE = 2
XFEAT_TOP_K = 512
XFEAT_UPDATE_EVERY = 4
XFEAT_MATCH_RADIUS_PX = 12.0
MIN_REOBSERVATION_SIMILARITY = 0.78
MAX_REOBSERVATION_DISTANCE_PX = 90.0


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
        xfeat_repo = os.environ.get("XFEAT_REPO")
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

    def debug_info(self) -> dict:
        return {
            "backend": "xfeat",
            "status": self.status,
            "device": self.device,
            "top_k": self.top_k,
            "update_every": XFEAT_UPDATE_EVERY,
            "last_error": self.last_error,
        }


class CameraTracker:
    """Estimate camera motion and maintain a sparse landmark map."""

    def __init__(self, max_points: int = 120, min_points: int = 40):
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
        self.lifecycle_stats = {
            "created": 0,
            "updated": 0,
            "reobserved": 0,
            "descriptor_reassociated": 0,
            "marked_missing": 0,
            "pruned": 0,
        }

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.prev_track_ids = []
        self.frame_index = 0
        self.next_track_id = 1
        self.camera_position_world = np.zeros(3, dtype=np.float32)
        self.rotation_wc = np.eye(3, dtype=np.float32)
        self.landmarks = {}
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
            "pnp_inliers": int(pnp_inliers),
            "pnp_reprojection_error": None if pnp_reprojection_error is None else round(float(pnp_reprojection_error), 4),
            "sparse_map": sparse_map,
            "persistent_map": self._export_persistent_map(),
        }

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _detect_features(self, gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_points,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )

    def _append_new_tracks(self, gray: np.ndarray):
        existing = 0 if self.prev_points is None else len(self.prev_points)
        needed = self.max_points - existing
        if needed <= 0:
            return

        new_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=needed,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )
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
            if int(landmark.get("missed_frames", 0)) > MAX_MISSING_FRAMES:
                continue
            if "descriptor" not in landmark:
                continue

            image_xy = landmark.get("image_xy")
            if image_xy and len(image_xy) >= 2:
                distance = float(np.hypot(u - float(image_xy[0]), v - float(image_xy[1])))
                if distance > MAX_REOBSERVATION_DISTANCE_PX:
                    continue

            similarity = self._descriptor_similarity(descriptor, landmark.get("descriptor"))
            if similarity < MIN_REOBSERVATION_SIMILARITY:
                continue
            recency_bonus = max(0.0, 1.0 - int(landmark.get("missed_frames", 0)) / max(MAX_MISSING_FRAMES, 1))
            score = similarity + 0.05 * recency_bonus
            if score > best_score:
                best_score = score
                best_id = track_id

        return best_id, descriptor

    def _public_landmark(self, landmark: dict) -> dict:
        return {
            key: value
            for key, value in landmark.items()
            if key != "descriptor"
        }

    def _count_landmarks(self, status: str) -> int:
        if status == "persistent":
            return sum(
                1 for item in self.landmarks.values()
                if item.get("status") in {"visible", "missing"}
                and int(item.get("hits", 0)) >= MIN_HITS_FOR_PERSISTENCE
            )
        return sum(1 for item in self.landmarks.values() if item.get("status") == status)

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
            landmark["quality"] = round(
                self._landmark_quality(
                    int(landmark.get("hits", 0)),
                    age,
                    int(landmark.get("missed_frames", 0)),
                ),
                3,
            )

    def _prune_landmarks(self):
        stale_ids = []
        for track_id, landmark in self.landmarks.items():
            missed = int(landmark.get("missed_frames", 0))
            if missed > MAX_MISSING_FRAMES:
                stale_ids.append(track_id)

        if len(self.landmarks) - len(stale_ids) > MAX_LANDMARKS:
            candidates = sorted(
                (
                    (float(item.get("quality", 0.0)), int(item.get("last_seen", 0)), track_id)
                    for track_id, item in self.landmarks.items()
                    if track_id not in stale_ids
                ),
                key=lambda item: (item[0], item[1]),
            )
            overflow = len(self.landmarks) - len(stale_ids) - MAX_LANDMARKS
            stale_ids.extend(track_id for _, _, track_id in candidates[:overflow])

        for track_id in stale_ids:
            if track_id in self.landmarks:
                del self.landmarks[track_id]
                self.lifecycle_stats["pruned"] += 1

    def _export_visible_map(self):
        visible = [
            self._public_landmark(item) for item in self.landmarks.values()
            if item.get("status") == "visible"
        ]
        visible.sort(key=lambda item: (item.get("quality", 0.0), item.get("hits", 0), item.get("last_seen", 0)), reverse=True)
        return visible[:80]

    def _export_persistent_map(self):
        persistent = [
            self._public_landmark(item) for item in self.landmarks.values()
            if item.get("status") in {"visible", "missing"}
            and int(item.get("hits", 0)) >= MIN_HITS_FOR_PERSISTENCE
        ]
        persistent.sort(
            key=lambda item: (
                item.get("status") == "visible",
                item.get("quality", 0.0),
                item.get("hits", 0),
                item.get("last_seen", 0),
            ),
            reverse=True,
        )
        return persistent[:240]

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

    def _estimate_pose_from_anchors(self, points, track_ids, intrinsics):
        if intrinsics is None or points is None or len(points) < 6:
            return None

        object_points = []
        image_points = []
        for track_id, pt in zip(track_ids, points):
            landmark = self.landmarks.get(track_id)
            if not landmark:
                continue
            world_point = landmark.get("position_world")
            if not world_point or len(world_point) < 3:
                continue
            object_points.append(world_point)
            image_points.append([float(pt[0]), float(pt[1])])

        if len(object_points) < 6:
            return None

        object_points = np.asarray(object_points, dtype=np.float32)
        image_points = np.asarray(image_points, dtype=np.float32)
        camera_matrix = self._camera_matrix(intrinsics)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

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

        rotation_cw, _ = cv2.Rodrigues(rvec)
        rotation_wc = rotation_cw.T.astype(np.float32)
        camera_position_world = (-rotation_wc @ tvec.reshape(3)).astype(np.float32)

        reprojection_error = None
        if inliers is not None and len(inliers) > 0:
            inlier_ids = inliers.reshape(-1)
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
        }

    def _update_landmarks(self, points, track_ids, depth_map, intrinsics, gray=None):
        if depth_map is None or intrinsics is None or points is None or len(points) == 0:
            self._mark_unseen_landmarks_missing(set())
            self._prune_landmarks()
            return self._export_visible_map()

        visible_ids = set()
        for track_id, pt in zip(track_ids, points):
            u = float(pt[0])
            v = float(pt[1])
            depth_value = self._sample_depth(depth_map, u, v)
            world_point = self._lift_pixel(u, v, depth_value, intrinsics)
            should_update_descriptor = gray is not None and (
                track_id not in self.landmarks
                or self.landmarks[track_id].get("status") == "missing"
                or self._should_update_descriptors()
            )
            descriptor = self._extract_descriptor(gray, u, v) if should_update_descriptor else None

            if track_id in self.landmarks:
                was_missing = self.landmarks[track_id].get("status") == "missing"
                prev = np.array(self.landmarks[track_id]["position_world"], dtype=np.float32)
                world_point = 0.7 * prev + 0.3 * world_point
                hits = self.landmarks[track_id]["hits"] + 1
                first_seen = self.landmarks[track_id].get("first_seen", self.frame_index)
                if was_missing:
                    self.lifecycle_stats["reobserved"] += 1
                else:
                    self.lifecycle_stats["updated"] += 1
            else:
                hits = 1
                first_seen = self.frame_index
                self.lifecycle_stats["created"] += 1

            age = self.frame_index - int(first_seen)
            quality = self._landmark_quality(hits, age, missed_frames=0)

            self.landmarks[track_id] = {
                "id": track_id,
                "position_world": np.round(world_point, 4).tolist(),
                "image_xy": [round(u, 1), round(v, 1)],
                "depth": round(depth_value, 4),
                "hits": hits,
                "first_seen": first_seen,
                "last_seen": self.frame_index,
                "age": age,
                "missed_frames": 0,
                "status": "visible",
                "quality": round(quality, 3),
            }
            if descriptor is not None:
                self.landmarks[track_id]["descriptor"] = descriptor
            visible_ids.add(track_id)

        self._mark_unseen_landmarks_missing(visible_ids)
        self._prune_landmarks()
        return self._export_visible_map()

    def refine_visible_landmarks(self, depth_map, intrinsics, camera_pose: dict) -> dict:
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
            depth_value = self._sample_depth(depth_map, u, v)
            world_point = self._lift_pixel(u, v, depth_value, intrinsics)
            should_update_descriptor = (
                self.prev_gray is not None
                and self._should_update_descriptors()
            )
            descriptor = self._extract_descriptor(self.prev_gray, u, v) if should_update_descriptor else None
            prev = np.array(self.landmarks[track_id]["position_world"], dtype=np.float32)
            world_point = 0.75 * prev + 0.25 * world_point

            self.landmarks[track_id]["position_world"] = np.round(world_point, 4).tolist()
            self.landmarks[track_id]["image_xy"] = [round(u, 1), round(v, 1)]
            self.landmarks[track_id]["depth"] = round(depth_value, 4)
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
            visible_ids.add(track_id)

        self._mark_unseen_landmarks_missing(visible_ids)
        self._prune_landmarks()
        sparse_map = self._export_visible_map()
        refined_pose = dict(camera_pose)
        refined_pose["sparse_map"] = sparse_map
        refined_pose["persistent_map"] = self._export_persistent_map()
        refined_pose["visible_landmark_count"] = len(sparse_map)
        refined_pose["sparse_landmark_count"] = len(self.landmarks)
        refined_pose["persistent_landmark_count"] = len(refined_pose["persistent_map"])
        refined_pose["missing_landmark_count"] = self._count_landmarks("missing")
        refined_pose["landmark_lifecycle"] = dict(self.lifecycle_stats)
        return refined_pose

    def update(self, frame: np.ndarray, depth_map=None, intrinsics=None) -> dict:
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

        good_mask = status.reshape(-1) == 1
        good_new = next_points.reshape(-1, 2)[good_mask]
        good_old = self.prev_points.reshape(-1, 2)[good_mask]
        good_ids = [track_id for track_id, ok in zip(self.prev_track_ids, good_mask.tolist()) if ok]

        if len(good_new) >= 4:
            shifts = good_new - good_old
            dx_px = float(np.median(shifts[:, 0]))
            dy_px = float(np.median(shifts[:, 1]))
            flow_delta = np.array(
                [-dx_px / max(frame.shape[1], 1), -dy_px / max(frame.shape[0], 1), 0.0],
                dtype=np.float32,
            )
            tracking_quality = float(min(1.0, len(good_new) / max(len(self.prev_track_ids), 1)))
            status_text = "tracking"
        else:
            dx_px = 0.0
            dy_px = 0.0
            flow_delta = np.zeros(3, dtype=np.float32)
            tracking_quality = 0.0
            status_text = "low_confidence"

        prev_camera_position = self.camera_position_world.copy()
        pose_source = "flow"
        pnp_inliers = 0
        pnp_error = None
        pnp_pose = self._estimate_pose_from_anchors(good_new, good_ids, intrinsics)
        pnp_position_ok = False
        if pnp_pose is not None:
            pnp_jump = float(np.linalg.norm(pnp_pose["camera_position_world"] - prev_camera_position))
            pnp_error_ok = (
                pnp_pose["reprojection_error"] is None
                or pnp_pose["reprojection_error"] <= MAX_PNP_REPROJECTION_ERROR
            )
            pnp_position_ok = pnp_jump <= MAX_PNP_POSITION_JUMP

        if pnp_pose is not None and pnp_pose["inliers"] >= 6 and pnp_error_ok and pnp_position_ok:
            self.rotation_wc = pnp_pose["rotation_wc"]
            self.camera_position_world = pnp_pose["camera_position_world"]
            pnp_inliers = pnp_pose["inliers"]
            pnp_error = pnp_pose["reprojection_error"]
            pose_source = "pnp"
        else:
            self.camera_position_world += flow_delta
            self.rotation_wc = np.eye(3, dtype=np.float32)

        delta = self.camera_position_world - prev_camera_position

        sparse_map = self._update_landmarks(good_new, good_ids, depth_map, intrinsics, gray=gray)

        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2).astype(np.float32) if len(good_new) else None
        self.prev_track_ids = good_ids
        self._append_new_tracks(gray)

        return self._pose_dict(
            status=status_text,
            tracking_quality=tracking_quality,
            image_shift_px=[dx_px, dy_px],
            delta_translation_world=delta,
            sparse_map=sparse_map,
            pose_source=pose_source,
            pnp_inliers=pnp_inliers,
            pnp_reprojection_error=pnp_error,
        )
