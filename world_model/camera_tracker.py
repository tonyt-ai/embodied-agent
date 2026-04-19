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

MAX_PNP_REPROJECTION_ERROR = 10.0
MAX_PNP_POSITION_JUMP = 0.75


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

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.prev_track_ids = []
        self.frame_index = 0
        self.next_track_id = 1
        self.camera_position_world = np.zeros(3, dtype=np.float32)
        self.rotation_wc = np.eye(3, dtype=np.float32)
        self.landmarks = {}

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
            "pnp_inliers": int(pnp_inliers),
            "pnp_reprojection_error": None if pnp_reprojection_error is None else round(float(pnp_reprojection_error), 4),
            "sparse_map": sparse_map,
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

        for _ in range(len(new_points)):
            self.prev_track_ids.append(self.next_track_id)
            self.next_track_id += 1

    def _sample_depth(self, depth_map: np.ndarray, u: float, v: float) -> float:
        h, w = depth_map.shape[:2]
        x = int(np.clip(round(u), 0, w - 1))
        y = int(np.clip(round(v), 0, h - 1))
        return float(depth_map[y, x])

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

    def _update_landmarks(self, points, track_ids, depth_map, intrinsics):
        if depth_map is None or intrinsics is None or points is None or len(points) == 0:
            return []

        visible = []
        for track_id, pt in zip(track_ids, points):
            u = float(pt[0])
            v = float(pt[1])
            depth_value = self._sample_depth(depth_map, u, v)
            world_point = self._lift_pixel(u, v, depth_value, intrinsics)

            if track_id in self.landmarks:
                prev = np.array(self.landmarks[track_id]["position_world"], dtype=np.float32)
                world_point = 0.7 * prev + 0.3 * world_point
                hits = self.landmarks[track_id]["hits"] + 1
            else:
                hits = 1

            self.landmarks[track_id] = {
                "id": track_id,
                "position_world": np.round(world_point, 4).tolist(),
                "image_xy": [round(u, 1), round(v, 1)],
                "depth": round(depth_value, 4),
                "hits": hits,
                "last_seen": self.frame_index,
            }
            visible.append(self.landmarks[track_id])

        stale_ids = [
            track_id
            for track_id, item in self.landmarks.items()
            if self.frame_index - item["last_seen"] > 30
        ]
        for track_id in stale_ids:
            del self.landmarks[track_id]

        visible.sort(key=lambda item: (item["hits"], item["last_seen"]), reverse=True)
        return visible[:80]

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
        visible = []
        for track_id, pt in zip(self.prev_track_ids, points):
            if track_id not in self.landmarks:
                continue

            u = float(pt[0])
            v = float(pt[1])
            depth_value = self._sample_depth(depth_map, u, v)
            world_point = self._lift_pixel(u, v, depth_value, intrinsics)
            prev = np.array(self.landmarks[track_id]["position_world"], dtype=np.float32)
            world_point = 0.75 * prev + 0.25 * world_point

            self.landmarks[track_id]["position_world"] = np.round(world_point, 4).tolist()
            self.landmarks[track_id]["image_xy"] = [round(u, 1), round(v, 1)]
            self.landmarks[track_id]["depth"] = round(depth_value, 4)
            self.landmarks[track_id]["last_seen"] = self.frame_index
            visible.append(self.landmarks[track_id])

        visible.sort(key=lambda item: (item["hits"], item["last_seen"]), reverse=True)
        sparse_map = visible[:80]
        refined_pose = dict(camera_pose)
        refined_pose["sparse_map"] = sparse_map
        refined_pose["visible_landmark_count"] = len(sparse_map)
        refined_pose["sparse_landmark_count"] = len(self.landmarks)
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

        sparse_map = self._update_landmarks(good_new, good_ids, depth_map, intrinsics)

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
