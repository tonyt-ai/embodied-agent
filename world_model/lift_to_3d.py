"""Utilities for lifting 2D detections into approximate 3D positions."""

from __future__ import annotations

import json
import math
import os
from typing import Tuple

import numpy as np

_CALIBRATION_CACHE = {}


def _calibration_path_from_env() -> str:
    env_path = os.environ.get("CAMERA_CALIBRATION_FILE")
    if env_path:
        return env_path
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "data", "camera_calibration.json")


def _load_calibration_json(path: str) -> dict | None:
    if not path:
        return None
    try:
        abs_path = os.path.abspath(path)
        stat = os.stat(abs_path)
    except OSError:
        return None

    cache_key = abs_path
    cached = _CALIBRATION_CACHE.get(cache_key)
    if cached and cached.get("mtime_ns") == stat.st_mtime_ns:
        return cached.get("payload")

    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None

    _CALIBRATION_CACHE[cache_key] = {
        "mtime_ns": stat.st_mtime_ns,
        "payload": payload,
    }
    return payload


def _intrinsics_from_calibration(calibration: dict, width: int, height: int, source_path: str) -> dict | None:
    matrix = calibration.get("camera_matrix")
    if not matrix or len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        return None

    try:
        fx = float(matrix[0][0])
        fy = float(matrix[1][1])
        cx = float(matrix[0][2])
        cy = float(matrix[1][2])
    except (TypeError, ValueError, IndexError):
        return None

    image_size = calibration.get("image_size") or {}
    calib_width = int(image_size.get("width", width))
    calib_height = int(image_size.get("height", height))

    if calib_width > 0 and calib_height > 0 and (calib_width != width or calib_height != height):
        scale_x = float(width) / float(calib_width)
        scale_y = float(height) / float(calib_height)
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y

    dist = calibration.get("distortion_coefficients") or []
    try:
        dist = [float(value) for value in dist]
    except (TypeError, ValueError):
        dist = []

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "source": "calibration-json",
        "fov_deg": 0.0,
        "camera_matrix": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": dist,
        "calibration_file": os.path.abspath(source_path),
        "calibration_image_size": {
            "width": calib_width,
            "height": calib_height,
        },
    }


def infer_camera_intrinsics(width: int, height: int) -> dict:
    """Infer or load a simple pinhole camera model from image dimensions.

    Env overrides are intentionally lightweight for webcam calibration:
    CAMERA_FX/CAMERA_FY/CAMERA_CX/CAMERA_CY win over CAMERA_FOV_DEG.
    """
    calibration_path = _calibration_path_from_env()
    calibration = _load_calibration_json(calibration_path)
    if calibration is not None:
        from_calibration = _intrinsics_from_calibration(calibration, width, height, calibration_path)
        if from_calibration is not None:
            return from_calibration

    fx_env = os.environ.get("CAMERA_FX")
    fy_env = os.environ.get("CAMERA_FY")
    cx_env = os.environ.get("CAMERA_CX")
    cy_env = os.environ.get("CAMERA_CY")
    fov_env = os.environ.get("CAMERA_FOV_DEG")

    source = "default-c920-fov"
    fov_deg = 70.42
    if fov_env:
        try:
            fov_deg = float(fov_env)
            source = "env-fov"
        except ValueError:
            fov_deg = 70.42

    focal = float((width * 0.5) / max(math.tan(math.radians(fov_deg) * 0.5), 1e-6))
    fx = focal
    fy = focal
    cx = width / 2.0
    cy = height / 2.0

    if fx_env or fy_env or cx_env or cy_env:
        source = "env-intrinsics"
        try:
            fx = float(fx_env) if fx_env else fx
            fy = float(fy_env) if fy_env else fy
            cx = float(cx_env) if cx_env else cx
            cy = float(cy_env) if cy_env else cy
        except ValueError:
            source = "default-c920-fov"
            fx = focal
            fy = focal
            cx = width / 2.0
            cy = height / 2.0

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "source": source,
        "fov_deg": fov_deg,
        "camera_matrix": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": [],
    }


def _bbox_to_pixel_region(bbox, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(bbox[0] * width))))
    y1 = max(0, min(height - 1, int(round(bbox[1] * height))))
    x2 = max(0, min(width, int(round(bbox[2] * width))))
    y2 = max(0, min(height, int(round(bbox[3] * height))))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def sample_depth_for_bbox(depth_map: np.ndarray, bbox) -> tuple[float, float]:
    """Sample robust depth from the middle of a bbox."""
    height, width = depth_map.shape[:2]
    x1, y1, x2, y2 = _bbox_to_pixel_region(bbox, width, height)

    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, 0.0

    mid_x1 = crop.shape[1] // 4
    mid_x2 = max(mid_x1 + 1, crop.shape[1] * 3 // 4)
    mid_y1 = crop.shape[0] // 4
    mid_y2 = max(mid_y1 + 1, crop.shape[0] * 3 // 4)
    center_crop = crop[mid_y1:mid_y2, mid_x1:mid_x2]
    region = center_crop if center_crop.size > 0 else crop

    depth_value = float(np.median(region))
    spread = float(np.std(region))
    confidence = 1.0 / (1.0 + spread * 8.0)
    confidence = max(0.0, min(1.0, confidence))
    return depth_value, confidence


def _landmarks_in_bbox(sparse_points, bbox, width: int, height: int):
    if not sparse_points:
        return []

    x1, y1, x2, y2 = _bbox_to_pixel_region(bbox, width, height)
    matches = []
    for point in sparse_points:
        image_xy = point.get("image_xy")
        position_world = point.get("position_world")
        if not image_xy or not position_world or len(image_xy) < 2 or len(position_world) < 3:
            continue

        u, v = float(image_xy[0]), float(image_xy[1])
        if x1 <= u <= x2 and y1 <= v <= y2:
            matches.append(point)

    matches.sort(key=lambda item: (item.get("hits", 0), item.get("last_seen", 0)), reverse=True)
    return matches


def lift_bbox_to_3d(
    bbox,
    depth_map: np.ndarray,
    camera_pose: dict,
    intrinsics: dict,
    sparse_points=None,
) -> dict:
    """Convert a normalized bbox into camera-frame and world-frame 3D points."""
    height, width = depth_map.shape[:2]
    x1, y1, x2, y2 = _bbox_to_pixel_region(bbox, width, height)

    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0

    depth_value, depth_confidence = sample_depth_for_bbox(depth_map, bbox)

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    x_cam = ((u - cx) / max(fx, 1e-6)) * depth_value
    y_cam = ((v - cy) / max(fy, 1e-6)) * depth_value
    z_cam = depth_value

    camera_point = np.array([x_cam, y_cam, z_cam], dtype=np.float32)
    camera_position_world = np.array(
        camera_pose.get(
            "camera_position_world",
            camera_pose.get("translation_world", [0.0, 0.0, 0.0]),
        ),
        dtype=np.float32,
    )
    rotation_wc = np.array(
        camera_pose.get("rotation_wc", np.eye(3, dtype=np.float32)),
        dtype=np.float32,
    )
    if rotation_wc.shape != (3, 3):
        rotation_wc = np.eye(3, dtype=np.float32)

    world_point = rotation_wc @ camera_point + camera_position_world
    landmark_matches = _landmarks_in_bbox(sparse_points, bbox, width, height)
    landmark_support = len(landmark_matches)
    landmark_blend_weight = 0.0

    if landmark_matches:
        landmark_world = np.array(
            [
                np.array(item["position_world"], dtype=np.float32)
                for item in landmark_matches[:12]
            ],
            dtype=np.float32,
        )
        landmark_point = landmark_world.mean(axis=0)
        landmark_blend_weight = min(0.65, 0.15 + 0.1 * min(landmark_support, 5))
        world_point = (1.0 - landmark_blend_weight) * world_point + landmark_blend_weight * landmark_point
        camera_point = rotation_wc.T @ (world_point - camera_position_world)
        depth_value = float(camera_point[2])
        depth_confidence = max(depth_confidence, min(1.0, 0.35 + 0.1 * landmark_support))

    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    image_area = float(width * height)
    size_ratio = float((bbox_w * bbox_h) / max(image_area, 1.0))

    return {
        "position_camera_3d": np.round(camera_point, 4).tolist(),
        "position_world_3d": np.round(world_point, 4).tolist(),
        "depth": round(depth_value, 4),
        "depth_confidence": round(depth_confidence, 3),
        "bbox_area_ratio": round(size_ratio, 5),
        "pixel_center": [round(u, 1), round(v, 1)],
        "landmark_support": landmark_support,
        "landmark_blend_weight": round(landmark_blend_weight, 3),
    }
