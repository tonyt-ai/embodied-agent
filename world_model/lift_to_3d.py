"""Utilities for lifting 2D detections into approximate 3D positions."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def infer_camera_intrinsics(width: int, height: int) -> dict:
    """Infer a simple pinhole camera model from image dimensions."""
    focal = float(max(width, height))
    return {
        "fx": focal,
        "fy": focal,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "width": width,
        "height": height,
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


def lift_bbox_to_3d(bbox, depth_map: np.ndarray, camera_pose: dict, intrinsics: dict) -> dict:
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
    world_translation = np.array(
        camera_pose.get("translation_world", [0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    world_point = camera_point + world_translation

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
    }
