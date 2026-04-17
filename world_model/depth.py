"""Lightweight monocular depth heuristics for real-time prototyping.

This module avoids heavyweight depth dependencies so the current demo
can expose an approximate 3D signal with the packages already present
in the repository. The returned depth map is not metric or physically
accurate; it is a normalized, scene-relative estimate suitable for
early UI debugging and 3D state plumbing.
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_depth(frame: np.ndarray) -> np.ndarray:
    """Estimate a smooth pseudo-depth map from a BGR frame.

    The heuristic combines:
    - a vertical prior (lower pixels are likely closer in egocentric views)
    - local sharpness / edge strength
    - local brightness normalization

    Returns a float32 depth map with values roughly in the range [0.35, 2.5],
    where smaller values are closer to the camera.
    """
    if frame is None or frame.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape

    vertical = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(h, 1)
    vertical = np.repeat(vertical, w, axis=1)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    sharpness = cv2.GaussianBlur(np.abs(lap), (0, 0), 1.2)
    sharpness = sharpness / (float(sharpness.max()) + 1e-6)

    local_brightness = cv2.GaussianBlur(gray, (0, 0), 5.0)
    brightness_inv = 1.0 - local_brightness

    nearness = 0.60 * vertical + 0.25 * sharpness + 0.15 * brightness_inv
    nearness = np.clip(nearness, 0.0, 1.0)

    depth = 2.5 - 2.15 * nearness
    depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 2.0)
    return depth.astype(np.float32)


def summarize_depth(depth_map: np.ndarray) -> dict:
    """Return compact debug statistics for the current depth map."""
    if depth_map is None or depth_map.size == 0:
        return {
            "min_depth": 0.0,
            "max_depth": 0.0,
            "mean_depth": 0.0,
        }

    return {
        "min_depth": round(float(depth_map.min()), 3),
        "max_depth": round(float(depth_map.max()), 3),
        "mean_depth": round(float(depth_map.mean()), 3),
    }
