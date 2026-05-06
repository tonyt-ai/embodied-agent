"""Depth estimation backend for the embodied-agent prototype.

This module calls a monocular depth model backed by Hugging Face
Transformers and Depth Anything. If the model or dependency cannot be
loaded, it falls back to a lightweight heuristic so the rest of the demo
can still run.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL_ID = os.environ.get("DEPTH_ANYTHING_MODEL", "LiheYoung/depth-anything-small-hf")
USE_HEURISTIC_ONLY = os.environ.get("DEPTH_BACKEND", "depth-anything").lower() == "heuristic"
DEPTH_LOCAL_ONLY = os.environ.get("DEPTH_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no"}

DEPTH_MIN_RANGE = 0.35
DEPTH_MAX_RANGE = 2.50
LOW_PERCENTILE = float(os.environ.get("DEPTH_LOW_PERCENTILE", "2.0"))
HIGH_PERCENTILE = float(os.environ.get("DEPTH_HIGH_PERCENTILE", "98.0"))
MIN_STABILIZATION_ANCHORS = int(os.environ.get("DEPTH_MIN_STABILIZATION_ANCHORS", "6"))
MIN_TRACKING_QUALITY = float(os.environ.get("DEPTH_MIN_TRACKING_QUALITY", "0.25"))
MIN_ACTIVE_TRACKS = int(os.environ.get("DEPTH_MIN_ACTIVE_TRACKS", "12"))
MAX_ALLOWED_RELATIVE_ERROR = float(os.environ.get("DEPTH_MAX_ALLOWED_RELATIVE_ERROR", "0.35"))
MAX_SHIFT_NORM_FOR_WEAK_ANCHORS = float(os.environ.get("DEPTH_MAX_SHIFT_NORM_FOR_WEAK_ANCHORS", "0.08"))


class DepthEstimator:
    """Lazy singleton wrapper around Depth Anything inference."""

    def __init__(self):
        self.model_id = DEFAULT_MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.backend = "heuristic"
        self.status = "heuristic-fallback"
        self.last_error: Optional[str] = None
        self.load_time_ms: Optional[float] = None
        self._lock = threading.Lock()
        self._loaded = False
        self._processor = None
        self._model = None
        self._last_norm_low: Optional[float] = None
        self._last_norm_high: Optional[float] = None

    def ensure_loaded(self):
        if self._loaded or USE_HEURISTIC_ONLY:
            return

        with self._lock:
            if self._loaded:
                return

            t0 = time.perf_counter()
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                self._processor = AutoImageProcessor.from_pretrained(
                    self.model_id,
                    local_files_only=DEPTH_LOCAL_ONLY,
                )
                self._model = AutoModelForDepthEstimation.from_pretrained(
                    self.model_id,
                    local_files_only=DEPTH_LOCAL_ONLY,
                )
                self._model = self._model.to(self.device)
                self._model.eval()
                self._loaded = True
                self.backend = "depth-anything"
                self.status = "ready"
                self.load_time_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                self.last_error = str(exc)
                self.backend = "heuristic"
                self.status = "load-failed"
                self._loaded = False
                self._processor = None
                self._model = None

    def _normalize_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        low = float(np.percentile(raw_depth, LOW_PERCENTILE))
        high = float(np.percentile(raw_depth, HIGH_PERCENTILE))
        if high <= low:
            high = low + 1e-6

        self._last_norm_low = low
        self._last_norm_high = high

        clipped = np.clip(raw_depth, low, high)
        normalized = (clipped - low) / (high - low + 1e-6)
        depth = DEPTH_MIN_RANGE + normalized * (DEPTH_MAX_RANGE - DEPTH_MIN_RANGE)
        return depth.astype(np.float32)

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros((1, 1), dtype=np.float32)

        if USE_HEURISTIC_ONLY:
            self.backend = "heuristic"
            self.status = "forced-heuristic"
            raw_depth = heuristic_depth(frame)
            return self._normalize_depth(raw_depth)

        self.ensure_loaded()
        if not self._loaded or self._processor is None or self._model is None:
            raw_depth = heuristic_depth(frame)
            return self._normalize_depth(raw_depth)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.device == "cuda":
            pixel_values = pixel_values.to(dtype=self.dtype)

        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values)

        post = self._processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        raw_depth = post[0]["predicted_depth"].detach().float().cpu().numpy().astype(np.float32)

        if raw_depth.ndim != 2:
            raw_depth = np.squeeze(raw_depth).astype(np.float32)

        return self._normalize_depth(raw_depth)

    def get_debug_info(self) -> dict:
        return {
            "backend": self.backend,
            "status": self.status,
            "model_id": self.model_id,
            "device": self.device,
            "load_time_ms": round(self.load_time_ms, 1) if self.load_time_ms is not None else None,
            "last_error": self.last_error,
            "norm_low": round(self._last_norm_low, 4) if self._last_norm_low is not None else None,
            "norm_high": round(self._last_norm_high, 4) if self._last_norm_high is not None else None,
            "low_percentile": LOW_PERCENTILE,
            "high_percentile": HIGH_PERCENTILE,
        }


def _sample_depth_value(depth_map: np.ndarray, u: float, v: float) -> float:
    h, w = depth_map.shape[:2]
    x = int(np.clip(round(u), 0, w - 1))
    y = int(np.clip(round(v), 0, h - 1))
    return float(depth_map[y, x])


class AnchorDepthStabilizer:
    """Stabilize per-frame depth using visible 3D anchors and camera pose."""

    def __init__(self):
        self.reset_count = 0
        self.last_reason = "cold-start"

    def reset(self, reason: str):
        self.reset_count += 1
        self.last_reason = reason

    def stabilize(self, depth_map: np.ndarray, camera_pose: dict | None) -> tuple[np.ndarray, dict]:
        debug = {
            "active": False,
            "reason": "no-pose",
            "anchors_considered": 0,
            "anchors_used": 0,
            "triangulated_anchors": 0,
            "stable_anchors": 0,
            "fit_scale": 1.0,
            "fit_offset": 0.0,
            "median_abs_error": None,
            "median_relative_error": None,
            "reset_count": self.reset_count,
            "anchor_source": "none",
        }
        if depth_map is None or depth_map.size == 0 or not camera_pose:
            return depth_map, debug

        status = str(camera_pose.get("status", "unknown"))
        tracking_quality = float(camera_pose.get("tracking_quality", 0.0))
        active_tracks = int(camera_pose.get("active_tracks", 0))
        local_sparse_map = camera_pose.get("local_sparse_map", []) or []
        sparse_map = camera_pose.get("sparse_map", []) or []
        camera_position_world = np.array(
            camera_pose.get(
                "camera_position_world",
                camera_pose.get("translation_world", [0.0, 0.0, 0.0]),
            ),
            dtype=np.float32,
        )
        rotation_cw = np.array(
            camera_pose.get("rotation_cw", np.eye(3, dtype=np.float32)),
            dtype=np.float32,
        )
        if rotation_cw.shape != (3, 3):
            rotation_cw = np.eye(3, dtype=np.float32)
        image_shift = camera_pose.get("image_shift_px", [0.0, 0.0]) or [0.0, 0.0]
        h, w = depth_map.shape[:2]
        diagonal = max(float(np.hypot(w, h)), 1.0)
        shift_norm = float(np.hypot(float(image_shift[0]), float(image_shift[1])) / diagonal)
        debug["image_shift_norm"] = round(shift_norm, 4)

        if status in {"tracking-lost", "reseeded"}:
            self.reset(status)
            debug["reason"] = status
            debug["reset_count"] = self.reset_count
            return depth_map, debug

        if tracking_quality < MIN_TRACKING_QUALITY or active_tracks < MIN_ACTIVE_TRACKS:
            debug["reason"] = "low-tracking-confidence"
            debug["reset_count"] = self.reset_count
            return depth_map, debug

        def collect_anchor_pairs(anchor_map, *, require_triangulated=False, require_stable=False):
            pairs = []
            triangulated_count = 0
            stable_count = 0
            for point in anchor_map:
                if require_triangulated and not point.get("is_triangulated"):
                    continue
                if require_stable and not (
                    point.get("is_geometry_verified")
                    or point.get("is_triangulated")
                ):
                    continue
                if point.get("is_triangulated"):
                    triangulated_count += 1
                if point.get("is_geometry_verified"):
                    stable_count += 1

                image_xy = point.get("image_xy")
                position_world = point.get("position_world")
                hits = int(point.get("hits", 0))
                if point.get("is_local_map"):
                    hits += 1
                if not image_xy or not position_world or len(image_xy) < 2 or len(position_world) < 3:
                    continue
                if hits < 2:
                    continue

                world_point = np.array(position_world, dtype=np.float32)
                camera_point = rotation_cw @ (world_point - camera_position_world)
                expected_depth = float(camera_point[2])
                if not np.isfinite(expected_depth) or expected_depth <= 0.0:
                    continue

                u = float(image_xy[0])
                v = float(image_xy[1])
                predicted_depth = _sample_depth_value(depth_map, u, v)
                if not np.isfinite(predicted_depth) or predicted_depth <= 0.0:
                    continue

                pairs.append((predicted_depth, expected_depth))
            return pairs, triangulated_count, stable_count

        anchor_candidates = [
            ("triangulated-local-map", local_sparse_map, True, False),
            ("triangulated-visible-map", sparse_map, True, False),
            ("stable-local-map", local_sparse_map, False, True),
            ("stable-visible-map", sparse_map, False, True),
        ]

        anchor_pairs = []
        for anchor_source, anchor_map, require_triangulated, require_stable in anchor_candidates:
            if not anchor_map:
                continue
            candidate_pairs, triangulated_count, stable_count = collect_anchor_pairs(
                anchor_map,
                require_triangulated=require_triangulated,
                require_stable=require_stable,
            )
            if len(candidate_pairs) < MIN_STABILIZATION_ANCHORS:
                continue
            anchor_pairs = candidate_pairs
            debug["anchor_source"] = anchor_source
            debug["triangulated_anchors"] = triangulated_count
            debug["stable_anchors"] = stable_count
            break

        debug["anchors_considered"] = len(anchor_pairs)
        if len(anchor_pairs) < MIN_STABILIZATION_ANCHORS:
            debug["reason"] = "too-few-anchors"
            debug["reset_count"] = self.reset_count
            return depth_map, debug

        if shift_norm > MAX_SHIFT_NORM_FOR_WEAK_ANCHORS and len(anchor_pairs) < (MIN_STABILIZATION_ANCHORS + 4):
            debug["reason"] = "large-motion-weak-anchor-support"
            debug["reset_count"] = self.reset_count
            return depth_map, debug

        pred = np.array([pair[0] for pair in anchor_pairs], dtype=np.float32)
        expected = np.array([pair[1] for pair in anchor_pairs], dtype=np.float32)
        finite_mask = np.isfinite(pred) & np.isfinite(expected)
        pred = pred[finite_mask]
        expected = expected[finite_mask]
        if len(pred) < MIN_STABILIZATION_ANCHORS:
            debug["reason"] = "invalid-anchor-values"
            debug["reset_count"] = self.reset_count
            return depth_map, debug

        if float(np.ptp(pred)) < 1e-5:
            scale = float(np.median(expected) / max(np.median(pred), 1e-6))
            offset = 0.0
        else:
            A = np.column_stack([pred, np.ones_like(pred)])
            scale, offset = np.linalg.lstsq(A, expected, rcond=None)[0]
            scale = float(scale)
            offset = float(offset)

        scale = float(np.clip(scale, 0.4, 2.5))
        offset = float(np.clip(offset, -1.5, 1.5))

        corrected_anchor_depths = scale * pred + offset
        abs_error = np.abs(corrected_anchor_depths - expected)
        rel_error = abs_error / np.maximum(expected, 1e-4)
        inlier_mask = rel_error < MAX_ALLOWED_RELATIVE_ERROR

        if int(inlier_mask.sum()) >= MIN_STABILIZATION_ANCHORS:
            pred_inliers = pred[inlier_mask]
            expected_inliers = expected[inlier_mask]
            if float(np.ptp(pred_inliers)) >= 1e-5:
                A = np.column_stack([pred_inliers, np.ones_like(pred_inliers)])
                scale, offset = np.linalg.lstsq(A, expected_inliers, rcond=None)[0]
                scale = float(np.clip(scale, 0.4, 2.5))
                offset = float(np.clip(offset, -1.5, 1.5))
                corrected_anchor_depths = scale * pred_inliers + offset
                abs_error = np.abs(corrected_anchor_depths - expected_inliers)
                rel_error = abs_error / np.maximum(expected_inliers, 1e-4)
                anchors_used = int(inlier_mask.sum())
            else:
                anchors_used = len(pred_inliers)
        else:
            anchors_used = len(pred)

        median_relative_error = float(np.median(rel_error)) if len(rel_error) else 1.0
        if median_relative_error > MAX_ALLOWED_RELATIVE_ERROR:
            debug["reason"] = "fit-rejected"
            debug["median_relative_error"] = round(median_relative_error, 4)
            debug["fit_scale"] = round(scale, 4)
            debug["fit_offset"] = round(offset, 4)
            debug["reset_count"] = self.reset_count
            return depth_map, debug

        corrected_map = scale * depth_map + offset
        corrected_map = np.clip(corrected_map, DEPTH_MIN_RANGE, DEPTH_MAX_RANGE).astype(np.float32)

        debug.update({
            "active": True,
            "reason": "anchors-and-pose",
            "anchors_used": anchors_used,
            "fit_scale": round(scale, 4),
            "fit_offset": round(offset, 4),
            "median_abs_error": round(float(np.median(abs_error)), 4) if len(abs_error) else None,
            "median_relative_error": round(median_relative_error, 4),
            "reset_count": self.reset_count,
        })
        return corrected_map, debug


def heuristic_depth(frame: np.ndarray) -> np.ndarray:
    """Fallback pseudo-depth heuristic used when the real model is unavailable."""
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


_DEPTH_ESTIMATOR = DepthEstimator()
_ANCHOR_DEPTH_STABILIZER = AnchorDepthStabilizer()


def estimate_depth(frame: np.ndarray) -> np.ndarray:
    """Estimate a robust per-frame relative depth map from a BGR frame."""
    return _DEPTH_ESTIMATOR.estimate(frame)


def stabilize_depth_with_anchors(depth_map: np.ndarray, camera_pose: dict | None) -> tuple[np.ndarray, dict]:
    """Use visible anchors and current camera pose to stabilize depth."""
    return _ANCHOR_DEPTH_STABILIZER.stabilize(depth_map, camera_pose)


def summarize_depth(depth_map: np.ndarray) -> dict:
    """Return compact debug statistics for the current depth map."""
    if depth_map is None or depth_map.size == 0:
        base = {
            "min_depth": 0.0,
            "max_depth": 0.0,
            "mean_depth": 0.0,
        }
    else:
        base = {
            "min_depth": round(float(depth_map.min()), 3),
            "max_depth": round(float(depth_map.max()), 3),
            "mean_depth": round(float(depth_map.mean()), 3),
        }

    return {
        **base,
        **_DEPTH_ESTIMATOR.get_debug_info(),
    }
