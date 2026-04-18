"""Depth estimation backend for the embodied-agent prototype.

This module calls a monocular depth model backed by Hugging Face
Transformers and Depth Anything. If the model or dependency cannot be
loaded, it falls back to a lightweight heuristic so the rest
of the demo can still run.
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

    def ensure_loaded(self):
        if self._loaded or USE_HEURISTIC_ONLY:
            return

        with self._lock:
            if self._loaded:
                return

            t0 = time.perf_counter()
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                self._processor = AutoImageProcessor.from_pretrained(self.model_id)
                self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
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

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros((1, 1), dtype=np.float32)

        if USE_HEURISTIC_ONLY:
            self.backend = "heuristic"
            self.status = "forced-heuristic"
            return heuristic_depth(frame)

        self.ensure_loaded()
        if not self._loaded or self._processor is None or self._model is None:
            return heuristic_depth(frame)

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
        predicted_depth = post[0]["predicted_depth"]
        depth_map = predicted_depth.detach().float().cpu().numpy().astype(np.float32)

        if depth_map.ndim != 2:
            depth_map = np.squeeze(depth_map).astype(np.float32)

        # Normalize to a stable relative range expected by the rest of the pipeline.
        depth_map = depth_map - float(depth_map.min())
        depth_map = depth_map / (float(depth_map.max()) + 1e-6)
        depth_map = 0.35 + depth_map * 2.15
        return depth_map.astype(np.float32)

    def get_debug_info(self) -> dict:
        return {
            "backend": self.backend,
            "status": self.status,
            "model_id": self.model_id,
            "device": self.device,
            "load_time_ms": round(self.load_time_ms, 1) if self.load_time_ms is not None else None,
            "last_error": self.last_error,
        }


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


def estimate_depth(frame: np.ndarray) -> np.ndarray:
    """Estimate a relative depth map from a BGR frame."""
    return _DEPTH_ESTIMATOR.estimate(frame)


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
