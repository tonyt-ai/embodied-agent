"""Lightweight egomotion tracking for the embodied-agent prototype.

The tracker estimates frame-to-frame image translation using phase
correlation over a low-resolution grayscale representation. It exposes
an accumulated world translation in arbitrary units so the rest of the
stack can start reasoning about persistent 3D positions without needing
full SLAM on day one.
"""

from __future__ import annotations

import cv2
import numpy as np


class CameraTracker:
    """Estimate simple camera pose from frame-to-frame motion."""

    def __init__(self, resize_width: int = 320, resize_height: int = 180):
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.prev_gray = None
        self.frame_index = 0
        self.translation_world = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.prev_gray = None
        self.frame_index = 0
        self.translation_world = np.zeros(3, dtype=np.float32)

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            (self.resize_width, self.resize_height),
            interpolation=cv2.INTER_AREA,
        )
        return gray.astype(np.float32)

    def update(self, frame: np.ndarray) -> dict:
        """Return a lightweight pose estimate."""
        gray = self._prepare(frame)
        self.frame_index += 1

        if self.prev_gray is None:
            self.prev_gray = gray
            return {
                "frame_index": self.frame_index,
                "status": "initialized",
                "tracking_quality": 1.0,
                "image_shift_px": [0.0, 0.0],
                "delta_translation_world": [0.0, 0.0, 0.0],
                "translation_world": self.translation_world.round(4).tolist(),
            }

        shift, response = cv2.phaseCorrelate(self.prev_gray, gray)
        dx_px = float(shift[0])
        dy_px = float(shift[1])

        scale_x = 1.0 / max(self.resize_width, 1)
        scale_y = 1.0 / max(self.resize_height, 1)
        delta = np.array(
            [-dx_px * scale_x, -dy_px * scale_y, 0.0],
            dtype=np.float32,
        )

        self.translation_world += delta
        self.prev_gray = gray

        quality = float(max(0.0, min(1.0, response)))
        return {
            "frame_index": self.frame_index,
            "status": "tracking" if quality > 0.02 else "low_confidence",
            "tracking_quality": round(quality, 3),
            "image_shift_px": [round(dx_px, 3), round(dy_px, 3)],
            "delta_translation_world": np.round(delta, 4).tolist(),
            "translation_world": np.round(self.translation_world, 4).tolist(),
        }
