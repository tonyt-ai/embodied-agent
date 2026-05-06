"""SLAM backend abstraction for the embodied-agent world model.

The built-in backend is intentionally local and inspectable: it wraps the
current sparse tracker with sliding-window bundle adjustment. External
backends can later implement the same small surface area and run in a
separate process without changing server.py.
"""

from __future__ import annotations

import os
import threading
import time

from camera_tracker import CameraTracker


class BuiltinSparseSlamBackend:
    """Built-in sparse SLAM backend with local sliding-window BA."""

    name = "builtin-sparse-ba"

    def __init__(self):
        self.tracker = CameraTracker()
        self._lock = threading.RLock()
        self._mapping_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._last_scheduled_keyframe_count = 0
        self._worker = threading.Thread(target=self._mapping_loop, name="sparse-slam-mapper", daemon=True)
        self._worker.start()

    def _mapping_loop(self):
        while not self._stop_requested.is_set():
            if not self._mapping_requested.wait(timeout=0.25):
                continue
            self._mapping_requested.clear()
            with self._lock:
                self.tracker.consolidate_map()
                self.tracker.sliding_ba_stats["mapping_thread"] = "background"
                self.tracker.sliding_ba_stats["last_background_update"] = round(time.time(), 3)

    def _schedule_mapping_if_needed(self, before_keyframes: int):
        after_keyframes = len(self.tracker.keyframes)
        if after_keyframes <= before_keyframes:
            return
        if after_keyframes == self._last_scheduled_keyframe_count:
            return
        self._last_scheduled_keyframe_count = after_keyframes
        self.tracker.sliding_ba_stats["last_status"] = "queued-background"
        self.tracker.sliding_ba_stats["mapping_thread"] = "background"
        self._mapping_requested.set()

    def reset(self):
        with self._lock:
            self._mapping_requested.clear()
            self.tracker.reset()
            self.tracker.sliding_ba_stats["mapping_thread"] = "background"
            self._last_scheduled_keyframe_count = 0

    def update(self, frame, depth_map=None, intrinsics=None, semantic_mask=None):
        with self._lock:
            pose = self.tracker.update(
                frame,
                depth_map=depth_map,
                intrinsics=intrinsics,
                semantic_mask=semantic_mask,
            )
        pose["slam_backend"] = self.name
        return pose

    def refine_visible_landmarks(self, depth_map, intrinsics, camera_pose: dict, semantic_mask=None):
        with self._lock:
            before_keyframes = len(self.tracker.keyframes)
            pose = self.tracker.refine_visible_landmarks(
                depth_map,
                intrinsics,
                camera_pose,
                semantic_mask=semantic_mask,
            )
            self._schedule_mapping_if_needed(before_keyframes)
        pose["slam_backend"] = self.name
        return pose

    def close(self):
        self._stop_requested.set()
        self._mapping_requested.set()


class ExternalSlamBackend:
    """Placeholder for a dedicated external SLAM process.

    This keeps the server integration ready for MASt3R-SLAM/ORB-SLAM-style
    experiments while failing loudly instead of silently pretending to run.
    """

    def __init__(self, name: str):
        self.name = name

    def reset(self):
        return None

    def update(self, frame, depth_map=None, intrinsics=None, semantic_mask=None):
        raise RuntimeError(f"SLAM_BACKEND={self.name!r} is not wired yet")

    def refine_visible_landmarks(self, depth_map, intrinsics, camera_pose: dict, semantic_mask=None):
        return camera_pose


def create_slam_backend():
    backend = os.environ.get("SLAM_BACKEND", "builtin").lower()
    if backend in {"builtin", "sparse", "builtin-sparse-ba"}:
        return BuiltinSparseSlamBackend()
    return ExternalSlamBackend(backend)
