"""Replay a video through built-in SLAM and export the reconstructed point cloud."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from depth import estimate_depth, stabilize_depth_with_anchors
from lift_to_3d import infer_camera_intrinsics
from slam_backend import BuiltinSparseSlamBackend


def _landmark_position(landmark: dict):
    position = landmark.get("position_world")
    if isinstance(position, (list, tuple)) and len(position) >= 3:
        return position
    tri = landmark.get("triangulated_position_world")
    if isinstance(tri, (list, tuple)) and len(tri) >= 3:
        return tri
    return None


def _is_geometry_landmark(landmark: dict) -> bool:
    return bool(
        landmark.get("is_triangulated")
        or landmark.get("is_geometry_verified")
        or landmark.get("ba_lite_refined")
        or landmark.get("sliding_ba_refined")
    )


def _sample_color(frame: np.ndarray | None, image_xy):
    if frame is None or not image_xy or len(image_xy) < 2:
        return (255, 255, 255)
    h, w = frame.shape[:2]
    u = int(np.clip(round(float(image_xy[0])), 0, w - 1))
    v = int(np.clip(round(float(image_xy[1])), 0, h - 1))
    bgr = frame[v, u]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _write_ascii_ply(path: Path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for x, y, z, r, g, b in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def run_export(
    *,
    video_path: Path,
    out_ply: Path,
    out_report: Path,
    frame_stride: int,
    max_frames: int,
    min_hits: int,
    only_geometry: bool,
    include_missing: bool,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    backend = BuiltinSparseSlamBackend()
    decoded = 0
    processed = 0
    start = time.perf_counter()
    last_frame = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            if frame_stride > 1 and (decoded - 1) % frame_stride != 0:
                continue
            if max_frames > 0 and processed >= max_frames:
                break

            last_frame = frame.copy()
            intrinsics = infer_camera_intrinsics(frame.shape[1], frame.shape[0])
            raw_depth = estimate_depth(frame)
            pose = backend.update(frame, depth_map=raw_depth, intrinsics=intrinsics)
            depth_map, _ = stabilize_depth_with_anchors(raw_depth, pose)
            backend.refine_visible_landmarks(depth_map, intrinsics, pose)
            processed += 1

        backend.tracker.consolidate_map()
        landmarks = list(backend.tracker.landmarks.values())
        points = []
        for landmark in landmarks:
            if int(landmark.get("hits", 0)) < min_hits:
                continue
            if only_geometry and not _is_geometry_landmark(landmark):
                continue
            if not include_missing and str(landmark.get("status", "")) == "missing":
                continue

            position = _landmark_position(landmark)
            if not position:
                continue
            coords = np.asarray(position[:3], dtype=np.float32)
            if coords.shape[0] < 3 or not np.isfinite(coords).all():
                continue
            color = _sample_color(last_frame, landmark.get("image_xy"))
            points.append((float(coords[0]), float(coords[1]), float(coords[2]), *color))

        _write_ascii_ply(out_ply, points)

        report = {
            "video_path": str(video_path.resolve()),
            "output_ply": str(out_ply.resolve()),
            "decoded_frames": decoded,
            "processed_frames": processed,
            "frame_stride": frame_stride,
            "max_frames": max_frames,
            "landmarks_total": len(landmarks),
            "points_exported": len(points),
            "only_geometry": only_geometry,
            "include_missing": include_missing,
            "min_hits": min_hits,
            "runtime_seconds": round(time.perf_counter() - start, 3),
        }
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        cap.release()
        backend.close()


def main():
    parser = argparse.ArgumentParser(description="Export SLAM point cloud by replaying a video.")
    parser.add_argument("--video", default="public/scene_sophie.mp4")
    parser.add_argument("--out-ply", default="world_model/data/scene_slam_exported.ply")
    parser.add_argument("--out-report", default="world_model/data/scene_slam_exported_report.json")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--only-geometry", action="store_true")
    parser.add_argument("--include-missing", action="store_true")
    args = parser.parse_args()

    report = run_export(
        video_path=Path(args.video),
        out_ply=Path(args.out_ply),
        out_report=Path(args.out_report),
        frame_stride=max(1, int(args.frame_stride)),
        max_frames=max(0, int(args.max_frames)),
        min_hits=max(1, int(args.min_hits)),
        only_geometry=bool(args.only_geometry),
        include_missing=bool(args.include_missing),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
