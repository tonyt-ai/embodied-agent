"""Offline sparse reconstruction pipeline for scene videos.

Runs the existing SLAM backend frame-by-frame on a video, then exports a sparse
point cloud (.ply) plus a JSON report with reconstruction quality metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from depth import estimate_depth, stabilize_depth_with_anchors
from lift_to_3d import infer_camera_intrinsics
from slam_backend import BuiltinSparseSlamBackend


def _serialize_intrinsics(intrinsics: dict) -> dict:
    return {
        "fx": float(intrinsics.get("fx", 0.0)),
        "fy": float(intrinsics.get("fy", 0.0)),
        "cx": float(intrinsics.get("cx", 0.0)),
        "cy": float(intrinsics.get("cy", 0.0)),
        "width": int(intrinsics.get("width", 0)),
        "height": int(intrinsics.get("height", 0)),
        "source": str(intrinsics.get("source", "unknown")),
        "calibration_file": intrinsics.get("calibration_file"),
        "distortion_coefficients": [float(v) for v in intrinsics.get("distortion_coefficients", [])],
    }


def _is_geometry_landmark(landmark: dict) -> bool:
    return bool(
        landmark.get("is_triangulated")
        or landmark.get("is_geometry_verified")
        or landmark.get("ba_lite_refined")
        or landmark.get("sliding_ba_refined")
    )


def _landmark_world_position(landmark: dict) -> list[float] | None:
    world_point = landmark.get("position_world")
    if world_point and len(world_point) >= 3:
        return world_point
    triangulated = landmark.get("triangulated_position_world")
    if triangulated and len(triangulated) >= 3:
        return triangulated
    return None


def _is_exportable_landmark(landmark: dict, min_observations: int, max_reproj_error: float) -> bool:
    world_point = _landmark_world_position(landmark)
    if not world_point or len(world_point) < 3:
        return False
    if not _is_geometry_landmark(landmark):
        return False
    observations = landmark.get("observations", []) or []
    observation_count = int(landmark.get("observation_count", len(observations)))
    hits = int(landmark.get("hits", 0))
    if observation_count < min_observations and hits < max(2, min_observations * 2):
        return False
    reproj = landmark.get("mean_reprojection_error")
    if reproj is None:
        return bool(landmark.get("is_triangulated") or landmark.get("is_geometry_verified"))
    return float(reproj) <= max_reproj_error


def _sample_color(frame: np.ndarray | None, image_xy) -> tuple[int, int, int]:
    if frame is None or image_xy is None or len(image_xy) < 2:
        return (255, 255, 255)
    h, w = frame.shape[:2]
    u = int(np.clip(round(float(image_xy[0])), 0, w - 1))
    v = int(np.clip(round(float(image_xy[1])), 0, h - 1))
    bgr = frame[v, u]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _write_ascii_ply(path: Path, points: list[tuple[float, float, float, int, int, int]]):
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


def _quality_assessment(point_count: int, keyframes: int, mean_stable_reproj: float | None) -> dict:
    if point_count >= 220 and keyframes >= 12:
        level = "good"
    elif point_count >= 120 and keyframes >= 8:
        level = "usable"
    else:
        level = "weak"

    reconstructable = level in {"good", "usable"}
    reproj_ok = mean_stable_reproj is not None and float(mean_stable_reproj) <= 8.0
    return {
        "level": level,
        "reconstructable": bool(reconstructable and (reproj_ok or mean_stable_reproj is None)),
        "reasoning": {
            "point_count": point_count,
            "keyframes": keyframes,
            "mean_stable_reprojection_error": mean_stable_reproj,
        },
    }


def run_offline_reconstruction(
    video_path: Path,
    output_ply: Path,
    report_path: Path,
    *,
    frame_stride: int,
    max_frames: int,
    mapping_interval: int,
    min_observations: int,
    max_reproj_error: float,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    backend = BuiltinSparseSlamBackend()
    processed = 0
    decoded = 0
    start = time.perf_counter()
    last_pose = {}
    last_frame = None
    intrinsics = None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    depth_ms_total = 0.0
    slam_ms_total = 0.0
    stabilization_active_count = 0
    stabilization_total = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1

            if frame_stride > 1 and ((decoded - 1) % frame_stride != 0):
                continue
            if max_frames > 0 and processed >= max_frames:
                break

            processed += 1
            last_frame = frame.copy()
            intrinsics = infer_camera_intrinsics(frame.shape[1], frame.shape[0])

            t0 = time.perf_counter()
            raw_depth = estimate_depth(frame)
            t1 = time.perf_counter()
            pose = backend.update(frame, depth_map=raw_depth, intrinsics=intrinsics)
            depth_map, stabilization_debug = stabilize_depth_with_anchors(raw_depth, pose)
            pose = backend.refine_visible_landmarks(depth_map, intrinsics, pose)
            t2 = time.perf_counter()

            depth_ms_total += (t1 - t0) * 1000.0
            slam_ms_total += (t2 - t1) * 1000.0
            stabilization_total += 1
            if stabilization_debug.get("active"):
                stabilization_active_count += 1

            last_pose = pose
            if mapping_interval > 0 and processed % mapping_interval == 0:
                backend.tracker.consolidate_map()

        backend.tracker.consolidate_map()

        landmarks = list(backend.tracker.landmarks.values())
        exportable = [
            item for item in landmarks
            if _is_exportable_landmark(item, min_observations=min_observations, max_reproj_error=max_reproj_error)
        ]
        points = []
        for landmark in exportable:
            position = _landmark_world_position(landmark)
            if position is None:
                continue
            coords = np.asarray(position[:3], dtype=np.float32)
            if coords.shape[0] < 3 or not np.isfinite(coords).all():
                continue
            color = _sample_color(last_frame, landmark.get("image_xy"))
            points.append(
                (
                    float(coords[0]),
                    float(coords[1]),
                    float(coords[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                )
            )
        _write_ascii_ply(output_ply, points)

        triangulated_count = sum(1 for item in landmarks if item.get("is_triangulated"))
        geometry_verified_count = sum(1 for item in landmarks if item.get("is_geometry_verified"))
        stable_count = sum(1 for item in landmarks if item.get("is_stable"))
        keyframes = len(backend.tracker.keyframes)
        mean_stable_reproj = backend.tracker._mean_stable_reprojection_error()
        quality = _quality_assessment(len(points), keyframes, mean_stable_reproj)

        elapsed_s = time.perf_counter() - start
        report = {
            "video_path": str(video_path.resolve()),
            "output_ply": str(output_ply.resolve()),
            "report_path": str(report_path.resolve()),
            "runtime_seconds": round(elapsed_s, 3),
            "decoded_frames": decoded,
            "processed_frames": processed,
            "frame_stride": frame_stride,
            "video_fps": round(fps, 3),
            "video_total_frames": total_frames,
            "intrinsics": _serialize_intrinsics(intrinsics or {}),
            "feature_backend": os.environ.get("FEATURE_BACKEND", "hybrid"),
            "depth_backend": os.environ.get("DEPTH_BACKEND", "depth-anything"),
            "mean_depth_ms": round(depth_ms_total / max(processed, 1), 3),
            "mean_slam_ms": round(slam_ms_total / max(processed, 1), 3),
            "stabilization_active_ratio": round(stabilization_active_count / max(stabilization_total, 1), 3),
            "landmarks_total": len(landmarks),
            "landmarks_exported": len(points),
            "triangulated_landmarks": triangulated_count,
            "geometry_verified_landmarks": geometry_verified_count,
            "stable_landmarks": stable_count,
            "keyframes": keyframes,
            "last_pose_source": last_pose.get("pose_source", "unknown"),
            "mean_stable_reprojection_error": mean_stable_reproj,
            "quality_assessment": quality,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        cap.release()
        backend.close()


def main():
    parser = argparse.ArgumentParser(description="Offline sparse reconstruction from video.")
    parser.add_argument("--video", default="public/scene_sophie.mp4", help="Input video path.")
    parser.add_argument(
        "--out-ply",
        default="world_model/data/reconstruction_scene_offline.ply",
        help="Output sparse point cloud (.ply).",
    )
    parser.add_argument(
        "--report-json",
        default="world_model/data/reconstruction_scene_offline_report.json",
        help="Output report JSON path.",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Process one frame every N decoded frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on processed frames (0 = all).")
    parser.add_argument(
        "--mapping-interval",
        type=int,
        default=20,
        help="Run heavy local map consolidation every N processed frames.",
    )
    parser.add_argument("--min-observations", type=int, default=2, help="Min observations for exported landmarks.")
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=9.0,
        help="Max mean reprojection error (px) for exported landmarks.",
    )
    args = parser.parse_args()

    report = run_offline_reconstruction(
        Path(args.video),
        Path(args.out_ply),
        Path(args.report_json),
        frame_stride=max(1, int(args.frame_stride)),
        max_frames=max(0, int(args.max_frames)),
        mapping_interval=max(0, int(args.mapping_interval)),
        min_observations=max(1, int(args.min_observations)),
        max_reproj_error=float(args.max_reprojection_error),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
