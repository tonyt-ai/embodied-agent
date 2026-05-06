"""Evaluate SLAM map quality at a target keyframe checkpoint against COLMAP.

Usage example:
python world_model/eval_slam_checkpoint.py ^
  --video public/scene_sophie.mp4 ^
  --reference world_model/data/colmap_scene/sparse_points_filtered.ply ^
  --target-keyframes 10 ^
  --out-ply world_model/data/scene_slam_kf10.ply ^
  --out-json world_model/data/scene_slam_kf10_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from compare_pointclouds import _load_ascii_ply_xyz, compare_clouds
from depth import estimate_depth, stabilize_depth_with_anchors
from hands import HandTracker
from lift_to_3d import infer_camera_intrinsics
from perception_candidates import build_semantic_candidates
from semantic_stabilizer import SemanticStabilizer, build_foreground_mask
from slam_backend import BuiltinSparseSlamBackend
from ultralytics import YOLO


def _landmark_position(landmark: dict):
    for key in ("position_world", "triangulated_position_world", "position_world_depth_prior"):
        value = landmark.get(key)
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            continue
        coords = np.asarray(value[:3], dtype=np.float32)
        if np.isfinite(coords).all():
            return coords
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


def _apply_online_like_preprocess(
    frame: np.ndarray,
    max_capture_width: int,
    jpeg_quality: int,
) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = min(1.0, float(max_capture_width) / max(float(source_w), 1.0))
    frame_w = max(1, int(round(float(source_w) * scale)))
    frame_h = max(1, int(round(float(source_h) * scale)))
    if frame_w != source_w or frame_h != source_h:
        frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
    encode_ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 30, 100))],
    )
    if not encode_ok:
        return frame
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else frame


def _detect_semantic_candidates(
    frame: np.ndarray,
    model: YOLO,
    conf_min: float,
    *,
    segmentation_model=None,
    segmentation_backend: str = "yolo-seg",
    add_unmatched_seg: bool = True,
    unmatched_min_conf: float = 0.15,
    unmatched_min_area: float = 0.004,
    unmatched_max_area: float = 0.25,
    unmatched_max_items: int = 8,
):
    return build_semantic_candidates(
        frame,
        model,
        segmentation_model=segmentation_model,
        detector_conf_min=float(conf_min),
        segmentation_conf_min=0.08,
        segmentation_source=segmentation_backend,
        add_unmatched=bool(add_unmatched_seg),
        unmatched_min_conf=float(unmatched_min_conf),
        unmatched_min_area=float(unmatched_min_area),
        unmatched_max_area=float(unmatched_max_area),
        unmatched_max_items=int(unmatched_max_items),
    )


def _build_hand_dynamic_bboxes(image_shape, hands, radius_norm: float):
    if not hands:
        return []
    h, w = image_shape[:2]
    radius_px = max(8.0, float(radius_norm) * float(min(h, w)))
    boxes = []
    for hand in hands:
        center = hand.get("pixel_center")
        if not center or len(center) != 2:
            continue
        try:
            u = float(center[0])
            v = float(center[1])
        except (TypeError, ValueError):
            continue
        x1 = max(0.0, u - radius_px)
        y1 = max(0.0, v - radius_px)
        x2 = min(float(w - 1), u + radius_px)
        y2 = min(float(h - 1), v + radius_px)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            [
                round(x1 / max(w, 1), 4),
                round(y1 / max(h, 1), 4),
                round(x2 / max(w, 1), 4),
                round(y2 / max(h, 1), 4),
            ]
        )
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLAM checkpoint against COLMAP.")
    parser.add_argument("--video", default="public/scene_sophie.mp4")
    parser.add_argument("--reference", default="world_model/data/colmap_scene/sparse_points_filtered.ply")
    parser.add_argument("--target-keyframes", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--only-geometry", action="store_true")
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--depth-mode", choices=("estimate", "ones"), default="estimate")
    parser.add_argument("--out-ply", default="world_model/data/scene_slam_kf10.ply")
    parser.add_argument("--out-json", default="world_model/data/scene_slam_kf10_eval.json")
    parser.add_argument("--simulate-online", action="store_true")
    parser.add_argument("--online-capture-ms", type=float, default=200.0)
    parser.add_argument("--online-max-width", type=int, default=640)
    parser.add_argument("--online-jpeg-quality", type=int, default=80)
    parser.add_argument("--enable-semantics", action="store_true")
    parser.add_argument("--semantic-min-confidence", type=float, default=0.18)
    parser.add_argument("--enable-segmentation", action="store_true")
    parser.add_argument("--segmentation-backend", default="yolo-seg", choices=("yolo-seg", "fastsam-s", "fastsam-x"))
    parser.add_argument("--yolo-seg-model", default="world_model/models/yolov8n-seg.pt")
    parser.add_argument("--fastsam-s-model", default="world_model/models/FastSAM-s.pt")
    parser.add_argument("--fastsam-x-model", default="world_model/models/FastSAM-x.pt")
    parser.add_argument("--add-unmatched-seg-objects", action="store_true")
    parser.add_argument("--unmatched-seg-min-confidence", type=float, default=0.15)
    parser.add_argument("--unmatched-seg-min-area", type=float, default=0.004)
    parser.add_argument("--unmatched-seg-max-area", type=float, default=0.25)
    parser.add_argument("--unmatched-seg-max", type=int, default=8)
    parser.add_argument(
        "--dynamic-labels",
        default="person,cat,dog,bird",
    )
    parser.add_argument("--enable-hand-dynamic-mask", action="store_true")
    parser.add_argument("--hand-dynamic-mask-radius-norm", type=float, default=0.06)
    args = parser.parse_args()

    video_path = Path(args.video)
    reference_path = Path(args.reference)
    out_ply = Path(args.out_ply)
    out_json = Path(args.out_json)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    backend = BuiltinSparseSlamBackend()
    semantic_stabilizer = None
    yolo = None
    segmentation_model = None
    if args.enable_semantics:
        model_path = Path(__file__).parent / "models" / "yolov8n.pt"
        yolo = YOLO(str(model_path))
        if args.enable_segmentation:
            seg_path = {
                "yolo-seg": Path(args.yolo_seg_model),
                "fastsam-s": Path(args.fastsam_s_model),
                "fastsam-x": Path(args.fastsam_x_model),
            }[args.segmentation_backend]
            if not seg_path.is_file():
                root_fallback = Path(__file__).resolve().parent.parent / seg_path.name
                if root_fallback.is_file():
                    seg_path = root_fallback
            if seg_path.is_file():
                segmentation_model = YOLO(str(seg_path))
        dynamic_labels = [item.strip().lower() for item in args.dynamic_labels.split(",") if item.strip()]
        semantic_stabilizer = SemanticStabilizer(
            min_confidence=float(args.semantic_min_confidence),
            dynamic_labels=dynamic_labels,
        )
    hand_tracker = HandTracker()
    prev_hands = []
    started = time.perf_counter()
    decoded = 0
    processed = 0
    reached = False
    checkpoint_pose = None
    last_frame = None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_interval_s = 1.0 / max(fps, 1e-6)
    next_capture_time_s = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            current_time_s = (decoded - 1) * frame_interval_s

            if args.simulate_online:
                if current_time_s + 1e-9 < next_capture_time_s:
                    continue
                next_capture_time_s += max(0.001, float(args.online_capture_ms) / 1000.0)
                frame = _apply_online_like_preprocess(
                    frame,
                    max_capture_width=int(args.online_max_width),
                    jpeg_quality=int(args.online_jpeg_quality),
                )

            if args.frame_stride > 1 and (decoded - 1) % args.frame_stride != 0:
                continue
            if args.max_frames > 0 and processed >= args.max_frames:
                break

            intrinsics = infer_camera_intrinsics(frame.shape[1], frame.shape[0])
            semantic_mask = None
            hand_dynamic_boxes = []
            if args.enable_hand_dynamic_mask:
                hand_dynamic_boxes = _build_hand_dynamic_bboxes(
                    frame.shape[:2],
                    prev_hands,
                    radius_norm=float(args.hand_dynamic_mask_radius_norm),
                )
            if args.enable_semantics and semantic_stabilizer is not None and yolo is not None:
                candidates = _detect_semantic_candidates(
                    frame,
                    yolo,
                    conf_min=float(args.semantic_min_confidence),
                    segmentation_model=segmentation_model,
                    segmentation_backend=str(args.segmentation_backend),
                    add_unmatched_seg=bool(args.add_unmatched_seg_objects),
                    unmatched_min_conf=float(args.unmatched_seg_min_confidence),
                    unmatched_min_area=float(args.unmatched_seg_min_area),
                    unmatched_max_area=float(args.unmatched_seg_max_area),
                    unmatched_max_items=int(args.unmatched_seg_max),
                )
                sem_info = semantic_stabilizer.update(candidates)
                dynamic_boxes = list(sem_info.get("dynamic_bboxes", []))
                if hand_dynamic_boxes:
                    dynamic_boxes.extend(hand_dynamic_boxes)
                semantic_mask = build_foreground_mask(frame.shape[:2], dynamic_boxes)
            elif hand_dynamic_boxes:
                semantic_mask = build_foreground_mask(frame.shape[:2], hand_dynamic_boxes)

            if args.depth_mode == "ones":
                depth = np.ones((frame.shape[0], frame.shape[1]), dtype=np.float32)
                pose = backend.update(frame, depth_map=depth, intrinsics=intrinsics, semantic_mask=semantic_mask)
                pose = backend.refine_visible_landmarks(depth, intrinsics, pose, semantic_mask=semantic_mask)
            else:
                raw_depth = estimate_depth(frame)
                pose = backend.update(frame, depth_map=raw_depth, intrinsics=intrinsics, semantic_mask=semantic_mask)
                depth, _ = stabilize_depth_with_anchors(raw_depth, pose)
                pose = backend.refine_visible_landmarks(depth, intrinsics, pose, semantic_mask=semantic_mask)
            if args.enable_hand_dynamic_mask:
                hands, _ = hand_tracker.detect(frame, depth, intrinsics, pose)
                prev_hands = hands

            processed += 1
            last_frame = frame.copy()
            keyframes = int(pose.get("keyframes", 0))
            if keyframes >= int(args.target_keyframes):
                reached = True
                checkpoint_pose = dict(pose)
                break

        backend.tracker.consolidate_map()
        landmarks = list(backend.tracker.landmarks.values())
        points = []
        for landmark in landmarks:
            if int(landmark.get("hits", 0)) < int(args.min_hits):
                continue
            if args.only_geometry and not _is_geometry_landmark(landmark):
                continue
            if not args.include_missing and str(landmark.get("status", "")) == "missing":
                continue
            pos = _landmark_position(landmark)
            if pos is None:
                continue
            color = _sample_color(last_frame, landmark.get("image_xy"))
            points.append((float(pos[0]), float(pos[1]), float(pos[2]), *color))

        _write_ascii_ply(out_ply, points)

        result = {
            "video_path": str(video_path.resolve()),
            "reference_path": str(reference_path.resolve()),
            "target_keyframes": int(args.target_keyframes),
            "checkpoint_reached": bool(reached),
            "decoded_frames": decoded,
            "processed_frames": processed,
            "frame_stride": int(args.frame_stride),
            "depth_mode": args.depth_mode,
            "simulate_online": bool(args.simulate_online),
            "online_capture_ms": float(args.online_capture_ms),
            "online_max_width": int(args.online_max_width),
            "online_jpeg_quality": int(args.online_jpeg_quality),
            "enable_semantics": bool(args.enable_semantics),
            "semantic_min_confidence": float(args.semantic_min_confidence),
            "enable_hand_dynamic_mask": bool(args.enable_hand_dynamic_mask),
            "hand_dynamic_mask_radius_norm": float(args.hand_dynamic_mask_radius_norm),
            "video_fps": fps,
            "landmarks_total": len(landmarks),
            "points_exported": len(points),
            "out_ply": str(out_ply.resolve()),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }

        if checkpoint_pose is not None:
            result["checkpoint_pose"] = {
                "keyframes": int(checkpoint_pose.get("keyframes", 0)),
                "pose_source": checkpoint_pose.get("pose_source"),
                "pnp_inliers": int(checkpoint_pose.get("pnp_inliers", 0)),
                "frames_since_pnp_lock": int(checkpoint_pose.get("frames_since_pnp_lock", 0)),
                "persistent_landmark_count": int(checkpoint_pose.get("persistent_landmark_count", 0)),
                "geometry_verified_landmark_count": int(checkpoint_pose.get("geometry_verified_landmark_count", 0)),
                "triangulated_landmark_count": int(checkpoint_pose.get("triangulated_landmark_count", 0)),
                "local_keyframe_baseline": float(checkpoint_pose.get("local_keyframe_baseline", 0.0)),
                "triangulation": checkpoint_pose.get("triangulation", {}),
            }

        if reference_path.exists() and out_ply.exists():
            ref_pts = _load_ascii_ply_xyz(reference_path)
            test_pts = _load_ascii_ply_xyz(out_ply)
            if len(ref_pts) > 0 and len(test_pts) > 0:
                result["comparison"] = compare_clouds(ref_pts, test_pts)
                result["reference_points"] = int(len(ref_pts))
                result["test_points"] = int(len(test_pts))

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        cap.release()
        backend.close()


if __name__ == "__main__":
    main()
