"""Validate static placement target locking on the first static segment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from depth import estimate_depth, stabilize_depth_with_anchors
from lift_to_3d import infer_camera_intrinsics, lift_bbox_to_3d
from perception_candidates import build_semantic_candidates
from slam_backend import BuiltinSparseSlamBackend
from world_state import WorldState


def _apply_video_profile_defaults(video: str) -> None:
    """Mirror demo profile knobs for offline static-target validation."""
    if "sophie" not in str(video).lower():
        return
    os.environ.setdefault("DEMO_SCENE_PROFILE", "sophie")
    os.environ.setdefault(
        "STATIC_TARGET_LABELS",
        "tray,mat,black mat,table mat,placemat,dish,plate,unknown_seg",
    )
    os.environ.setdefault("STATIC_TARGET_INFER_LARGE_LABEL", "tray")
    os.environ.setdefault("STATIC_TARGET_INFER_SMALL_LABEL", "")
    os.environ.setdefault("STATIC_TARGET_INFER_DARK_LABEL", "mat")
    os.environ.setdefault("STATIC_TARGET_DARK_LUMA_MAX", "85")
    os.environ.setdefault("STATIC_TARGET_LOCK_UNKNOWN", "0")
    os.environ.setdefault("STATIC_TARGET_HITS_MIN", "2")
    os.environ.setdefault("OBJECT_SURFACE_STATIC_SECONDS", "30")


def _crop_luma_mean(frame, bbox_norm):
    if frame is None or not hasattr(frame, "shape") or not bbox_norm or len(bbox_norm) < 4:
        return None
    h, w = frame.shape[:2]
    try:
        x1 = int(max(0, min(w - 1, round(float(bbox_norm[0]) * w))))
        y1 = int(max(0, min(h - 1, round(float(bbox_norm[1]) * h))))
        x2 = int(max(0, min(w, round(float(bbox_norm[2]) * w))))
        y2 = int(max(0, min(h, round(float(bbox_norm[3]) * h))))
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return round(float(np.mean(gray)), 3)


def _resolve_model(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    root = Path(__file__).resolve().parent.parent / p.name
    return root if root.is_file() else p


def _candidate_to_detection(candidate, frame, depth_map, pose, intrinsics):
    bbox = candidate.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
        return None
    lifted = lift_bbox_to_3d(
        bbox,
        depth_map,
        pose,
        intrinsics,
        sparse_points=pose.get("local_sparse_map") or pose.get("sparse_map") or pose.get("persistent_map") or [],
    )
    cx = float((float(bbox[0]) + float(bbox[2])) * 0.5)
    cy = float((float(bbox[1]) + float(bbox[3])) * 0.5)
    return {
        "label": str(candidate.get("label", "unknown")).lower(),
        "confidence": float(candidate.get("confidence", 0.0) or 0.0),
        "bbox": [float(v) for v in bbox[:4]],
        "x": round(cx, 4),
        "y": round(cy, 4),
        "embedding": [0.0] * 32,
        "mask_polygon": candidate.get("mask_polygon"),
        "segmentation_source": candidate.get("segmentation_source", "bbox"),
        "crop_luma_mean": _crop_luma_mean(frame, bbox),
        "position_3d": lifted.get("position_world_3d", [0.0, 0.0, 0.0]),
        "position_camera_3d": lifted.get("position_camera_3d", [0.0, 0.0, 0.0]),
        "velocity_3d": [0.0, 0.0, 0.0],
        "depth": float(lifted.get("depth", 0.0) or 0.0),
        "depth_confidence": float(lifted.get("depth_confidence", 0.0) or 0.0),
        "landmark_support": int(lifted.get("landmark_support", 0) or 0),
        "landmark_blend_weight": float(lifted.get("landmark_blend_weight", 0.0) or 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Check static target locking on a video.")
    parser.add_argument("--video", default="public/scene_hand.mp4")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--detector-model", default="world_model/models/yolov8n.pt")
    parser.add_argument("--segmentation-model", default="world_model/models/yolov8n-seg.pt")
    parser.add_argument("--segmentation-backend", default="yolo-seg")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    _apply_video_profile_defaults(args.video)
    if "sophie" in str(args.video).lower() and args.seconds == 20.0:
        args.seconds = 30.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(str(_resolve_model(args.detector_model))).to(device)
    seg_path = _resolve_model(args.segmentation_model)
    segmenter = YOLO(str(seg_path)).to(device) if seg_path.is_file() else None

    cap = cv2.VideoCapture(str(Path(args.video)))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    max_decoded = int(max(1, round(float(args.seconds) * fps)))

    backend = BuiltinSparseSlamBackend()
    world = WorldState(collection_mode=False)
    decoded = 0
    processed = 0
    candidate_counts = []
    object_counts = []

    try:
        while decoded < max_decoded:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            if int(args.frame_stride) > 1 and (decoded - 1) % int(args.frame_stride) != 0:
                continue
            processed += 1
            intrinsics = infer_camera_intrinsics(frame.shape[1], frame.shape[0])
            raw_depth = estimate_depth(frame)
            pose = backend.update(frame, depth_map=raw_depth, intrinsics=intrinsics)
            depth_map, _ = stabilize_depth_with_anchors(raw_depth, pose)
            pose = backend.refine_visible_landmarks(depth_map, intrinsics, pose)
            candidates = build_semantic_candidates(
                frame,
                detector,
                segmentation_model=segmenter,
                device=device,
                detector_conf_min=0.10,
                segmentation_conf_min=0.08,
                segmentation_source=str(args.segmentation_backend),
                add_unmatched=True,
                unmatched_min_conf=0.12,
                unmatched_min_area=0.003,
                unmatched_max_area=0.30,
                unmatched_max_items=12,
            )
            detections = []
            for candidate in candidates:
                label = str(candidate.get("label", "")).lower()
                if label in {"dining table", "chair", "couch", "tv", "potted plant", "person"}:
                    continue
                if float(candidate.get("confidence", 0.0) or 0.0) < 0.12:
                    continue
                item = _candidate_to_detection(candidate, frame, depth_map, pose, intrinsics)
                if item is not None:
                    detections.append(item)
            candidate_counts.append(len(candidates))
            object_counts.append(len(detections))
            world.update(
                detections,
                camera_pose=pose,
                hands=[],
                world_debug={"object_surface_static_phase": True},
                sparse_map=pose.get("sparse_map", []),
            )
    finally:
        cap.release()
        backend.close()

    targets = list(world.static_targets.values())
    locked = [t for t in targets if bool(t.get("locked", False))]
    result = {
        "video": str(args.video),
        "processed_frames": int(processed),
        "decoded_frames": int(decoded),
        "candidate_count_mean": round(float(np.mean(candidate_counts)) if candidate_counts else 0.0, 3),
        "object_count_mean": round(float(np.mean(object_counts)) if object_counts else 0.0, 3),
        "static_targets_total": int(len(targets)),
        "static_targets_locked": int(len(locked)),
        "locked_labels": sorted({str(t.get("label", "unknown")) for t in locked}),
        "targets": targets[:24],
        "last_static_target_update": world.world_debug.get("static_target_update", {}),
    }
    blob = json.dumps(result, indent=2)
    print(blob)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(blob, encoding="utf-8")


if __name__ == "__main__":
    main()
