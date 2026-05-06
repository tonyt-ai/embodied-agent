"""Evaluate semantic-stabilized SLAM settings on a video sequence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from depth import estimate_depth, stabilize_depth_with_anchors
from dino_encoder import encode_bbox
from lift_to_3d import infer_camera_intrinsics
from perception_candidates import build_semantic_candidates
from semantic_stabilizer import SemanticStabilizer, build_foreground_mask
from slam_backend import BuiltinSparseSlamBackend


def detect_semantic_candidates(
    frame,
    model,
    *,
    segmentation_model=None,
    segmentation_backend: str = "yolo-seg",
    add_unmatched_seg: bool = False,
    unmatched_min_conf: float = 0.15,
    unmatched_min_area: float = 0.004,
    unmatched_max_area: float = 0.25,
    unmatched_max_items: int = 8,
    conf_min: float,
    dino_update_every: int,
    dino_dim: int,
    max_embed_candidates: int,
    frame_counter: int,
):
    candidates = build_semantic_candidates(
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

    if candidates and dino_update_every > 0 and frame_counter % dino_update_every == 0:
        ranked = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
        for item in ranked[:max(1, max_embed_candidates)]:
            try:
                item["embedding"] = encode_bbox(frame, item["bbox"], out_dim=dino_dim)
            except Exception:
                item["embedding"] = None
    return candidates


def run_eval(args) -> dict:
    model_path = Path(__file__).parent / "models" / "yolov8n.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(model_path)).to(device)
    segmentation_model = None
    if getattr(args, "enable_segmentation", False):
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
            segmentation_model = YOLO(str(seg_path)).to(device)
    stabilizer = SemanticStabilizer(
        min_confidence=args.semantic_min_confidence,
        match_threshold=args.match_threshold,
        min_hits=args.min_hits,
        stable_confidence=args.stable_confidence,
        max_misses=args.max_misses,
        dynamic_labels=[item.strip().lower() for item in args.dynamic_labels.split(",") if item.strip()],
    )
    backend = BuiltinSparseSlamBackend()

    cap = cv2.VideoCapture(str(Path(args.video)))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    idx = 0
    used = 0
    pnp_inliers = []
    geometric_inliers = []
    dynamic_landmarks = []
    visible_landmarks = []
    stable_tracks = []
    dynamic_tracks = []
    pose_sources = {}
    tracking_lost = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if args.frame_stride > 1 and (idx - 1) % args.frame_stride != 0:
                continue
            used += 1
            if args.max_frames > 0 and used > args.max_frames:
                break

            semantic_mask = None
            sem_info = {"num_stable_tracks": 0, "num_dynamic_tracks": 0, "dynamic_bboxes": []}
            if args.enable_semantics:
                candidates = detect_semantic_candidates(
                    frame,
                    yolo,
                    segmentation_model=segmentation_model,
                    segmentation_backend=args.segmentation_backend,
                    add_unmatched_seg=args.add_unmatched_seg_objects,
                    unmatched_min_conf=args.unmatched_seg_min_confidence,
                    unmatched_min_area=args.unmatched_seg_min_area,
                    unmatched_max_area=args.unmatched_seg_max_area,
                    unmatched_max_items=args.unmatched_seg_max,
                    conf_min=args.semantic_min_confidence,
                    dino_update_every=args.dino_update_every,
                    dino_dim=args.dino_dim,
                    max_embed_candidates=args.max_embed_candidates,
                    frame_counter=used,
                )
                sem_info = stabilizer.update(candidates)
                semantic_mask = build_foreground_mask(frame.shape[:2], sem_info.get("dynamic_bboxes", []))

            depth = estimate_depth(frame)
            intrinsics = infer_camera_intrinsics(frame.shape[1], frame.shape[0])
            pose = backend.update(frame, depth_map=depth, intrinsics=intrinsics, semantic_mask=semantic_mask)
            depth2, _ = stabilize_depth_with_anchors(depth, pose)
            pose = backend.refine_visible_landmarks(depth2, intrinsics, pose, semantic_mask=semantic_mask)

            src = str(pose.get("pose_source", "unknown"))
            pose_sources[src] = pose_sources.get(src, 0) + 1
            if str(pose.get("status", "")).lower() in {"tracking-lost", "reseeded", "low_confidence"}:
                tracking_lost += 1

            pnp_inliers.append(int(pose.get("pnp_inliers", 0)))
            geometric_inliers.append(int(pose.get("geometric_inlier_count", 0)))
            dynamic_landmarks.append(int(pose.get("dynamic_landmark_count", 0)))
            visible_landmarks.append(int(pose.get("visible_landmark_count", 0)))
            stable_tracks.append(int(sem_info.get("num_stable_tracks", 0)))
            dynamic_tracks.append(int(sem_info.get("num_dynamic_tracks", 0)))

        pnp_ratio = float(pose_sources.get("pnp", 0)) / max(used, 1)
        essential_ratio = float(pose_sources.get("essential", 0)) / max(used, 1)
        lost_ratio = float(tracking_lost) / max(used, 1)
        score = (
            statistics.mean(pnp_inliers or [0.0])
            + 40.0 * pnp_ratio
            + 0.02 * statistics.mean(visible_landmarks or [0.0])
            - 25.0 * lost_ratio
        )

        return {
            "config": {
                "enable_semantics": bool(args.enable_semantics),
                "semantic_min_confidence": args.semantic_min_confidence,
                "match_threshold": args.match_threshold,
                "min_hits": args.min_hits,
                "stable_confidence": args.stable_confidence,
                "max_misses": args.max_misses,
                "dino_update_every": args.dino_update_every,
                "dino_dim": args.dino_dim,
                "max_embed_candidates": args.max_embed_candidates,
                "frame_stride": args.frame_stride,
                "max_frames": args.max_frames,
            },
            "frames_used": used,
            "pose_sources": pose_sources,
            "metrics": {
                "mean_pnp_inliers": round(float(statistics.mean(pnp_inliers or [0.0])), 3),
                "mean_geometric_inliers": round(float(statistics.mean(geometric_inliers or [0.0])), 3),
                "mean_visible_landmarks": round(float(statistics.mean(visible_landmarks or [0.0])), 3),
                "mean_dynamic_landmarks": round(float(statistics.mean(dynamic_landmarks or [0.0])), 3),
                "mean_stable_tracks": round(float(statistics.mean(stable_tracks or [0.0])), 3),
                "mean_dynamic_tracks": round(float(statistics.mean(dynamic_tracks or [0.0])), 3),
                "pnp_ratio": round(pnp_ratio, 4),
                "essential_ratio": round(essential_ratio, 4),
                "tracking_lost_ratio": round(lost_ratio, 4),
                "score": round(float(score), 3),
            },
        }
    finally:
        cap.release()
        backend.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic SLAM settings on a video.")
    parser.add_argument("--video", default="public/scene_sophie.mp4")
    parser.add_argument("--enable-semantics", action="store_true")
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
    parser.add_argument("--semantic-min-confidence", type=float, default=0.18)
    parser.add_argument("--match-threshold", type=float, default=0.28)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--stable-confidence", type=float, default=0.25)
    parser.add_argument("--max-misses", type=int, default=7)
    parser.add_argument("--dynamic-labels", default="person,cat,dog,bird")
    parser.add_argument("--dino-update-every", type=int, default=6)
    parser.add_argument("--dino-dim", type=int, default=48)
    parser.add_argument("--max-embed-candidates", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=220)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    result = run_eval(args)
    blob = json.dumps(result, indent=2)
    print(blob)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(blob, encoding="utf-8")


if __name__ == "__main__":
    main()
