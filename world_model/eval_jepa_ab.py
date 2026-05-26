"""A/B evaluation for JEPA-enhanced interaction pipeline.

Runs two passes on the same video:
- baseline: JEPA disabled
- jepa: JEPA enabled (VJEPA2 if configured, otherwise fallback backend)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from depth import estimate_depth, stabilize_depth_with_anchors
from hands import HandTracker
from jepa_encoder import JepaFeatureEncoder
from lift_to_3d import infer_camera_intrinsics, lift_bbox_to_3d
from slam_backend import BuiltinSparseSlamBackend
from world_state import WorldState


def _bbox_from_hand_landmarks(landmarks_px, image_shape):
    if not isinstance(landmarks_px, list) or not landmarks_px:
        return None
    h, w = image_shape[:2]
    xs, ys = [], []
    for pt in landmarks_px:
        if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
            continue
        x = float(pt[0]); y = float(pt[1])
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x); ys.append(y)
    if len(xs) < 4:
        return None
    x1 = max(0.0, min(float(w - 1), min(xs)))
    y1 = max(0.0, min(float(h - 1), min(ys)))
    x2 = max(0.0, min(float(w - 1), max(xs)))
    y2 = max(0.0, min(float(h - 1), max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 / max(w, 1), y1 / max(h, 1), x2 / max(w, 1), y2 / max(h, 1)]


def _run_once(video_path: str, *, use_jepa: bool):
    os.environ["JEPA_ENABLED"] = "1" if use_jepa else "0"
    os.environ["JEPA_USE_FOR_CONTACT"] = "1" if use_jepa else "0"
    os.environ.setdefault("HAND_MIN_DET_CONF", "0.35")
    os.environ.setdefault("HAND_MIN_TRACK_CONF", "0.35")
    os.environ.setdefault("SLAM_KEYFRAME_MIN_INTERVAL", "7")
    os.environ.setdefault("SLAM_KEYFRAME_MIN_TRANSLATION", "0.016")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    model = YOLO(str(Path(__file__).parent / "models" / "yolov8n.pt"))
    backend = BuiltinSparseSlamBackend()
    hand_tracker = HandTracker()
    world_state = WorldState(collection_mode=False)
    jepa = JepaFeatureEncoder()
    allowed = {"cup", "mug", "apple", "orange", "banana", "bowl", "vase", "bottle", "book", "cell phone", "toy", "box"}

    processed = 0
    hand_frames = 0
    interactions_frames = 0
    contact_frames = 0
    jepa_score_samples = []
    pnp_inliers = []
    geom_verified = []
    persistent = []

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % 2 != 0:
                continue
            processed += 1

            intr = infer_camera_intrinsics(frame.shape[1], frame.shape[0])
            raw_depth = estimate_depth(frame)
            pose = backend.update(frame, depth_map=raw_depth, intrinsics=intr)
            depth_map, _ = stabilize_depth_with_anchors(raw_depth, pose)
            pose = backend.refine_visible_landmarks(depth_map, intr, pose)

            detections = []
            h, w = frame.shape[:2]
            results = model(frame, verbose=False)
            for r in results:
                for b in r.boxes:
                    conf = float(b.conf[0].item())
                    if conf < 0.20:
                        continue
                    label = str(model.names[int(b.cls[0].item())]).lower()
                    if label not in allowed:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    bbox = [
                        max(0.0, min(1.0, x1 / max(w, 1))),
                        max(0.0, min(1.0, y1 / max(h, 1))),
                        max(0.0, min(1.0, x2 / max(w, 1))),
                        max(0.0, min(1.0, y2 / max(h, 1))),
                    ]
                    lifted = lift_bbox_to_3d(bbox, depth_map, pose, intr, sparse_points=pose.get("sparse_map", []))
                    item = {
                        "label": label,
                        "confidence": conf,
                        "bbox": bbox,
                        "x": float((bbox[0] + bbox[2]) * 0.5),
                        "y": float((bbox[1] + bbox[3]) * 0.5),
                        "position_3d": lifted.get("position_world_3d", [0.0, 0.0, 0.0]),
                        "position_camera_3d": lifted.get("position_camera_3d", [0.0, 0.0, 0.0]),
                        "depth": float(lifted.get("depth", 0.0) or 0.0),
                        "depth_confidence": float(lifted.get("depth_confidence", 0.0) or 0.0),
                        "landmark_support": int(lifted.get("landmark_support", 0) or 0),
                        "landmark_blend_weight": float(lifted.get("landmark_blend_weight", 0.0) or 0.0),
                        "proxy_radius_m": float(lifted.get("proxy_radius_m", 0.06) or 0.06),
                        "proxy_extent_m": lifted.get("proxy_extent_m", [0.08, 0.08, 0.08]),
                        "surface_points_3d": lifted.get("surface_points_3d", []),
                        "embedding": [0.0] * 32,
                    }
                    if use_jepa and jepa.ready:
                        item["jepa_embedding"] = jepa.encode_bbox(frame, bbox)
                    detections.append(item)

            hands, _ = hand_tracker.detect(frame, depth_map, intr, pose)
            if use_jepa and jepa.ready:
                for hand in hands:
                    hbbox = _bbox_from_hand_landmarks(hand.get("landmarks_px"), frame.shape[:2])
                    hand["jepa_embedding"] = jepa.encode_bbox(frame, hbbox) if hbbox is not None else []
            if hands:
                hand_frames += 1

            world_state.update(
                detections,
                camera_pose=pose,
                hands=hands,
                world_debug={"intrinsics": intr},
                sparse_map=pose.get("sparse_map", []),
            )
            interactions = world_state.hand_object_interactions or []
            if interactions:
                interactions_frames += 1
                for it in interactions:
                    if "jepa_interaction_score" in it:
                        jepa_score_samples.append(float(it.get("jepa_interaction_score", 0.0) or 0.0))
            if any(bool(it.get("is_contacting", False)) for it in interactions):
                contact_frames += 1

            pnp_inliers.append(int(pose.get("pnp_inliers", 0) or 0))
            geom_verified.append(int(pose.get("geometry_verified_landmark_count", 0) or 0))
            persistent.append(int(pose.get("persistent_landmark_count", 0) or 0))

    finally:
        cap.release()
        backend.close()

    return {
        "mode": "jepa" if use_jepa else "baseline",
        "processed_frames": int(processed),
        "hand_frames": int(hand_frames),
        "hand_frame_ratio": round(float(hand_frames / max(processed, 1)), 4),
        "interaction_frames": int(interactions_frames),
        "contact_frames": int(contact_frames),
        "contact_ratio": round(float(contact_frames / max(processed, 1)), 4),
        "mean_pnp_inliers": round(float(np.mean(pnp_inliers)) if pnp_inliers else 0.0, 3),
        "mean_geom_verified": round(float(np.mean(geom_verified)) if geom_verified else 0.0, 3),
        "mean_persistent": round(float(np.mean(persistent)) if persistent else 0.0, 3),
        "mean_jepa_score": round(float(np.mean(jepa_score_samples)) if jepa_score_samples else 0.0, 4),
        "jepa_samples": int(len(jepa_score_samples)),
        "jepa_backend": jepa.backend if use_jepa else "disabled",
    }


def main():
    parser = argparse.ArgumentParser(description="Run JEPA A/B evaluation on a video.")
    parser.add_argument("--video", default="public/scene_hand.mp4")
    parser.add_argument("--out-json", default="world_model/data/jepa_ab_eval.json")
    parser.add_argument("--vjepa2-repo", default="")
    parser.add_argument("--vjepa2-model", default="")
    args = parser.parse_args()

    if args.vjepa2_repo:
        os.environ["VJEPA2_REPO"] = args.vjepa2_repo
    if args.vjepa2_model:
        os.environ["VJEPA2_MODEL"] = args.vjepa2_model

    baseline = _run_once(args.video, use_jepa=False)
    jepa = _run_once(args.video, use_jepa=True)

    result = {
        "video": str(Path(args.video).resolve()),
        "baseline": baseline,
        "jepa": jepa,
        "delta": {
            "contact_frames": int(jepa["contact_frames"] - baseline["contact_frames"]),
            "contact_ratio": round(float(jepa["contact_ratio"] - baseline["contact_ratio"]), 4),
            "interaction_frames": int(jepa["interaction_frames"] - baseline["interaction_frames"]),
            "mean_pnp_inliers": round(float(jepa["mean_pnp_inliers"] - baseline["mean_pnp_inliers"]), 3),
            "mean_geom_verified": round(float(jepa["mean_geom_verified"] - baseline["mean_geom_verified"]), 3),
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
