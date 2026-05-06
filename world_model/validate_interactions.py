"""Offline hand-object interaction validation for the demo video."""

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
from hands import HandTracker
from jepa_encoder import JepaFeatureEncoder
from lift_to_3d import infer_camera_intrinsics, lift_bbox_to_3d
from perception_candidates import build_semantic_candidates
from semantic_labels import normalize_label
from slam_backend import BuiltinSparseSlamBackend
from world_state import WorldState


MOVABLE_LABELS = {
    "cup",
    "mug",
    "apple",
    "banana",
    "orange",
    "bowl",
    "bottle",
    "cell phone",
    "book",
    "mouse",
    "donut",
    "toy",
    "toy giraffe",
    "baby bottle",
}

def _apply_video_profile_defaults(video: str) -> None:
    """Mirror demo profile knobs for offline validations."""
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
    os.environ.setdefault("DEMO_TRANSFER_TARGETS", "mat,tray")
    os.environ.setdefault("SCENE_TARGET_LABELS", "mat,tray")
    os.environ.setdefault("SCENE_MOVABLE_LABELS", "bottle,baby bottle,toy giraffe")
    os.environ.setdefault("SCENE_FORBIDDEN_LABELS", "coaster,dish,plate,cup,mug")
    os.environ.setdefault("JEPA_ENABLED", "1")
    os.environ.setdefault("JEPA_OUT_DIM", "64")
    os.environ.setdefault(
        "TEMPORAL_HEAD_MODEL_PATH",
        str(Path(__file__).resolve().parent / "models" / "temporal_interaction_head_sophie.pt"),
    )
    os.environ.setdefault("HAND_LABEL_CONTACT_ENTER_FRAMES", "bottle:1,donut:1,mouse:1,toy:1")
    os.environ.setdefault("HAND_LABEL_TOUCH_DISTANCE_M", "bottle:0.12,donut:0.12,mouse:0.11,toy:0.11")
    os.environ.setdefault("HAND_LABEL_TOUCH_START_DISTANCE_M", "bottle:0.13,donut:0.13,mouse:0.12,toy:0.12")
    os.environ.setdefault("HAND_LABEL_TOUCH_END_DISTANCE_M", "bottle:0.16,donut:0.16,mouse:0.15,toy:0.15")


def _load_annotations(path: str) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    events = data.get("events", []) if isinstance(data, dict) else data
    return [ev for ev in events if isinstance(ev, dict)]


def _new_window_stats(ev: dict, margin_s: float) -> dict:
    grab = float(ev.get("grab_start_s", 0.0) or 0.0)
    release = float(ev.get("release_s", grab) or grab)
    return {
        "id": str(ev.get("id") or ev.get("object") or "event"),
        "object": normalize_label(ev.get("object", "object")),
        "source": normalize_label(ev.get("source", "")),
        "target": normalize_label(ev.get("target", "")),
        "grab_start_s": grab,
        "release_s": release,
        "window_s": [round(max(0.0, grab - margin_s), 3), round(release + margin_s, 3)],
        "frames": 0,
        "hand_frames": 0,
        "interaction_frames": 0,
        "contact_frames": 0,
        "strict_touch_frames": 0,
        "temporal_contact_frames_at_threshold": 0,
        "temporal_place_frames_at_0_5": 0,
        "labels_seen": {},
        "contact_labels": {},
        "min_distance_by_label_m": {},
        "pick_place_count": 0,
        "pick_place_labels": {},
        "place_target_labels": {},
        "pick_place_events": [],
        "first_contact_s": None,
        "first_strict_touch_s": None,
        "first_temporal_contact_s": None,
        "first_temporal_place_s": None,
        "first_pick_place_s": None,
    }


def _window_contains(stats: dict, t_s: float) -> bool:
    start, end = stats.get("window_s", [0.0, 0.0])
    return float(start) <= float(t_s) <= float(end)


def _bump(mapping: dict, key: str) -> None:
    mapping[key] = int(mapping.get(key, 0) or 0) + 1


STATIC_TARGET_LABELS = {
    normalize_label(item)
    for item in os.environ.get(
        "STATIC_TARGET_LABELS",
        "coaster,dish,plate,platter,cake stand,tray,mat,black mat,table mat,placemat,unknown_seg",
    ).split(",")
    if item.strip()
}


def _resolve_model(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    root = Path(__file__).resolve().parent.parent / p.name
    return root if root.is_file() else p


def _bbox_from_hand_landmarks(landmarks_px, image_shape):
    if not isinstance(landmarks_px, list) or not landmarks_px:
        return None
    h, w = image_shape[:2]
    xs, ys = [], []
    for pt in landmarks_px:
        if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
            continue
        x = float(pt[0])
        y = float(pt[1])
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    if len(xs) < 4:
        return None
    x1 = max(0.0, min(float(w - 1), min(xs)))
    y1 = max(0.0, min(float(h - 1), min(ys)))
    x2 = max(0.0, min(float(w - 1), max(xs)))
    y2 = max(0.0, min(float(h - 1), max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 / max(w, 1), y1 / max(h, 1), x2 / max(w, 1), y2 / max(h, 1)]


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


def _candidate_to_detection(candidate, frame, depth_map, pose, intrinsics):
    bbox = candidate.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
        return None
    raw_label = str(candidate.get("label", "")).lower()
    label = normalize_label(raw_label)
    if label not in MOVABLE_LABELS and label not in STATIC_TARGET_LABELS and not label.startswith("unknown_seg"):
        return None
    lifted = lift_bbox_to_3d(
        bbox,
        depth_map,
        pose,
        intrinsics,
        sparse_points=pose.get("local_sparse_map") or pose.get("sparse_map") or pose.get("persistent_map") or [],
    )
    return {
        "label": label,
        "raw_label": raw_label,
        "confidence": float(candidate.get("confidence", 0.0) or 0.0),
        "bbox": [float(v) for v in bbox[:4]],
        "x": float((float(bbox[0]) + float(bbox[2])) * 0.5),
        "y": float((float(bbox[1]) + float(bbox[3])) * 0.5),
        "embedding": [0.0] * 32,
        "position_3d": lifted.get("position_world_3d", [0.0, 0.0, 0.0]),
        "position_camera_3d": lifted.get("position_camera_3d", [0.0, 0.0, 0.0]),
        "velocity_3d": [0.0, 0.0, 0.0],
        "depth": float(lifted.get("depth", 0.0) or 0.0),
        "depth_confidence": float(lifted.get("depth_confidence", 0.0) or 0.0),
        "landmark_support": int(lifted.get("landmark_support", 0) or 0),
        "landmark_blend_weight": float(lifted.get("landmark_blend_weight", 0.0) or 0.0),
        "mask_polygon": candidate.get("mask_polygon"),
        "segmentation_source": candidate.get("segmentation_source", "bbox"),
        "crop_luma_mean": _crop_luma_mean(frame, bbox),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate hand-object interactions on a video.")
    parser.add_argument("--video", default="public/scene_hand.mp4")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--detector-model", default="world_model/models/yolov8n.pt")
    parser.add_argument("--segmentation-model", default="world_model/models/yolov8n-seg.pt")
    parser.add_argument("--out-json", default="world_model/data/interaction_validation_latest.json")
    parser.add_argument("--static-seconds", type=float, default=float(os.environ.get("OBJECT_SURFACE_STATIC_SECONDS", "20.0")))
    parser.add_argument("--annotations", default="")
    parser.add_argument("--annotation-margin-s", type=float, default=1.5)
    args = parser.parse_args()

    _apply_video_profile_defaults(args.video)
    if "sophie" in str(args.video).lower() and args.static_seconds == 20.0:
        args.static_seconds = 30.0
    global STATIC_TARGET_LABELS
    STATIC_TARGET_LABELS = {
        normalize_label(item)
        for item in os.environ.get(
            "STATIC_TARGET_LABELS",
            "coaster,dish,plate,platter,cake stand,tray,mat,black mat,table mat,placemat,unknown_seg",
        ).split(",")
        if item.strip()
    }

    os.environ.setdefault("HAND_INTERACTION_SIDES", "right")
    os.environ.setdefault("HAND_FORCE_SIDE", "right")
    os.environ.setdefault("HAND_METRIC_PRIOR_PALM_WIDTH_M", "0.085")
    os.environ.setdefault("HAND_FINGER_RADIUS_M", "0.009")
    os.environ.setdefault("HAND_THUMB_RADIUS_M", "0.010")
    os.environ.setdefault("HAND_PALM_CAPSULE_RADIUS_M", "0.018")
    os.environ.setdefault("HAND_REQUIRE_3D_CONTACT_START", "1")
    os.environ.setdefault("HAND_CONTACT_ENTER_FRAMES", "2")
    os.environ.setdefault("HAND_CONTACT_EXIT_FRAMES", "3")
    os.environ.setdefault("HAND_CONTACT_2D_OVERLAP_MIN", "0.025")
    os.environ.setdefault("HAND_CONTACT_2D_EFFECTIVE_DISTANCE_M", "0.075")
    os.environ.setdefault("HAND_LABEL_CONTACT_ENTER_FRAMES", "apple:1,banana:1,orange:1")
    os.environ.setdefault("HAND_TOUCH_DISTANCE_M", "0.055")
    os.environ.setdefault("HAND_TOUCH_START_DISTANCE_M", "0.065")
    os.environ.setdefault("HAND_TOUCH_END_DISTANCE_M", "0.095")
    os.environ.setdefault("HAND_LABEL_TOUCH_DISTANCE_M", "apple:0.08,banana:0.08,orange:0.08")
    os.environ.setdefault("HAND_LABEL_TOUCH_START_DISTANCE_M", "apple:0.09,banana:0.09,orange:0.09")
    os.environ.setdefault("HAND_LABEL_TOUCH_END_DISTANCE_M", "apple:0.12,banana:0.12,orange:0.12")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(str(_resolve_model(args.detector_model))).to(device)
    seg_path = _resolve_model(args.segmentation_model)
    segmenter = YOLO(str(seg_path)).to(device) if seg_path.is_file() else None
    cap = cv2.VideoCapture(str(Path(args.video)))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    backend = BuiltinSparseSlamBackend()
    hand_tracker = HandTracker()
    world = WorldState(collection_mode=False)
    jepa = JepaFeatureEncoder()

    decoded = 0
    processed = 0
    hand_frames = 0
    contact_frames = 0
    strict_touch_frames = 0
    interaction_frames = 0
    min_dist_by_label = {}
    contact_distances = []
    labels_seen = {}
    contact_labels = {}
    memory_visible_counts = []
    memory_hidden_counts = []
    memory_label_hits = {}
    temporal_contact_frames = 0
    temporal_place_frames = 0
    temporal_contact_tp = 0
    temporal_contact_fp = 0
    temporal_contact_tn = 0
    temporal_contact_fn = 0
    temporal_samples = 0
    temporal_contact_probs = []
    temporal_place_probs = []
    temporal_contact_probs_by_label = {}
    temporal_place_probs_by_label = {}
    releases = []
    grabbed_events = []
    previous_grabbed_key = ""
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    temporal_contact_threshold = float(os.environ.get("TEMPORAL_HEAD_CONTACT_THRESHOLD", "0.20"))
    annotations = _load_annotations(args.annotations)
    timeline_windows = [_new_window_stats(ev, float(args.annotation_margin_s)) for ev in annotations]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            if args.frame_stride > 1 and (decoded - 1) % int(args.frame_stride) != 0:
                continue
            if args.max_frames > 0 and processed >= int(args.max_frames):
                break
            processed += 1
            t_s = (decoded - 1) / max(fps, 1e-6)
            active_windows = [stats for stats in timeline_windows if _window_contains(stats, t_s)]
            for stats in active_windows:
                stats["frames"] += 1

            intr = infer_camera_intrinsics(frame.shape[1], frame.shape[0])
            raw_depth = estimate_depth(frame)
            pose = backend.update(frame, depth_map=raw_depth, intrinsics=intr)
            depth_map, _ = stabilize_depth_with_anchors(raw_depth, pose)
            pose = backend.refine_visible_landmarks(depth_map, intr, pose)
            candidates = build_semantic_candidates(
                frame,
                detector,
                segmentation_model=segmenter,
                device=device,
                detector_conf_min=0.10,
                segmentation_conf_min=0.08,
                segmentation_source="yolo-seg",
                add_unmatched=True,
                unmatched_min_conf=0.12,
                unmatched_min_area=0.003,
                unmatched_max_area=0.30,
                unmatched_max_items=8,
            )
            detections = []
            for c in candidates:
                item = _candidate_to_detection(c, frame, depth_map, pose, intr)
                if item is None:
                    continue
                if item["label"].startswith("unknown_seg") and t_s > float(args.static_seconds):
                    continue
                if item["label"] in MOVABLE_LABELS:
                    labels_seen[item["label"]] = labels_seen.get(item["label"], 0) + 1
                    for stats in active_windows:
                        _bump(stats["labels_seen"], item["label"])
                if jepa.ready:
                    item["jepa_embedding"] = jepa.encode_bbox(frame, item["bbox"])
                    if item["jepa_embedding"]:
                        item["embedding"] = item["jepa_embedding"][:32]
                        item["embedding_source"] = jepa.backend or "jepa"
                detections.append(item)

            hands, _ = hand_tracker.detect(frame, depth_map, intr, pose)
            if hands:
                hand_frames += 1
                for stats in active_windows:
                    stats["hand_frames"] += 1
            if jepa.ready:
                for hand in hands:
                    hb = _bbox_from_hand_landmarks(hand.get("landmarks_px"), frame.shape[:2])
                    hand["jepa_embedding"] = jepa.encode_bbox(frame, hb) if hb is not None else []

            world.update(
                detections,
                camera_pose=pose,
                hands=hands,
                world_debug={"object_surface_static_phase": bool(t_s <= float(args.static_seconds))},
                sparse_map=pose.get("sparse_map", []),
            )
            memory = world.export_object_memory()
            memory_visible_counts.append(sum(1 for obj in memory if bool(obj.get("visible", False))))
            memory_hidden_counts.append(sum(1 for obj in memory if not bool(obj.get("visible", False))))
            for obj in memory:
                label = str(obj.get("label", "object"))
                memory_label_hits[label] = max(memory_label_hits.get(label, 0), int(obj.get("observation_count", 0) or 0))

            interactions = world.hand_object_interactions or []
            if interactions:
                interaction_frames += 1
                for stats in active_windows:
                    stats["interaction_frames"] += 1
            frame_contact = False
            frame_strict = False
            frame_temporal_contact = False
            frame_temporal_place = False
            active_grabbed_key = ""
            for it in interactions:
                label = str(it.get("nearest_object_label", "unknown"))
                dist = float(it.get("distance_m", 9.9) or 9.9)
                min_dist_by_label[label] = min(dist, min_dist_by_label.get(label, dist))
                for stats in active_windows:
                    stats["min_distance_by_label_m"][label] = min(
                        dist,
                        float(stats["min_distance_by_label_m"].get(label, dist)),
                    )
                contact_prob = float(it.get("pred_contact_prob", 0.0) or 0.0)
                place_prob = float(it.get("pred_placement_prob", 0.0) or 0.0)
                temporal_samples += 1
                temporal_contact_probs.append(contact_prob)
                temporal_place_probs.append(place_prob)
                temporal_contact_probs_by_label.setdefault(label, []).append(contact_prob)
                temporal_place_probs_by_label.setdefault(label, []).append(place_prob)
                if contact_prob >= temporal_contact_threshold:
                    frame_temporal_contact = True
                    for stats in active_windows:
                        if stats["first_temporal_contact_s"] is None:
                            stats["first_temporal_contact_s"] = round(float(t_s), 3)
                if place_prob >= 0.5:
                    frame_temporal_place = True
                    for stats in active_windows:
                        if stats["first_temporal_place_s"] is None:
                            stats["first_temporal_place_s"] = round(float(t_s), 3)
                if bool(it.get("is_touching_strict", False)):
                    frame_strict = True
                    for stats in active_windows:
                        if stats["first_strict_touch_s"] is None:
                            stats["first_strict_touch_s"] = round(float(t_s), 3)
                if bool(it.get("is_contacting", False)):
                    frame_contact = True
                    norm_label = normalize_label(label)
                    if not active_grabbed_key and norm_label in MOVABLE_LABELS:
                        active_grabbed_key = str(it.get("nearest_object_id") or norm_label)
                        if previous_grabbed_key != active_grabbed_key:
                            grabbed_events.append({
                                "video_time_s": round(float(t_s), 3),
                                "object_id": active_grabbed_key,
                                "label": norm_label,
                                "distance_m": round(float(dist), 4) if np.isfinite(dist) else None,
                                "source": "contact",
                            })
                    for stats in active_windows:
                        _bump(stats["contact_labels"], label)
                        if stats["first_contact_s"] is None:
                            stats["first_contact_s"] = round(float(t_s), 3)
                    if np.isfinite(dist) and 0.0 <= dist < 1.0:
                        contact_distances.append(dist)
                    contact_labels[label] = contact_labels.get(label, 0) + 1
            if not active_grabbed_key:
                for it in interactions:
                    if not bool(it.get("is_touching_strict", False)):
                        continue
                    norm_label = normalize_label(it.get("nearest_object_label", "unknown"))
                    if norm_label not in MOVABLE_LABELS:
                        continue
                    active_grabbed_key = str(it.get("nearest_object_id") or norm_label)
                    if previous_grabbed_key != active_grabbed_key:
                        try:
                            strict_dist = float(it.get("distance_m", 9.9) or 9.9)
                        except (TypeError, ValueError):
                            strict_dist = float("nan")
                        grabbed_events.append({
                            "video_time_s": round(float(t_s), 3),
                            "object_id": active_grabbed_key,
                            "label": norm_label,
                            "distance_m": round(float(strict_dist), 4) if np.isfinite(strict_dist) else None,
                            "source": "strict_touch",
                        })
                    break
            previous_grabbed_key = active_grabbed_key
            if frame_contact:
                contact_frames += 1
                for stats in active_windows:
                    stats["contact_frames"] += 1
            if frame_strict:
                strict_touch_frames += 1
                for stats in active_windows:
                    stats["strict_touch_frames"] += 1
            if frame_temporal_contact:
                temporal_contact_frames += 1
                for stats in active_windows:
                    stats["temporal_contact_frames_at_threshold"] += 1
            if frame_temporal_place:
                temporal_place_frames += 1
                for stats in active_windows:
                    stats["temporal_place_frames_at_0_5"] += 1
            if frame_contact and frame_temporal_contact:
                temporal_contact_tp += 1
            elif (not frame_contact) and frame_temporal_contact:
                temporal_contact_fp += 1
            elif frame_contact and (not frame_temporal_contact):
                temporal_contact_fn += 1
            else:
                temporal_contact_tn += 1

            while len(releases) < len(world.manipulation_events):
                ev = dict(world.manipulation_events[len(releases)])
                ev["video_time_s"] = round(float(t_s), 3)
                releases.append(ev)
                if ev.get("event") == "pick_place":
                    for stats in active_windows:
                        stats["pick_place_count"] += 1
                        label = normalize_label(ev.get("label", "object"))
                        _bump(stats["pick_place_labels"], label)
                        relation = ev.get("place_relation") or {}
                        target_label = normalize_label(
                            relation.get("target_label")
                            or relation.get("nearest_object_label")
                            or ""
                        )
                        if target_label:
                            _bump(stats["place_target_labels"], target_label)
                        if stats["first_pick_place_s"] is None:
                            stats["first_pick_place_s"] = round(float(t_s), 3)
                        if len(stats["pick_place_events"]) < 4:
                            stats["pick_place_events"].append({
                                "video_time_s": round(float(t_s), 3),
                                "label": label,
                                "moved": bool(ev.get("moved", False)),
                                "move_distance_m": round(float(ev.get("move_distance_m", 0.0) or 0.0), 4),
                                "target_label": target_label,
                                "target_distance_m": round(float(relation.get("nearest_distance_m", 0.0) or 0.0), 4) if relation else None,
                            })
    finally:
        cap.release()
        close_hand_tracker = getattr(hand_tracker, "close", None)
        if callable(close_hand_tracker):
            close_hand_tracker()
        backend.close()

    pick_place_events = [ev for ev in releases if ev.get("event") == "pick_place"]
    for stats in timeline_windows:
        frames = max(1, int(stats["frames"]))
        stats["hand_frame_ratio"] = round(float(stats["hand_frames"]) / frames, 4)
        stats["contact_frame_ratio"] = round(float(stats["contact_frames"]) / frames, 4)
        stats["strict_touch_frame_ratio"] = round(float(stats["strict_touch_frames"]) / frames, 4)
        stats["temporal_contact_frame_ratio_at_threshold"] = round(float(stats["temporal_contact_frames_at_threshold"]) / frames, 4)
        stats["temporal_place_frame_ratio_at_0_5"] = round(float(stats["temporal_place_frames_at_0_5"]) / frames, 4)
        stats["min_distance_by_label_m"] = {
            k: round(float(v), 4)
            for k, v in sorted(stats["min_distance_by_label_m"].items())
        }
        expected_object = stats.get("object", "")
        if expected_object == "baby bottle":
            expected_aliases = {"baby bottle", "bottle", "cup"}
        elif expected_object == "toy giraffe":
            expected_aliases = {"toy giraffe", "giraffe", "toy", "donut", "mouse"}
        else:
            expected_aliases = {expected_object}
        expected_target = stats.get("target", "")
        target_aliases = {
            expected_target,
            "mat" if expected_target in {"black mat", "table mat", "placemat", "dish", "plate"} else expected_target,
            "tray" if expected_target in {"plastic tray", "white tray"} else expected_target,
        }
        contact_hit = any(label in expected_aliases for label in stats.get("contact_labels", {}))
        pick_place_hit = any(label in expected_aliases for label in stats.get("pick_place_labels", {}))
        target_hit = (
            not expected_target
            or any(label in target_aliases for label in stats.get("place_target_labels", {}))
        )
        stats["geometry_grab_detected"] = bool(contact_hit or int(stats["contact_frames"]) > 0)
        stats["geometry_place_detected"] = bool(pick_place_hit and target_hit)
        stats["geometry_transfer_detected"] = bool(stats["geometry_grab_detected"] and stats["geometry_place_detected"])
    result = {
        "video": str(args.video),
        "processed_frames": int(processed),
        "decoded_frames": int(decoded),
        "hand_frames": int(hand_frames),
        "hand_frame_ratio": round(hand_frames / max(processed, 1), 4),
        "interaction_frames": int(interaction_frames),
        "contact_frames": int(contact_frames),
        "strict_touch_frames": int(strict_touch_frames),
        "contact_frame_ratio": round(contact_frames / max(processed, 1), 4),
        "labels_seen": labels_seen,
        "contact_labels": contact_labels,
        "object_memory": {
            "max_visible": int(max(memory_visible_counts) if memory_visible_counts else 0),
            "max_hidden": int(max(memory_hidden_counts) if memory_hidden_counts else 0),
            "final_count": int(len(world.export_object_memory())),
            "final_hidden": int(sum(1 for obj in world.export_object_memory() if not bool(obj.get("visible", False)))),
            "best_observation_count_by_label": memory_label_hits,
            "final": world.export_object_memory()[:12],
        },
        "min_distance_by_label_m": {k: round(float(v), 4) for k, v in sorted(min_dist_by_label.items())},
        "contact_distance_mean_m": round(float(np.mean(contact_distances)) if contact_distances else 0.0, 4),
        "contact_distance_p90_m": round(float(np.percentile(contact_distances, 90)) if contact_distances else 0.0, 4),
        "temporal_head": {
            "model_path": os.environ.get("TEMPORAL_HEAD_MODEL_PATH", ""),
            "contact_threshold": round(float(temporal_contact_threshold), 4),
            "samples": int(temporal_samples),
            "contact_frames_at_threshold": int(temporal_contact_frames),
            "placement_frames_at_0_5": int(temporal_place_frames),
            "contact_frame_precision_at_0_5": round(
                temporal_contact_tp / max(1, temporal_contact_tp + temporal_contact_fp),
                4,
            ),
            "contact_frame_recall_at_0_5": round(
                temporal_contact_tp / max(1, temporal_contact_tp + temporal_contact_fn),
                4,
            ),
            "contact_frame_f1_at_0_5": round(
                (2 * temporal_contact_tp) / max(1, 2 * temporal_contact_tp + temporal_contact_fp + temporal_contact_fn),
                4,
            ),
            "contact_frame_confusion_at_0_5": {
                "tp": int(temporal_contact_tp),
                "fp": int(temporal_contact_fp),
                "tn": int(temporal_contact_tn),
                "fn": int(temporal_contact_fn),
            },
            "contact_prob_mean": round(float(np.mean(temporal_contact_probs)) if temporal_contact_probs else 0.0, 4),
            "contact_prob_p90": round(float(np.percentile(temporal_contact_probs, 90)) if temporal_contact_probs else 0.0, 4),
            "placement_prob_mean": round(float(np.mean(temporal_place_probs)) if temporal_place_probs else 0.0, 4),
            "placement_prob_p90": round(float(np.percentile(temporal_place_probs, 90)) if temporal_place_probs else 0.0, 4),
            "contact_prob_mean_by_label": {
                k: round(float(np.mean(v)), 4)
                for k, v in sorted(temporal_contact_probs_by_label.items())
                if v
            },
            "placement_prob_mean_by_label": {
                k: round(float(np.mean(v)), 4)
                for k, v in sorted(temporal_place_probs_by_label.items())
                if v
            },
        },
        "timeline_validation": timeline_windows,
        "pick_place_events": pick_place_events,
        "pick_place_count": int(len(pick_place_events)),
        "grabbed_events": grabbed_events,
        "grabbed_event_count": int(len(grabbed_events)),
        "static_targets_total": int(len(world.static_targets)),
        "static_targets_locked": int(sum(1 for t in world.static_targets.values() if bool(t.get("locked", False)))),
        "final_hand_count": int(len(world.hands or [])),
        "final_interactions": world.hand_object_interactions[-8:],
    }
    blob = json.dumps(result, indent=2)
    print(blob)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(blob, encoding="utf-8")


if __name__ == "__main__":
    main()
