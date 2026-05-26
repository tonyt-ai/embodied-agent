"""Websocket server providing world-model updates and simple planning.

This module listens for incoming frames over a websocket, detects the
most likely `cup` object using a YOLO model, optionally extracts an
DINO embedding for the crop, updates the in-memory `WorldState`, and
optionally records transitions to disk for later training. It also
exposes a simple query API (e.g. `simulate_actions`) that uses a
learned dynamics model to simulate short rollouts.
"""

import argparse
import asyncio
import base64
import cv2
import json
import math
import numpy as np
import os
import time
import torch
import websockets

from depth import (
    DEPTH_MAX_RANGE,
    DEPTH_MIN_RANGE,
    estimate_depth,
    stabilize_depth_with_anchors,
    summarize_depth,
)
from dino_encoder import encode_bbox
from hands import HandTracker
from jepa_encoder import JepaFeatureEncoder
from lift_to_3d import infer_camera_intrinsics, lift_bbox_to_3d
from colmap_depth_prior import ColmapDepthPrior
from depth_fusion import BackgroundDepthFusion
from interaction_guidance import build_interaction_guidance
from jepa_interaction_head import interaction_score as jepa_interaction_score
from pathlib import Path
from perception_candidates import (
    associate_segmentation,
    bbox_iou,
    build_semantic_candidates,
    collect_unmatched_segmentation_candidates,
    dedupe_candidates_nms,
    default_bbox_polygon,
    extract_detector_candidates,
    extract_segmentation_candidates,
    normalize_polygon,
)
from semantic_labels import normalize_label
from planner import simulate_all_actions
from semantic_stabilizer import SemanticStabilizer, build_foreground_mask
from slam_backend import create_slam_backend
from ultralytics import YOLO
from world_state import WorldState
from sophie_visual_tracker import SophieVisualTracker

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load object detector / segmenter from local model files
PERCEPTION_DETECTOR_MODE = os.environ.get("PERCEPTION_DETECTOR_MODE", "yolo").strip().lower()
PERCEPTION_SEGMENTATION_BACKEND = os.environ.get("PERCEPTION_SEGMENTATION_BACKEND", "yolo-seg").strip().lower()
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", os.path.join(MODEL_DIR, "yolov8n.pt"))
YOLO_WORLD_MODEL_PATH = os.environ.get("YOLO_WORLD_MODEL_PATH", os.path.join(MODEL_DIR, "yolov8s-worldv2.pt"))
YOLO_SEG_MODEL_PATH = os.environ.get("YOLO_SEG_MODEL_PATH", os.path.join(MODEL_DIR, "yolov8n-seg.pt"))
FASTSAM_S_MODEL_PATH = os.environ.get("FASTSAM_S_MODEL_PATH", os.path.join(MODEL_DIR, "FastSAM-s.pt"))
FASTSAM_X_MODEL_PATH = os.environ.get("FASTSAM_X_MODEL_PATH", os.path.join(MODEL_DIR, "FastSAM-x.pt"))
if not os.path.isfile(YOLO_SEG_MODEL_PATH):
    root_seg = os.path.join(os.path.dirname(BASE_DIR), "yolov8n-seg.pt")
    if os.path.isfile(root_seg):
        YOLO_SEG_MODEL_PATH = root_seg
if not os.path.isfile(FASTSAM_S_MODEL_PATH):
    root_fastsam_s = os.path.join(os.path.dirname(BASE_DIR), "FastSAM-s.pt")
    if os.path.isfile(root_fastsam_s):
        FASTSAM_S_MODEL_PATH = root_fastsam_s
if not os.path.isfile(FASTSAM_X_MODEL_PATH):
    root_fastsam_x = os.path.join(os.path.dirname(BASE_DIR), "FastSAM-x.pt")
    if os.path.isfile(root_fastsam_x):
        FASTSAM_X_MODEL_PATH = root_fastsam_x
YOLO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
detector_model_path = YOLO_MODEL_PATH
if PERCEPTION_DETECTOR_MODE == "yolo_world" and os.path.isfile(YOLO_WORLD_MODEL_PATH):
    detector_model_path = YOLO_WORLD_MODEL_PATH
model = YOLO(detector_model_path).to(YOLO_DEVICE)
segmentation_model = None
segmentation_model_path = None
if PERCEPTION_SEGMENTATION_BACKEND == "yolo-seg" and os.path.isfile(YOLO_SEG_MODEL_PATH):
    segmentation_model_path = YOLO_SEG_MODEL_PATH
if PERCEPTION_SEGMENTATION_BACKEND == "fastsam-s" and os.path.isfile(FASTSAM_S_MODEL_PATH):
    segmentation_model_path = FASTSAM_S_MODEL_PATH
if PERCEPTION_SEGMENTATION_BACKEND == "fastsam-x" and os.path.isfile(FASTSAM_X_MODEL_PATH):
    segmentation_model_path = FASTSAM_X_MODEL_PATH
if segmentation_model_path:
    try:
        segmentation_model = YOLO(segmentation_model_path).to(YOLO_DEVICE)
    except Exception:
        segmentation_model = None
        segmentation_model_path = None

# File to append collected transitions (JSON lines)
TRANSITIONS_FILE = os.path.join(DATA_DIR, "transitions.jsonl")
INTERACTION_CAPTURE_FILE = os.environ.get(
    "INTERACTION_CAPTURE_FILE",
    os.path.join(DATA_DIR, "interaction_capture.jsonl"),
)

# Runtime flags and DINO configuration
parser = argparse.ArgumentParser()
parser.add_argument("--capture", action="store_true", help="Enable transition capture mode")
args = parser.parse_args()

COLLECT_TRANSITIONS = args.capture
AUTO_INTERACTION_CAPTURE = os.environ.get(
    "AUTO_INTERACTION_CAPTURE",
    "1" if COLLECT_TRANSITIONS else "0",
).lower() in {"1", "true", "yes"}
INTERACTION_CAPTURE_EVERY_FRAMES = int(os.environ.get("INTERACTION_CAPTURE_EVERY_FRAMES", "10"))
USE_DINO_EMBEDDING = os.environ.get("USE_DINO_EMBEDDING", "1").lower() in {"1", "true", "yes"}
DINO_DIM = 32
DINO_UPDATE_EVERY = int(os.environ.get("DINO_UPDATE_EVERY", "8"))   # recompute sparse DINO identity memory
DINO_BOOTSTRAP_SECONDS = float(os.environ.get("DINO_BOOTSTRAP_SECONDS", "20.0"))
DINO_BOOTSTRAP_UPDATE_EVERY = int(os.environ.get("DINO_BOOTSTRAP_UPDATE_EVERY", "4"))
MULTI_OBJECT_TRACKING = os.environ.get("DEMO_MULTI_OBJECT_TRACKING", "1").lower() not in {"0", "false", "no"}
MAX_TRACKED_OBJECTS = int(os.environ.get("DEMO_MAX_TRACKED_OBJECTS", "6"))
TRACKED_OBJECT_MIN_CONFIDENCE = float(os.environ.get("DEMO_TRACKED_OBJECT_MIN_CONFIDENCE", "0.20"))
TRACKED_OBJECT_LABELS = [
    item.strip().lower()
    for item in os.environ.get("DEMO_TRACKED_OBJECT_LABELS", "").split(",")
    if item.strip()
]
if not TRACKED_OBJECT_LABELS:
    TRACKED_OBJECT_LABELS = [
        "cup",
        "mug",
        "bottle",
        "book",
        "cell phone",
        "laptop",
        "mouse",
        "knife",
        "scissors",
        "banana",
        "apple",
        "orange",
        "bowl",
        "vase",
    ]
TRACKED_OBJECT_LABEL_DENYLIST = {
    item.strip().lower()
    for item in os.environ.get(
        "DEMO_TRACKED_OBJECT_LABEL_DENYLIST",
        "dining table,chair,couch,tv,potted plant",
    ).split(",")
    if item.strip()
}
MAX_OBJECT_DINO_EMBEDS_PER_UPDATE = int(os.environ.get("DEMO_MAX_OBJECT_DINO_EMBEDS_PER_UPDATE", "3"))
PERCEPTION_ADD_UNMATCHED_SEG_OBJECTS = os.environ.get(
    "PERCEPTION_ADD_UNMATCHED_SEG_OBJECTS",
    "1",
).lower() in {"1", "true", "yes"}
PERCEPTION_UNMATCHED_SEG_MAX = int(os.environ.get("PERCEPTION_UNMATCHED_SEG_MAX", "8"))
PERCEPTION_UNMATCHED_SEG_MIN_CONF = float(os.environ.get("PERCEPTION_UNMATCHED_SEG_MIN_CONF", "0.15"))
PERCEPTION_UNMATCHED_SEG_MIN_AREA = float(os.environ.get("PERCEPTION_UNMATCHED_SEG_MIN_AREA", "0.004"))
PERCEPTION_UNMATCHED_SEG_MAX_AREA = float(os.environ.get("PERCEPTION_UNMATCHED_SEG_MAX_AREA", "0.25"))
SOPHIE_DINO_REID_ENABLED = os.environ.get("SOPHIE_DINO_REID_ENABLED", "1").lower() in {"1", "true", "yes"}
SOPHIE_DINO_REID_MAX_CANDIDATES = int(os.environ.get("SOPHIE_DINO_REID_MAX_CANDIDATES", "12"))
SOPHIE_DINO_REID_MIN_SIM = float(os.environ.get("SOPHIE_DINO_REID_MIN_SIM", "0.70"))
SOPHIE_DINO_REID_MIN_MARGIN = float(os.environ.get("SOPHIE_DINO_REID_MIN_MARGIN", "0.04"))
SEMANTIC_ENABLED = os.environ.get("SLAM_SEMANTIC_STABILIZATION", "1").lower() not in {"0", "false", "no"}
SEMANTIC_DINO_DIM = int(os.environ.get("SLAM_SEMANTIC_DINO_DIM", "48"))
SEMANTIC_DINO_UPDATE_EVERY = int(os.environ.get("SLAM_SEMANTIC_DINO_UPDATE_EVERY", "6"))
SEMANTIC_MAX_EMBED_CANDIDATES = int(os.environ.get("SLAM_SEMANTIC_MAX_EMBED_CANDIDATES", "6"))
SEMANTIC_MIN_CONFIDENCE = float(os.environ.get("SLAM_SEMANTIC_MIN_CONFIDENCE", "0.18"))
SEMANTIC_DYNAMIC_LABELS = [
    item.strip().lower()
    for item in os.environ.get(
        "SLAM_DYNAMIC_LABELS",
        "person,cat,dog,bird",
    ).split(",")
    if item.strip()
]
HAND_DYNAMIC_MASK_ENABLED = os.environ.get("SLAM_HAND_DYNAMIC_MASK_ENABLED", "1").lower() not in {"0", "false", "no"}
HAND_DYNAMIC_MASK_RADIUS_NORM = float(os.environ.get("SLAM_HAND_DYNAMIC_MASK_RADIUS_NORM", "0.06"))
HAND_WORLD_ALIGN_ENABLED = os.environ.get("HAND_WORLD_ALIGN_ENABLED", "0").lower() not in {"0", "false", "no"}
HAND_WORLD_ALIGN_IMAGE_NORM_RADIUS = float(os.environ.get("HAND_WORLD_ALIGN_IMAGE_NORM_RADIUS", "0.11"))
HAND_WORLD_ALIGN_ALPHA = float(os.environ.get("HAND_WORLD_ALIGN_ALPHA", "0.55"))
HAND_WORLD_ALIGN_MAX_SHIFT_M = float(os.environ.get("HAND_WORLD_ALIGN_MAX_SHIFT_M", "0.45"))
HAND_WORLD_ALIGN_RUNTIME_ENABLED = bool(HAND_WORLD_ALIGN_ENABLED)
VLM_REFINER_ENABLED = os.environ.get("PERCEPTION_VLM_REFINER_ENABLED", "0").lower() in {"1", "true", "yes"}
VLM_REFINER_EVERY_N_KEYFRAMES = int(os.environ.get("PERCEPTION_VLM_REFINER_EVERY_N_KEYFRAMES", "4"))
def _resolve_default_colmap_sparse_txt_dir() -> str:
    candidates = [
        os.path.join(DATA_DIR, "colmap_scene_hand_20s_apr29", "sparse_txt"),
        os.path.join(DATA_DIR, "colmap_scene_hand", "sparse_txt"),
        os.path.join(DATA_DIR, "colmap_scene", "sparse_txt"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return ""


COLMAP_PRIOR_SPARSE_TXT_DIR = os.environ.get("COLMAP_DEPTH_PRIOR_SPARSE_TXT_DIR", "").strip()
if not COLMAP_PRIOR_SPARSE_TXT_DIR:
    COLMAP_PRIOR_SPARSE_TXT_DIR = _resolve_default_colmap_sparse_txt_dir()
_colmap_prior_enabled_raw = os.environ.get("COLMAP_DEPTH_PRIOR_ENABLED", "").strip().lower()
if _colmap_prior_enabled_raw:
    COLMAP_PRIOR_ENABLED = _colmap_prior_enabled_raw in {"1", "true", "yes"}
else:
    COLMAP_PRIOR_ENABLED = bool(COLMAP_PRIOR_SPARSE_TXT_DIR)
COLMAP_PRIOR_FPS = float(os.environ.get("COLMAP_DEPTH_PRIOR_FPS", "3.0"))
COLMAP_PRIOR_RUNTIME_FPS = float(os.environ.get("COLMAP_DEPTH_PRIOR_RUNTIME_FPS", "5.0"))
COLMAP_PRIOR_EXCLUDE_DYNAMIC = os.environ.get("COLMAP_DEPTH_PRIOR_EXCLUDE_DYNAMIC", "1").lower() in {"1", "true", "yes"}
DEPTH_FUSION_BLEND_IN_PERSISTENT = os.environ.get("DEPTH_FUSION_BLEND_IN_PERSISTENT", "1").lower() in {"1", "true", "yes"}
DEPTH_FUSION_MAX_SEND_POINTS = int(os.environ.get("DEPTH_FUSION_MAX_SEND_POINTS", "900"))
OBJECT_SURFACE_STATIC_SECONDS = float(os.environ.get("OBJECT_SURFACE_STATIC_SECONDS", "20.0"))
JEPA_ENABLED = os.environ.get("JEPA_ENABLED", "0").lower() in {"1", "true", "yes"}
JEPA_MAX_OBJECTS_PER_FRAME = int(os.environ.get("JEPA_MAX_OBJECTS_PER_FRAME", "3"))
JEPA_USE_FOR_CONTACT = os.environ.get("JEPA_USE_FOR_CONTACT", "1").lower() in {"1", "true", "yes"}
DEPTH_DEBUG_LOW_PERCENTILE = float(os.environ.get("DEPTH_DEBUG_LOW_PERCENTILE", "3.0"))
DEPTH_DEBUG_HIGH_PERCENTILE = float(os.environ.get("DEPTH_DEBUG_HIGH_PERCENTILE", "97.0"))
DEPTH_DEBUG_EMA_ALPHA = float(os.environ.get("DEPTH_DEBUG_EMA_ALPHA", "0.25"))
_depth_debug_low_ema = None
_depth_debug_high_ema = None

OBJECT_DIMENSION_PRIORS_M = {
    "apple": {"height": 0.08, "diameter": 0.07},
    "cup": {"height": 0.09, "diameter": 0.085},
    "mug": {"height": 0.087, "diameter": 0.085},
    "bowl": {"height": 0.05, "diameter": 0.13},
    "orange": {"height": 0.075, "diameter": 0.075},
}
TRACKED_STATIC_TARGET_LABELS = {
    normalize_label(item)
    for item in os.environ.get(
        "STATIC_TARGET_LABELS",
        "coaster,dish,plate,platter,cake stand,tray,mat,black mat,table mat,placemat,unknown_seg",
    ).split(",")
    if item.strip()
}


def _configure_open_vocab_detector_classes():
    """Restrict YOLO-World open-vocab classes to the current tracked label list when possible."""
    if PERCEPTION_DETECTOR_MODE != "yolo_world":
        return "disabled"
    if not hasattr(model, "set_classes"):
        return "unsupported"
    classes = list(TRACKED_OBJECT_LABELS) if TRACKED_OBJECT_LABELS else [
        "cup", "mug", "bowl", "bottle", "book", "cell phone", "laptop", "mouse",
        "knife", "scissors", "banana", "apple", "orange", "toy", "box", "hand",
    ]
    try:
        model.set_classes(classes)
        return f"configured:{len(classes)}"
    except Exception:
        return "failed"


OPEN_VOCAB_STATUS = _configure_open_vocab_detector_classes()

# DINO is a stable identity memory once a detection is matched into WorldState.
# New detections start with HSV and receive sparse DINO refreshes below.
last_dino_embedding = [0.0] * DINO_DIM
dino_frame_counter = 0
semantic_dino_frame_counter = 0
sophie_visual_tracker = SophieVisualTracker() if os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() == "sophie" else None

# State used for collecting transitions across frames
prev_state_vec = None
prev_move_vec = None
last_interaction_event_count = 0
state = WorldState(collection_mode=COLLECT_TRANSITIONS)
slam_backend = create_slam_backend()
semantic_stabilizer = SemanticStabilizer(
    min_confidence=SEMANTIC_MIN_CONFIDENCE,
    dynamic_labels=SEMANTIC_DYNAMIC_LABELS,
)
hand_tracker = HandTracker()
depth_fusion = BackgroundDepthFusion()
jepa_encoder = JepaFeatureEncoder()
colmap_depth_prior = None
if COLMAP_PRIOR_ENABLED and COLMAP_PRIOR_SPARSE_TXT_DIR:
    colmap_depth_prior = ColmapDepthPrior(
        COLMAP_PRIOR_SPARSE_TXT_DIR,
        prior_fps=COLMAP_PRIOR_FPS,
        runtime_fps=COLMAP_PRIOR_RUNTIME_FPS,
    )

DEMO_REQUIRE_HANDS_FOR_GUIDANCE = os.environ.get("DEMO_REQUIRE_HANDS_FOR_GUIDANCE", "0").lower() in {"1", "true", "yes"}
DEMO_MIN_POSE_SCORE = float(os.environ.get("DEMO_MIN_POSE_SCORE", "0.40"))
DEMO_MIN_MAP_SCORE = float(os.environ.get("DEMO_MIN_MAP_SCORE", "0.40"))
DEMO_MIN_HAND_SCORE = float(os.environ.get("DEMO_MIN_HAND_SCORE", "0.35"))
DEMO_MIN_OVERALL_SCORE = float(os.environ.get("DEMO_MIN_OVERALL_SCORE", "0.45"))
DEMO_REQUIRE_TARGET_HAND_ENGAGEMENT = os.environ.get(
    "DEMO_REQUIRE_TARGET_HAND_ENGAGEMENT",
    "0",
).lower() in {"1", "true", "yes"}
DEMO_MIN_INTERACTION_SCORE = float(os.environ.get("DEMO_MIN_INTERACTION_SCORE", "0.30"))


def reset_runtime_state():
    """Reset transient world-model state between repeatable debug runs."""
    global state, prev_state_vec, prev_move_vec, last_interaction_event_count, last_dino_embedding, dino_frame_counter, semantic_dino_frame_counter, sophie_visual_tracker

    state = WorldState(collection_mode=COLLECT_TRANSITIONS)
    slam_backend.reset()
    semantic_stabilizer.reset()
    hand_tracker.reset()
    depth_fusion.reset()
    prev_state_vec = None
    prev_move_vec = None
    last_interaction_event_count = 0
    last_dino_embedding = [0.0] * DINO_DIM
    dino_frame_counter = 0
    semantic_dino_frame_counter = 0
    sophie_visual_tracker = SophieVisualTracker() if os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() == "sophie" else None


def sanitize_for_json(value):
    """Recursively convert non-JSON-safe numeric values."""
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _compute_interaction_score(hand_object_interactions: list[dict] | None, target_label: str = "cup"):
    interactions = hand_object_interactions or []
    if not interactions:
        return 0.0, {
            "target_label": target_label,
            "target_interactions": 0,
            "target_contacting": 0,
            "target_near": 0,
        }

    target_entries = [
        item for item in interactions
        if str(item.get("nearest_object_label", "")).lower() == str(target_label).lower()
    ]
    if not target_entries:
        return 0.0, {
            "target_label": target_label,
            "target_interactions": 0,
            "target_contacting": 0,
            "target_near": 0,
        }

    contacting = sum(1 for item in target_entries if bool(item.get("is_contacting", False)))
    near = sum(1 for item in target_entries if bool(item.get("is_near", False)))
    confidences = [float(item.get("hand_confidence", 0.0) or 0.0) for item in target_entries]
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    contact_score = _clamp01(contacting / 1.0)
    near_score = _clamp01(near / 1.0)
    score = _clamp01(0.55 * contact_score + 0.30 * near_score + 0.15 * mean_conf)
    return score, {
        "target_label": target_label,
        "target_interactions": int(len(target_entries)),
        "target_contacting": int(contacting),
        "target_near": int(near),
    }


def compute_demo_gate(
    camera_pose: dict | None,
    hands: list[dict] | None,
    hand_object_interactions: list[dict] | None = None,
):
    camera_pose = camera_pose or {}
    hands = hands or []

    tracking_quality = float(camera_pose.get("tracking_quality", 0.0) or 0.0)
    pnp_inliers = float(camera_pose.get("pnp_inliers", 0.0) or 0.0)
    pose_source = str(camera_pose.get("pose_source", "unknown"))
    pose_source_bonus = 0.12 if pose_source == "pnp" else 0.0
    pose_score = _clamp01(0.55 * tracking_quality + 0.45 * _clamp01(pnp_inliers / 40.0) + pose_source_bonus)

    persistent = float(camera_pose.get("persistent_landmark_count", 0.0) or 0.0)
    geom_verified = float(camera_pose.get("geometry_verified_landmark_count", 0.0) or 0.0)
    local_baseline = float(camera_pose.get("local_keyframe_baseline", 0.0) or 0.0)
    map_score = _clamp01(
        0.45 * _clamp01(geom_verified / 220.0)
        + 0.35 * _clamp01(persistent / 420.0)
        + 0.20 * _clamp01(local_baseline / 0.25)
    )

    hand_count = len(hands)
    hand_conf_values = [float(item.get("confidence", 0.0) or 0.0) for item in hands]
    mean_hand_conf = float(np.mean(hand_conf_values)) if hand_conf_values else 0.0
    hand_score = _clamp01(0.5 * _clamp01(hand_count / 2.0) + 0.5 * _clamp01(mean_hand_conf))
    interaction_score, interaction_summary = _compute_interaction_score(hand_object_interactions)

    if DEMO_REQUIRE_HANDS_FOR_GUIDANCE:
        overall = _clamp01(
            0.35 * pose_score
            + 0.30 * map_score
            + 0.20 * hand_score
            + 0.15 * interaction_score
        )
    else:
        interaction_weight = 0.15 if DEMO_REQUIRE_TARGET_HAND_ENGAGEMENT else 0.0
        base_weight = max(1.0 - interaction_weight, 1e-6)
        overall = _clamp01(
            base_weight * (0.52 * pose_score + 0.48 * map_score)
            + interaction_weight * interaction_score
        )

    reasons = []
    if pose_score < DEMO_MIN_POSE_SCORE:
        reasons.append("low-pose-confidence")
    if map_score < DEMO_MIN_MAP_SCORE:
        reasons.append("low-map-confidence")
    if DEMO_REQUIRE_HANDS_FOR_GUIDANCE and hand_score < DEMO_MIN_HAND_SCORE:
        reasons.append("low-hand-confidence")
    if DEMO_REQUIRE_TARGET_HAND_ENGAGEMENT and interaction_score < DEMO_MIN_INTERACTION_SCORE:
        reasons.append("low-target-hand-engagement")
    if overall < DEMO_MIN_OVERALL_SCORE:
        reasons.append("low-overall-confidence")

    allow_guidance = len(reasons) == 0
    return {
        "allow_guidance": bool(allow_guidance),
        "reason": "ok" if allow_guidance else ",".join(reasons),
        "overall_score": round(overall, 3),
        "pose_score": round(pose_score, 3),
        "map_score": round(map_score, 3),
        "hand_score": round(hand_score, 3),
        "interaction_score": round(interaction_score, 3),
        "require_hands": bool(DEMO_REQUIRE_HANDS_FOR_GUIDANCE),
        "require_target_hand_engagement": bool(DEMO_REQUIRE_TARGET_HAND_ENGAGEMENT),
        "hand_count": int(hand_count),
        "target_interaction": interaction_summary,
    }


def save_transition(state_vec, action_vec, next_state_vec):
    """Append a single transition record to the transitions file."""
    record = {
        "state": sanitize_for_json(state_vec),
        "action": sanitize_for_json(action_vec),
        "next_state": sanitize_for_json(next_state_vec),
    }
    with open(TRANSITIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


def maybe_capture_interaction_frame(frame_counter: int, elapsed_s, world_state: WorldState, world_debug: dict):
    """Append rich self-supervision rows for geometry-labeled interactions."""
    global last_interaction_event_count

    if not AUTO_INTERACTION_CAPTURE:
        return False

    interactions = list(getattr(world_state, "hand_object_interactions", []) or [])
    events = list(getattr(world_state, "manipulation_events", []) or [])
    event_count = len(events)
    has_new_event = event_count > int(last_interaction_event_count)
    has_contact = any(bool(item.get("is_contacting", False)) or bool(item.get("is_touching_strict", False)) for item in interactions)
    interval = max(1, int(INTERACTION_CAPTURE_EVERY_FRAMES))
    should_sample = bool(interactions) and (frame_counter % interval == 0)
    if not (has_new_event or has_contact or should_sample):
        return False

    new_events = events[int(last_interaction_event_count):] if has_new_event else []
    last_interaction_event_count = event_count
    objects = world_state.export_objects()
    record = {
        "schema": "interaction_capture.v1",
        "frame": int(frame_counter),
        "elapsed_s": float(elapsed_s) if isinstance(elapsed_s, (int, float)) else None,
        "objects": [
            {
                "id": obj.get("id"),
                "label": obj.get("label"),
                "bbox": obj.get("bbox"),
                "confidence": obj.get("confidence"),
                "position_3d": obj.get("position_3d"),
                "position_camera_3d": obj.get("position_camera_3d"),
                "velocity_3d": obj.get("velocity_3d"),
                "embedding_source": obj.get("embedding_source"),
                "jepa_temporal_embedding": obj.get("jepa_temporal_embedding", []),
                "depth_confidence": obj.get("depth_confidence", 0.0),
            }
            for obj in objects
        ],
        "hands": [
            {
                "id": hand.get("id"),
                "side": hand.get("side"),
                "confidence": hand.get("confidence"),
                "center_3d": hand.get("center_3d"),
                "velocity_3d": hand.get("velocity_3d"),
                "predicted": hand.get("predicted", False),
                "missing_frames": hand.get("missing_frames", 0),
                "depth_evidence": hand.get("depth_evidence"),
                "jepa_temporal_embedding": hand.get("jepa_temporal_embedding", []),
            }
            for hand in getattr(world_state, "hand_tracks", {}).values()
        ],
        "interactions": interactions,
        "new_events": new_events,
        "recent_events": events[-6:],
        "static_targets": (
            world_state.export_world_state().get("static_targets", [])
            if hasattr(world_state, "export_world_state")
            else list(getattr(world_state, "static_targets", {}).values())
        ),
        "teacher_labels": {
            "has_contact": bool(has_contact),
            "new_event": bool(has_new_event),
            "contact_object_ids": [
                item.get("nearest_object_id")
                for item in interactions
                if bool(item.get("is_contacting", False))
            ],
            "strict_touch_object_ids": [
                item.get("nearest_object_id")
                for item in interactions
                if bool(item.get("is_touching_strict", False))
            ],
        },
        "quality": {
            "pose_source": (world_debug or {}).get("pose_source"),
            "pnp_inliers": (world_debug or {}).get("pnp_inliers", 0),
            "geometry_verified_landmark_count": (world_debug or {}).get("geometry_verified_landmark_count", 0),
            "static_targets_locked": int(sum(1 for t in getattr(world_state, "static_targets", {}).values() if bool(t.get("locked", False)))),
        },
    }
    with open(INTERACTION_CAPTURE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitize_for_json(record), allow_nan=False) + "\n")
    return True


def decode_data_url_image(data_url):
    """Decode a `data:` URL (base64) into an OpenCV BGR image.

    The client sends frames encoded as data URLs; this helper extracts
    the bytes and uses OpenCV to decode into a NumPy BGR array.
    """
    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def _encode_fast_crop_embedding(frame: np.ndarray, bbox_norm, out_dim: int = DINO_DIM):
    """Cheap appearance embedding fallback (HSV histogram) for all tracked objects."""
    h, w = frame.shape[:2]
    if not bbox_norm or len(bbox_norm) != 4:
        return [0.0] * out_dim
    try:
        x1 = int(max(0, min(w - 1, round(float(bbox_norm[0]) * w))))
        y1 = int(max(0, min(h - 1, round(float(bbox_norm[1]) * h))))
        x2 = int(max(0, min(w, round(float(bbox_norm[2]) * w))))
        y2 = int(max(0, min(h, round(float(bbox_norm[3]) * h))))
    except Exception:
        return [0.0] * out_dim
    if x2 <= x1 or y2 <= y1:
        return [0.0] * out_dim

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return [0.0] * out_dim
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    vec = np.concatenate([hist_h, hist_s, hist_v], axis=0).astype(np.float32)
    if vec.size < out_dim:
        vec = np.pad(vec, (0, out_dim - vec.size), mode="constant")
    elif vec.size > out_dim:
        vec = vec[:out_dim]
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec = vec / norm
    return vec.tolist()


def _nonzero_embedding(emb) -> bool:
    if not isinstance(emb, list) or not emb:
        return False
    try:
        return any(abs(float(v)) > 1e-8 for v in emb)
    except (TypeError, ValueError):
        return False


def _cosine_embedding(a, b) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    try:
        av = np.asarray(a[:n], dtype=np.float32)
        bv = np.asarray(b[:n], dtype=np.float32)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom <= 1e-8:
            return 0.0
        return float(np.clip(np.dot(av, bv) / denom, -1.0, 1.0))
    except Exception:
        return 0.0


def _bbox_area_norm(bbox) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0.0
    try:
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    except (TypeError, ValueError):
        return 0.0


def _sophie_identity_prototypes():
    """Return persistent DINO prototypes for Sophie objects from WorldState."""
    if os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() != "sophie":
        return {}
    prototypes = {}
    objects = getattr(state, "objects", {}) if state is not None else {}
    for obj in objects.values():
        label = normalize_label(
            obj.get("scene_memory_label")
            or obj.get("semantic_label")
            or obj.get("label")
            or obj.get("raw_label")
            or ""
        )
        visual = str(obj.get("visual_identity_class") or "").strip().lower()
        if visual == "baby_bottle":
            label = "baby bottle"
        elif visual == "toy_giraffe":
            label = "toy giraffe"
        if label not in {"baby bottle", "toy giraffe"}:
            continue
        emb = obj.get("dino_embedding") or (obj.get("embedding") if str(obj.get("embedding_source", "")).lower() == "dino" else [])
        if not _nonzero_embedding(emb):
            continue
        prototypes.setdefault(label, []).append(emb)
    out = {}
    for label, embeds in prototypes.items():
        arrs = []
        for emb in embeds[-8:]:
            try:
                arr = np.asarray(emb, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 1e-8:
                    arrs.append(arr / norm)
            except Exception:
                continue
        if arrs:
            proto = np.mean(np.stack(arrs, axis=0), axis=0)
            norm = float(np.linalg.norm(proto))
            if norm > 1e-8:
                out[label] = (proto / norm).astype(np.float32).tolist()
    return out


def _crop_luma_mean(frame: np.ndarray, bbox_norm) -> float | None:
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


def _should_track_object_label(label: str) -> bool:
    norm = normalize_label(label)
    if norm.startswith("unknown_seg"):
        return True
    if norm in TRACKED_STATIC_TARGET_LABELS:
        return True
    if norm == "cup":
        return True
    if norm in TRACKED_OBJECT_LABEL_DENYLIST:
        return False
    if not MULTI_OBJECT_TRACKING:
        return False
    if not TRACKED_OBJECT_LABELS:
        return True
    return norm in TRACKED_OBJECT_LABELS


def _candidate_to_tracked_object(frame, candidate, label, confidence=None, *, source_suffix=""):
    bbox = candidate.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    cx = (float(bbox[0]) + float(bbox[2])) * 0.5
    cy = (float(bbox[1]) + float(bbox[3])) * 0.5
    hsv_embedding = _encode_fast_crop_embedding(frame, bbox, out_dim=DINO_DIM)
    luma = _crop_luma_mean(frame, bbox)
    norm_label = normalize_label(label)
    visual_class = ""
    if norm_label == "toy giraffe":
        visual_class = "toy_giraffe"
    elif norm_label == "baby bottle":
        visual_class = "baby_bottle"
    item = {
        "label": norm_label,
        "raw_label": candidate.get("label", norm_label),
        "x": round(float(cx), 3),
        "y": round(float(cy), 3),
        "bbox": bbox,
        "confidence": round(float(confidence if confidence is not None else candidate.get("confidence", 0.0) or 0.0), 3),
        "embedding": hsv_embedding,
        "embedding_source": "hsv",
        "hsv_embedding": hsv_embedding,
        "dino_embedding": [],
        "mask_polygon": candidate.get("mask_polygon") or default_bbox_polygon(bbox),
        "segmentation_source": candidate.get("segmentation_source", "bbox") + source_suffix,
        "crop_luma_mean": luma,
    }
    if visual_class:
        item["visual_identity_class"] = visual_class
        item["semantic_label"] = norm_label
    return item


def _normalize_polygon(points_xy, w: int, h: int, max_points: int = 32):
    if points_xy is None:
        return None
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return None
    if pts.shape[0] > max_points:
        step = max(1, int(round(pts.shape[0] / max_points)))
        pts = pts[::step][:max_points]
    poly = []
    for x, y in pts:
        poly.append([
            round(float(np.clip(x / max(w, 1), 0.0, 1.0)), 4),
            round(float(np.clip(y / max(h, 1), 0.0, 1.0)), 4),
        ])
    return poly


def _default_bbox_polygon(bbox_norm):
    x1, y1, x2, y2 = [float(v) for v in bbox_norm]
    return [
        [round(x1, 4), round(y1, 4)],
        [round(x2, 4), round(y1, 4)],
        [round(x2, 4), round(y2, 4)],
        [round(x1, 4), round(y2, 4)],
    ]


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def _dedupe_candidates_nms(candidates, iou_threshold: float = 0.55):
    deduped = []
    for candidate in candidates:
        bbox = candidate.get("bbox")
        label = str(candidate.get("label", "")).lower()
        if not bbox or len(bbox) != 4:
            continue
        keep = True
        for existing in deduped:
            if str(existing.get("label", "")).lower() != label:
                continue
            if _bbox_iou(bbox, existing.get("bbox", [0, 0, 0, 0])) >= iou_threshold:
                keep = False
                break
        if keep:
            deduped.append(candidate)
    return deduped


def _extract_detector_candidates(results, w: int, h: int):
    candidates = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if conf < 0.10:
                continue
            label = str(model.names[cls_id])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = (x2 - x1) / max(w, 1)
            bh = (y2 - y1) / max(h, 1)
            if bw < 0.02 or bh < 0.02:
                continue
            bbox_norm = [
                round(x1 / w, 4),
                round(y1 / h, 4),
                round(x2 / w, 4),
                round(y2 / h, 4),
            ]
            candidates.append(
                {
                    "label": label,
                    "bbox": bbox_norm,
                    "confidence": round(conf, 4),
                    "embedding": None,
                    "mask_polygon": _default_bbox_polygon(bbox_norm),
                    "segmentation_source": "bbox",
                }
            )
    return candidates


def _extract_segmentation_candidates(seg_results, w: int, h: int):
    candidates = []
    for result in seg_results or []:
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        polys = getattr(masks, "xy", None) if masks is not None else None
        if boxes is None:
            continue
        for idx, box in enumerate(boxes):
            conf = float(box.conf[0].item())
            if conf < 0.08:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = (x2 - x1) / max(w, 1)
            bh = (y2 - y1) / max(h, 1)
            if bw < 0.015 or bh < 0.015:
                continue
            bbox_norm = [
                round(x1 / w, 4),
                round(y1 / h, 4),
                round(x2 / w, 4),
                round(y2 / h, 4),
            ]
            poly = None
            if polys is not None and idx < len(polys):
                poly = _normalize_polygon(polys[idx], w=w, h=h, max_points=32)
            if poly is None:
                poly = _default_bbox_polygon(bbox_norm)
            candidates.append(
                {
                    "bbox": bbox_norm,
                    "confidence": round(conf, 4),
                    "mask_polygon": poly,
                }
            )
    return candidates


def _associate_segmentation(detector_candidates, segmentation_candidates, min_iou: float = 0.18):
    if not detector_candidates or not segmentation_candidates:
        return detector_candidates
    used = set()
    for det in detector_candidates:
        bbox = det.get("bbox")
        if not bbox:
            continue
        best_idx = -1
        best_iou = 0.0
        for idx, seg in enumerate(segmentation_candidates):
            if idx in used:
                continue
            iou = _bbox_iou(bbox, seg.get("bbox", [0, 0, 0, 0]))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= min_iou:
            seg = segmentation_candidates[best_idx]
            det["mask_polygon"] = seg.get("mask_polygon", det.get("mask_polygon"))
            det["segmentation_source"] = PERCEPTION_SEGMENTATION_BACKEND
            det["segmentation_iou"] = round(float(best_iou), 3)
            used.add(best_idx)
    return detector_candidates


def _collect_unmatched_segmentation_candidates(detector_candidates, segmentation_candidates, min_iou: float = 0.18):
    unmatched = []
    if not segmentation_candidates:
        return unmatched
    for seg in segmentation_candidates:
        seg_bbox = seg.get("bbox")
        if not seg_bbox:
            continue
        seg_conf = float(seg.get("confidence", 0.0) or 0.0)
        if seg_conf < PERCEPTION_UNMATCHED_SEG_MIN_CONF:
            continue
        sx1, sy1, sx2, sy2 = [float(v) for v in seg_bbox]
        area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
        if area < PERCEPTION_UNMATCHED_SEG_MIN_AREA or area > PERCEPTION_UNMATCHED_SEG_MAX_AREA:
            continue
        best_iou = 0.0
        for det in detector_candidates:
            det_bbox = det.get("bbox")
            if not det_bbox:
                continue
            iou = _bbox_iou(seg_bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
        if best_iou >= min_iou:
            continue
        item = {
            "label": "unknown_seg",
            "bbox": seg_bbox,
            "confidence": round(seg_conf, 4),
            "embedding": None,
            "mask_polygon": seg.get("mask_polygon", _default_bbox_polygon(seg_bbox)),
            "segmentation_source": PERCEPTION_SEGMENTATION_BACKEND,
        }
        unmatched.append(item)
    unmatched = sorted(unmatched, key=lambda it: float(it.get("confidence", 0.0)), reverse=True)
    unmatched = _dedupe_candidates_nms(unmatched, iou_threshold=0.45)
    return unmatched[:max(0, PERCEPTION_UNMATCHED_SEG_MAX)]


def detect_cup_and_semantic_candidates(frame):
    """Run detector + optional segmenter and return tracked objects + semantic candidates."""
    global last_dino_embedding, dino_frame_counter, semantic_dino_frame_counter, sophie_visual_tracker

    semantic_candidates = build_semantic_candidates(
        frame,
        model,
        segmentation_model=segmentation_model,
        device=YOLO_DEVICE,
        detector_conf_min=0.10,
        segmentation_conf_min=0.08,
        segmentation_source=PERCEPTION_SEGMENTATION_BACKEND,
        add_unmatched=PERCEPTION_ADD_UNMATCHED_SEG_OBJECTS,
        unmatched_min_conf=PERCEPTION_UNMATCHED_SEG_MIN_CONF,
        unmatched_min_area=PERCEPTION_UNMATCHED_SEG_MIN_AREA,
        unmatched_max_area=PERCEPTION_UNMATCHED_SEG_MAX_AREA,
        unmatched_max_items=PERCEPTION_UNMATCHED_SEG_MAX,
    )

    tracked_objects = []
    ranked_candidates = sorted(semantic_candidates, key=lambda item: item.get("confidence", 0.0), reverse=True)
    ranked_candidates = dedupe_candidates_nms(ranked_candidates, iou_threshold=0.55)
    for candidate in ranked_candidates:
        raw_label = str(candidate.get("label", "")).lower()
        label = normalize_label(raw_label)
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        if confidence < TRACKED_OBJECT_MIN_CONFIDENCE:
            continue
        if not _should_track_object_label(raw_label):
            continue
        bbox = candidate.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        cx = (float(bbox[0]) + float(bbox[2])) * 0.5
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5
        hsv_embedding = _encode_fast_crop_embedding(frame, bbox, out_dim=DINO_DIM)
        luma = _crop_luma_mean(frame, bbox)
        tracked_objects.append({
            "label": label,
            "raw_label": raw_label,
            "x": round(float(cx), 3),
            "y": round(float(cy), 3),
            "bbox": bbox,
            "confidence": round(float(confidence), 3),
            "embedding": hsv_embedding,
            "embedding_source": "hsv",
            "hsv_embedding": hsv_embedding,
            "dino_embedding": [],
            "mask_polygon": candidate.get("mask_polygon"),
            "segmentation_source": candidate.get("segmentation_source", "bbox"),
            "crop_luma_mean": luma,
        })
        if len(tracked_objects) >= max(1, MAX_TRACKED_OBJECTS):
            break

    # Sophie-specific persistent identity re-ID:
    # YOLO/FastSAM proposes current-frame boxes/masks, DINO decides whether a
    # proposal matches the remembered baby bottle or toy giraffe identity. This
    # is not a label alias and not a two-pass render; it is online object memory.
    if SOPHIE_DINO_REID_ENABLED and USE_DINO_EMBEDDING and os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() == "sophie":
        prototypes = _sophie_identity_prototypes()
        if prototypes:
            reid_matches = []
            candidate_pool = sorted(
                semantic_candidates,
                key=lambda item: (
                    str(item.get("segmentation_source", "")) != "bbox",
                    float(item.get("confidence", 0.0) or 0.0),
                    _bbox_area_norm(item.get("bbox")),
                ),
                reverse=True,
            )
            encoded = 0
            for candidate in candidate_pool:
                bbox = candidate.get("bbox")
                area = _bbox_area_norm(bbox)
                if not bbox or area < 0.001 or area > 0.34:
                    continue
                raw_label = normalize_label(candidate.get("label", ""))
                if raw_label in TRACKED_STATIC_TARGET_LABELS or raw_label in TRACKED_OBJECT_LABEL_DENYLIST:
                    continue
                try:
                    emb = encode_bbox(frame, bbox, out_dim=DINO_DIM)
                except Exception:
                    continue
                if not _nonzero_embedding(emb):
                    continue
                encoded += 1
                sims = {
                    label: _cosine_embedding(emb, proto)
                    for label, proto in prototypes.items()
                }
                if not sims:
                    continue
                label, sim = max(sims.items(), key=lambda kv: kv[1])
                other = max([v for k, v in sims.items() if k != label] or [0.0])
                margin = float(sim) - float(other)
                if sim < SOPHIE_DINO_REID_MIN_SIM or margin < SOPHIE_DINO_REID_MIN_MARGIN:
                    continue
                # Prefer actual masks over broad detector boxes; confidence is
                # identity confidence, not detector class confidence.
                score = float(sim) + 0.08 * min(1.0, float(candidate.get("confidence", 0.0) or 0.0))
                if str(candidate.get("segmentation_source", "")) != "bbox":
                    score += 0.04
                reid_matches.append((score, label, sim, margin, emb, candidate))
                if encoded >= max(1, SOPHIE_DINO_REID_MAX_CANDIDATES):
                    break
            for score, label, sim, margin, emb, candidate in sorted(reid_matches, key=lambda item: item[0], reverse=True):
                replacement = _candidate_to_tracked_object(
                    frame,
                    candidate,
                    label,
                    confidence=max(float(candidate.get("confidence", 0.0) or 0.0), float(sim)),
                    source_suffix="+dino_reid",
                )
                if replacement is None:
                    continue
                replacement["embedding"] = emb
                replacement["embedding_source"] = "dino"
                replacement["dino_embedding"] = emb
                replacement["visual_identity_label"] = label
                replacement["visual_identity_score"] = round(float(sim), 4)
                replacement["visual_identity_margin"] = round(float(margin), 4)
                replacement["semantic_label"] = label
                replaced = False
                for idx, item in enumerate(tracked_objects):
                    if normalize_label(item.get("label", "")) == label or bbox_iou(item.get("bbox", [0, 0, 0, 0]), replacement["bbox"]) >= 0.18:
                        tracked_objects[idx] = replacement
                        replaced = True
                        break
                if not replaced and len(tracked_objects) < max(1, MAX_TRACKED_OBJECTS):
                    tracked_objects.append(replacement)
                if sophie_visual_tracker is not None:
                    sophie_visual_tracker.observe(frame, label, replacement["bbox"])

    if sophie_visual_tracker is not None:
        try:
            fps = float(os.environ.get("DEMO_VIDEO_FPS", "30.0"))
        except ValueError:
            fps = 30.0
        t_s = float(dino_frame_counter) / max(1.0, fps)
        visual_boxes = sophie_visual_tracker.update(frame, t_s=t_s)
        for label, bbox in visual_boxes.items():
            if not bbox:
                continue
            cx = (float(bbox[0]) + float(bbox[2])) * 0.5
            cy = (float(bbox[1]) + float(bbox[3])) * 0.5
            hsv_embedding = _encode_fast_crop_embedding(frame, bbox, out_dim=DINO_DIM)
            luma = _crop_luma_mean(frame, bbox)
            replacement = _candidate_to_tracked_object(
                frame,
                {
                    "label": label,
                    "bbox": bbox,
                    "confidence": 0.72,
                    "mask_polygon": default_bbox_polygon(bbox),
                    "segmentation_source": "sophie_visual_tracker",
                },
                label,
                confidence=0.72,
            )
            if replacement is None:
                continue
            replaced = False
            for idx, item in enumerate(tracked_objects):
                if normalize_label(item.get("label", "")) == label or bbox_iou(item.get("bbox", [0, 0, 0, 0]), bbox) >= 0.22:
                    # Keep a DINO-confirmed mask over the appearance tracker;
                    # otherwise the tracker carries the persistent label while
                    # awaiting the next DINO-confirmed proposal.
                    if str(item.get("embedding_source", "")).lower() != "dino":
                        tracked_objects[idx] = replacement
                    replaced = True
                    break
            if not replaced and len(tracked_objects) < max(1, MAX_TRACKED_OBJECTS):
                tracked_objects.append(replacement)

    dino_frame_counter += 1
    fps = 30.0
    try:
        fps = float(os.environ.get("DEMO_VIDEO_FPS", "30.0"))
    except ValueError:
        fps = 30.0
    bootstrap_frames = max(1, int(max(0.0, DINO_BOOTSTRAP_SECONDS) * max(1.0, fps)))
    update_every = DINO_BOOTSTRAP_UPDATE_EVERY if dino_frame_counter <= bootstrap_frames else DINO_UPDATE_EVERY
    should_update_object_embedding = USE_DINO_EMBEDDING and (dino_frame_counter % max(update_every, 1) == 0)
    if should_update_object_embedding and tracked_objects:
        embeds_done = 0
        for item in tracked_objects:
            label = str(item.get("label", "")).lower()
            if embeds_done >= max(1, MAX_OBJECT_DINO_EMBEDS_PER_UPDATE) and label != "cup":
                continue
            try:
                emb = encode_bbox(frame, item["bbox"], out_dim=DINO_DIM)
                item["embedding"] = emb
                item["embedding_source"] = "dino"
                item["dino_embedding"] = emb
                embeds_done += 1
                if label == "cup":
                    last_dino_embedding = emb
            except Exception as e:
                item["embedding"] = item.get("hsv_embedding", [0.0] * DINO_DIM)
                item["embedding_source"] = "hsv"
                print(f"DINO encode warning: {e}")

    semantic_dino_frame_counter += 1
    if USE_DINO_EMBEDDING and semantic_candidates and (semantic_dino_frame_counter % max(SEMANTIC_DINO_UPDATE_EVERY, 1) == 0):
        ranked = sorted(semantic_candidates, key=lambda item: item["confidence"], reverse=True)
        for item in ranked[:max(SEMANTIC_MAX_EMBED_CANDIDATES, 1)]:
            try:
                item["embedding"] = encode_bbox(frame, item["bbox"], out_dim=SEMANTIC_DINO_DIM)
            except Exception:
                item["embedding"] = None

    return tracked_objects, semantic_candidates


def build_hand_dynamic_bboxes(image_shape, hands):
    """Build normalized dynamic bboxes around tracked hand centers from previous frame."""
    if not HAND_DYNAMIC_MASK_ENABLED:
        return []
    if not hands:
        return []

    h, w = image_shape[:2]
    radius_px = max(8.0, HAND_DYNAMIC_MASK_RADIUS_NORM * float(min(h, w)))
    dynamic_bboxes = []
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
        dynamic_bboxes.append([
            round(x1 / max(w, 1), 4),
            round(y1 / max(h, 1), 4),
            round(x2 / max(w, 1), 4),
            round(y2 / max(h, 1), 4),
        ])
    return dynamic_bboxes


def _bbox_from_hand_landmarks(landmarks_px, image_shape):
    if not isinstance(landmarks_px, list) or not landmarks_px:
        return None
    h, w = image_shape[:2]
    xs = []
    ys = []
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
    return [
        round(x1 / max(w, 1), 4),
        round(y1 / max(h, 1), 4),
        round(x2 / max(w, 1), 4),
        round(y2 / max(h, 1), 4),
    ]


def attach_jepa_features(frame, objects, hands):
    if not JEPA_ENABLED or not jepa_encoder.ready:
        return objects, hands, {"enabled": bool(JEPA_ENABLED), "ready": bool(jepa_encoder.ready), "objects_encoded": 0, "hands_encoded": 0}
    objects_encoded = 0
    ranked_objects = sorted(
        objects,
        key=lambda obj: (
            str(obj.get("label", "")).lower() in {"bottle", "baby bottle", "toy giraffe", "toy", "mouse", "donut"},
            float(obj.get("confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )
    encode_object_ids = {
        id(obj)
        for obj in ranked_objects[: max(0, int(JEPA_MAX_OBJECTS_PER_FRAME))]
    }
    for obj in objects:
        bbox = obj.get("bbox")
        if id(obj) in encode_object_ids and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            emb = jepa_encoder.encode_bbox(frame, bbox)
            obj["jepa_embedding"] = emb
            objects_encoded += 1 if emb else 0
    hands_encoded = 0
    for hand in hands:
        hbbox = _bbox_from_hand_landmarks(hand.get("landmarks_px"), frame.shape[:2])
        if hbbox is None:
            hand["jepa_embedding"] = []
            continue
        emb = jepa_encoder.encode_bbox(frame, hbbox)
        hand["jepa_embedding"] = emb
        hand["jepa_bbox"] = hbbox
        hands_encoded += 1 if emb else 0
    return objects, hands, {
        "enabled": True,
        "ready": bool(jepa_encoder.ready),
        "backend": jepa_encoder.backend,
        "objects_encoded": int(objects_encoded),
        "hands_encoded": int(hands_encoded),
    }


def enrich_detections_with_3d(detections, depth_map, camera_pose, intrinsics, extract_surface_points: bool = True):
    """Attach approximate 3D position fields to 2D detections."""
    local_sparse_map = camera_pose.get("local_sparse_map", []) if camera_pose else []
    sparse_map = camera_pose.get("sparse_map", []) if camera_pose else []
    persistent_map = camera_pose.get("persistent_map", []) if camera_pose else []
    stabilization_points = local_sparse_map or sparse_map or persistent_map
    enriched = []
    rotation_wc = np.asarray(camera_pose.get("rotation_wc", np.eye(3, dtype=np.float32)), dtype=np.float32) if camera_pose else np.eye(3, dtype=np.float32)
    if rotation_wc.shape != (3, 3):
        rotation_wc = np.eye(3, dtype=np.float32)
    camera_world = np.asarray(camera_pose.get("camera_position_world", [0.0, 0.0, 0.0]), dtype=np.float32) if camera_pose else np.zeros(3, dtype=np.float32)

    def _pixel_to_world(px: float, py: float, z: float):
        if z <= 0.0 or not np.isfinite(z):
            return None
        x_cam = ((px - float(intrinsics.get("cx", 0.0))) / max(float(intrinsics.get("fx", 1.0)), 1e-6)) * z
        y_cam = ((py - float(intrinsics.get("cy", 0.0))) / max(float(intrinsics.get("fy", 1.0)), 1e-6)) * z
        p_cam = np.asarray([x_cam, y_cam, z], dtype=np.float32)
        p_world = rotation_wc @ p_cam + camera_world
        if not np.all(np.isfinite(p_world)):
            return None
        return [round(float(p_world[0]), 4), round(float(p_world[1]), 4), round(float(p_world[2]), 4)]

    def _dimension_prior_depth(label: str, bbox):
        prior = OBJECT_DIMENSION_PRIORS_M.get(str(label or "").lower())
        if prior is None or not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            return None
        frame_h, frame_w = depth_map.shape[:2] if depth_map is not None and depth_map.size > 0 else (1080, 1920)
        bw_px = max(1.0, (float(bbox[2]) - float(bbox[0])) * float(max(frame_w, 1)))
        bh_px = max(1.0, (float(bbox[3]) - float(bbox[1])) * float(max(frame_h, 1)))
        vals = []
        height = float(prior.get("height", 0.0) or 0.0)
        diameter = float(prior.get("diameter", 0.0) or 0.0)
        if height > 0.0 and bh_px > 1.0:
            vals.append(float(intrinsics.get("fy", 1.0) or 1.0) * height / bh_px)
        if diameter > 0.0 and bw_px > 1.0:
            vals.append(float(intrinsics.get("fx", 1.0) or 1.0) * diameter / bw_px)
        vals = [v for v in vals if np.isfinite(v) and 0.15 <= v <= 5.0]
        if not vals:
            return None
        return float(np.median(np.asarray(vals, dtype=np.float32)))

    def _reproject_with_depth(bbox, z: float):
        if z <= 0.0 or not np.isfinite(z):
            return None, None
        h, w = depth_map.shape[:2]
        x1 = int(np.clip(round(float(bbox[0]) * w), 0, w - 1))
        y1 = int(np.clip(round(float(bbox[1]) * h), 0, h - 1))
        x2 = int(np.clip(round(float(bbox[2]) * w), 0, w - 1))
        y2 = int(np.clip(round(float(bbox[3]) * h), 0, h - 1))
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        fx = float(intrinsics.get("fx", 1.0) or 1.0)
        fy = float(intrinsics.get("fy", 1.0) or 1.0)
        cx = float(intrinsics.get("cx", 0.0) or 0.0)
        cy = float(intrinsics.get("cy", 0.0) or 0.0)
        p_cam = np.asarray([((u - cx) / max(fx, 1e-6)) * z, ((v - cy) / max(fy, 1e-6)) * z, z], dtype=np.float32)
        p_world = rotation_wc @ p_cam + camera_world
        return p_cam, p_world
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            enriched.append(det)
            continue

        lifted = lift_bbox_to_3d(
            bbox,
            depth_map,
            camera_pose,
            intrinsics,
            sparse_points=stabilization_points,
        )
        item = det.copy()
        item["position_camera_3d"] = lifted["position_camera_3d"]
        item["position_3d"] = lifted["position_world_3d"]
        item["velocity_3d"] = [0.0, 0.0, 0.0]
        item["depth"] = lifted["depth"]
        item["depth_confidence"] = lifted["depth_confidence"]
        item["pixel_center"] = lifted["pixel_center"]
        item["bbox_area_ratio"] = lifted["bbox_area_ratio"]
        item["landmark_support"] = lifted["landmark_support"]
        item["landmark_blend_weight"] = lifted["landmark_blend_weight"]
        prior_depth = _dimension_prior_depth(item.get("label", ""), bbox)
        if prior_depth is not None:
            current_depth = float(item.get("depth", 0.0) or 0.0)
            support = int(item.get("landmark_support", 0) or 0)
            if current_depth <= 0.0:
                use_depth = prior_depth
                prior_weight = 1.0
            else:
                ratio = prior_depth / max(current_depth, 1e-6)
                max_ratio = 1.55 if support <= 0 else 1.30
                min_ratio = 1.0 / max_ratio
                use_depth = current_depth * float(np.clip(ratio, min_ratio, max_ratio))
                prior_weight = 0.65 if support <= 0 else 0.35
                use_depth = (1.0 - prior_weight) * current_depth + prior_weight * use_depth
            p_cam, p_world = _reproject_with_depth(bbox, use_depth)
            if p_cam is not None and p_world is not None and np.isfinite(p_world).all():
                item["position_camera_3d"] = np.round(p_cam, 4).tolist()
                item["position_3d"] = np.round(p_world, 4).tolist()
                item["depth"] = round(float(use_depth), 4)
                item["dimension_prior_depth"] = round(float(prior_depth), 4)
                item["dimension_prior_weight"] = round(float(prior_weight), 3)
        fx = float(intrinsics.get("fx", 1.0) or 1.0)
        fy = float(intrinsics.get("fy", 1.0) or 1.0)
        depth_m = float(lifted.get("depth", 0.0) or 0.0)
        bw = max(0.0, float(bbox[2]) - float(bbox[0]))
        bh = max(0.0, float(bbox[3]) - float(bbox[1]))
        frame_h, frame_w = depth_map.shape[:2] if depth_map is not None and depth_map.size > 0 else (1080, 1920)
        bw_px = bw * float(max(frame_w, 1))
        bh_px = bh * float(max(frame_h, 1))
        # Approximate object support radius in metric space from bbox size at depth.
        half_w_m = max(0.0, 0.5 * bw_px * depth_m / max(fx, 1e-6))
        half_h_m = max(0.0, 0.5 * bh_px * depth_m / max(fy, 1e-6))
        radius_m = float(np.clip(max(half_w_m, half_h_m), 0.015, 0.22))
        item["proxy_radius_m"] = round(radius_m, 4)
        item["proxy_extent_m"] = [
            round(float(np.clip(2.0 * half_w_m, 0.02, 0.5)), 4),
            round(float(np.clip(2.0 * half_h_m, 0.02, 0.5)), 4),
            round(float(np.clip(radius_m * 1.35, 0.02, 0.4)), 4),
        ]
        # Object-local surface points (TSDF-lite seed) from current depth inside bbox.
        surf_pts = []
        if extract_surface_points and depth_map is not None and depth_map.size > 0:
            h, w = depth_map.shape[:2]
            x1 = int(np.clip(round(float(bbox[0]) * w), 0, w - 1))
            y1 = int(np.clip(round(float(bbox[1]) * h), 0, h - 1))
            x2 = int(np.clip(round(float(bbox[2]) * w), 0, w - 1))
            y2 = int(np.clip(round(float(bbox[3]) * h), 0, h - 1))
            if x2 > x1 and y2 > y1:
                stride = max(4, int(min(x2 - x1, y2 - y1) / 8))
                for py in range(y1, y2, stride):
                    for px in range(x1, x2, stride):
                        z = float(depth_map[py, px])
                        if not np.isfinite(z) or z <= 0.03 or z > 8.0:
                            continue
                        p = _pixel_to_world(float(px), float(py), z)
                        if p is not None:
                            surf_pts.append(p)
                        if len(surf_pts) >= 72:
                            break
                    if len(surf_pts) >= 72:
                        break
        item["surface_points_3d"] = surf_pts
        enriched.append(item)

    return enriched




def refine_hands_with_scene_scale(hands, camera_pose, intrinsics):
    """Clamp hand depth to scene-consistent range when monocular depth jitters."""
    if not hands or not isinstance(camera_pose, dict):
        return hands, {"enabled": True, "adjusted": 0, "scene_median_z": None}
    cam_pos = np.asarray(camera_pose.get("camera_position_world", [0.0, 0.0, 0.0]), dtype=np.float32)
    rot_wc = np.asarray(camera_pose.get("rotation_wc", np.eye(3, dtype=np.float32)), dtype=np.float32)
    if rot_wc.shape != (3, 3):
        rot_wc = np.eye(3, dtype=np.float32)
    rot_cw = rot_wc.T
    points = (camera_pose.get("local_sparse_map") or camera_pose.get("sparse_map") or camera_pose.get("persistent_map") or [])
    zvals = []
    for pt in points[:2400]:
        pos = pt.get("position_world") or pt.get("triangulated_position_world") or pt.get("position_world_depth_prior")
        if not (isinstance(pos, (list, tuple)) and len(pos) >= 3):
            continue
        w = np.asarray([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        c = rot_cw @ (w - cam_pos)
        z = float(c[2])
        if np.isfinite(z) and 0.05 < z < 8.0:
            zvals.append(z)
    if len(zvals) < 8:
        return hands, {"enabled": True, "adjusted": 0, "scene_median_z": None}
    scene_med = float(np.median(np.asarray(zvals, dtype=np.float32)))
    zmin = max(0.08, scene_med * 0.35)
    zmax = min(8.0, scene_med * 2.2)
    adjusted = 0
    out = []
    fx = float(intrinsics.get("fx", 1.0) or 1.0)
    fy = float(intrinsics.get("fy", 1.0) or 1.0)
    cx = float(intrinsics.get("cx", 0.0) or 0.0)
    cy = float(intrinsics.get("cy", 0.0) or 0.0)

    for hand in hands:
        h = dict(hand)
        cam_pt = h.get("position_camera_3d")
        center = h.get("center_3d")
        if not (isinstance(cam_pt, (list, tuple)) and len(cam_pt) >= 3 and isinstance(center, (list, tuple)) and len(center) >= 3):
            out.append(h)
            continue
        z = float(cam_pt[2])
        if not np.isfinite(z):
            out.append(h)
            continue
        zc = float(np.clip(z, zmin, zmax))
        if abs(zc - z) < 1e-4:
            out.append(h)
            continue
        ratio = zc / max(z, 1e-6)
        cam_arr = np.asarray([float(cam_pt[0]), float(cam_pt[1]), float(cam_pt[2])], dtype=np.float32) * ratio
        world_arr = rot_wc @ cam_arr + cam_pos
        h["position_camera_3d"] = np.round(cam_arr, 4).tolist()
        h["center_3d"] = np.round(world_arr, 4).tolist()
        lm = h.get("landmarks_3d")
        if isinstance(lm, list) and lm:
            wrist = None
            if len(lm) > 0 and isinstance(lm[0], (list, tuple)) and len(lm[0]) >= 3:
                wrist = np.asarray([float(lm[0][0]), float(lm[0][1]), float(lm[0][2])], dtype=np.float32)
            if wrist is not None and np.isfinite(wrist).all():
                nlm = []
                for pt in lm:
                    if not (isinstance(pt, (list, tuple)) and len(pt) >= 3):
                        nlm.append(pt)
                        continue
                    p = np.asarray([float(pt[0]), float(pt[1]), float(pt[2])], dtype=np.float32)
                    p2 = wrist + (p - wrist) * ratio
                    nlm.append(np.round(p2, 4).tolist())
                h["landmarks_3d"] = nlm
        adjusted += 1
        out.append(h)
    return out, {"enabled": True, "adjusted": int(adjusted), "scene_median_z": round(scene_med, 4), "zmin": round(zmin, 4), "zmax": round(zmax, 4)}


def align_hands_to_objects(hands, objects):
    """Align lifted hand 3D to nearby tracked objects using image-space proximity."""
    if not HAND_WORLD_ALIGN_RUNTIME_ENABLED or not hands or not objects:
        return hands, {"enabled": bool(HAND_WORLD_ALIGN_RUNTIME_ENABLED), "adjusted_hands": 0}

    valid_objects = []
    for obj in objects:
        pos = obj.get("position_3d")
        center = obj.get("pixel_center")
        if (
            isinstance(pos, (list, tuple))
            and len(pos) >= 3
            and isinstance(center, (list, tuple))
            and len(center) >= 2
        ):
            valid_objects.append(obj)
    if not valid_objects:
        return hands, {"enabled": True, "adjusted_hands": 0}

    adjusted = []
    adjusted_count = 0
    max_shift = max(0.05, float(HAND_WORLD_ALIGN_MAX_SHIFT_M))
    alpha = min(1.0, max(0.0, float(HAND_WORLD_ALIGN_ALPHA)))
    norm_radius = max(0.01, float(HAND_WORLD_ALIGN_IMAGE_NORM_RADIUS))

    for hand in hands:
        item = dict(hand)
        hnorm = hand.get("image_norm_center")
        hpos = hand.get("center_3d")
        if not (isinstance(hnorm, (list, tuple)) and len(hnorm) >= 2 and isinstance(hpos, (list, tuple)) and len(hpos) >= 3):
            adjusted.append(item)
            continue
        hx, hy = float(hnorm[0]), float(hnorm[1])
        nearest = None
        nearest_dist = 1e9
        for obj in valid_objects:
            bbox = obj.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                onx = 0.5 * (float(bbox[0]) + float(bbox[2]))
                ony = 0.5 * (float(bbox[1]) + float(bbox[3]))
            else:
                continue
            d = math.hypot(hx - onx, hy - ony)
            if d < nearest_dist:
                nearest_dist = d
                nearest = obj
        if nearest is None or nearest_dist > norm_radius:
            adjusted.append(item)
            continue

        obj_pos = np.asarray(nearest.get("position_3d")[:3], dtype=np.float32)
        hand_pos = np.asarray(hpos[:3], dtype=np.float32)
        delta = obj_pos - hand_pos
        dnorm = float(np.linalg.norm(delta))
        if not np.isfinite(dnorm) or dnorm <= 1e-6:
            adjusted.append(item)
            continue
        if dnorm > max_shift:
            delta = delta * (max_shift / dnorm)

        shift = delta * alpha
        hand_aligned = hand_pos + shift
        item["center_3d"] = np.round(hand_aligned, 4).tolist()

        lm3d = hand.get("landmarks_3d")
        if isinstance(lm3d, list) and lm3d:
            shifted = []
            for pt in lm3d:
                if not (isinstance(pt, (list, tuple)) and len(pt) >= 3):
                    shifted.append(pt)
                    continue
                p = np.asarray(pt[:3], dtype=np.float32) + shift
                shifted.append(np.round(p, 4).tolist())
            item["landmarks_3d"] = shifted

        adjusted_count += 1
        adjusted.append(item)

    return adjusted, {
        "enabled": True,
        "adjusted_hands": int(adjusted_count),
        "image_norm_radius": float(norm_radius),
        "alpha": float(alpha),
        "max_shift_m": float(max_shift),
    }


def encode_depth_debug(depth_map, max_width=160):
    """Encode a small false-color preview of the depth map for the UI."""
    global _depth_debug_low_ema, _depth_debug_high_ema
    if depth_map is None or depth_map.size == 0:
        return None

    source_height, source_width = depth_map.shape[:2]
    scale = min(1.0, float(max_width) / max(float(source_width), 1.0))
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    preview = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_AREA)
    finite = preview[np.isfinite(preview)]
    if finite.size > 8:
        low = float(np.percentile(finite, DEPTH_DEBUG_LOW_PERCENTILE))
        high = float(np.percentile(finite, DEPTH_DEBUG_HIGH_PERCENTILE))
    else:
        low = float(DEPTH_MIN_RANGE)
        high = float(DEPTH_MAX_RANGE)
    if not np.isfinite(low):
        low = float(DEPTH_MIN_RANGE)
    if not np.isfinite(high):
        high = float(DEPTH_MAX_RANGE)
    if high <= low + 1e-5:
        high = low + 1e-5

    alpha = float(np.clip(DEPTH_DEBUG_EMA_ALPHA, 0.0, 1.0))
    if _depth_debug_low_ema is None:
        _depth_debug_low_ema = low
    else:
        _depth_debug_low_ema = (1.0 - alpha) * float(_depth_debug_low_ema) + alpha * low
    if _depth_debug_high_ema is None:
        _depth_debug_high_ema = high
    else:
        _depth_debug_high_ema = (1.0 - alpha) * float(_depth_debug_high_ema) + alpha * high

    low_vis = float(_depth_debug_low_ema)
    high_vis = float(max(low_vis + 1e-5, _depth_debug_high_ema))
    normalized = (preview - low_vis) / max(high_vis - low_vis, 1e-6)
    normalized = np.clip(normalized, 0.0, 1.0) * 255.0
    normalized = normalized.astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".jpg", heatmap, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        return None

    return {
        "width": width,
        "height": height,
        "mime_type": "image/jpeg",
        "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def is_effectively_empty(state_vec):
    """True when a state vector is missing or has negligible position.

    The planner and transition collection use this to avoid spurious
    writes when detections are noisy or absent.
    """
    return state_vec is None or len(state_vec) < 2 or sum(abs(v) for v in state_vec[:2]) < 1e-6


def maybe_collect_transition(curr_state_vec):
    """Conditionally save a transition (previous_state, action, next_state)."""
    global prev_state_vec, prev_move_vec

    if not COLLECT_TRANSITIONS:
        prev_state_vec = curr_state_vec
        prev_move_vec = None
        return False

    if is_effectively_empty(curr_state_vec):
        prev_move_vec = None
        return False

    if prev_state_vec is None or is_effectively_empty(prev_state_vec):
        prev_state_vec = curr_state_vec
        prev_move_vec = None
        return False

    prev_x, prev_y = prev_state_vec[0], prev_state_vec[1]
    curr_x, curr_y = curr_state_vec[0], curr_state_vec[1]

    dx = curr_x - prev_x
    dy = curr_y - prev_y
    move = math.sqrt(dx * dx + dy * dy)

    MIN_MOVE = 0.012
    MAX_MOVE = 0.25

    if move < MIN_MOVE:
        return False

    if move > MAX_MOVE:
        prev_state_vec = curr_state_vec
        prev_move_vec = None
        return False

    if prev_move_vec is not None:
        prev_dx, prev_dy = prev_move_vec
        ddx = abs(prev_dx - dx)
        ddy = abs(prev_dy - dy)
        if ddx > 0.30 or ddy > 0.30:
            prev_state_vec = curr_state_vec
            prev_move_vec = None
            return False

    norm = move + 1e-6
    action_vec = [dx / norm, dy / norm]

    next_state_vec = curr_state_vec.copy()
    next_state_vec[2] = dx
    next_state_vec[3] = dy

    save_transition(prev_state_vec, action_vec, next_state_vec)
    print(
        f"Saved transition: "
        f"cup ({prev_x:.3f},{prev_y:.3f}) -> ({curr_x:.3f},{curr_y:.3f}), "
        f"move={move:.4f}, action=({action_vec[0]:.3f},{action_vec[1]:.3f})"
    )

    prev_state_vec = next_state_vec
    prev_move_vec = (dx, dy)
    return True


def answer_query(query):
    """Handle simple queries coming from websocket clients."""
    global HAND_WORLD_ALIGN_RUNTIME_ENABLED
    qtype = query.get("query")

    if qtype == "simulate_actions":
        gate = compute_demo_gate(
            getattr(state, "camera_pose", None),
            getattr(state, "hands", []),
            getattr(state, "hand_object_interactions", []),
        )
        result = simulate_all_actions(state)
        result["guidance_gate"] = gate
        if not gate.get("allow_guidance", False):
            result["explanation"] = None
            result["gated"] = True
            prior_message = str(result.get("message", "Guidance is currently gated."))
            result["message"] = f"{prior_message} Guidance gated: {gate.get('reason', 'unknown')}."
        return {"type": "query_result", "result": result}

    if qtype == "reset_world_model":
        reset_runtime_state()
        return {"type": "query_result", "result": {"ok": True, "message": "world model reset"}}

    if qtype == "apply_label_refinements":
        result = state.apply_label_refinements(query.get("labels", []))
        return {"type": "query_result", "result": {"ok": True, **result}}

    if qtype == "interaction_guidance":
        result = build_interaction_guidance(state)
        result["query_source"] = query.get("source")
        result["source_frame_timestamp"] = query.get("frame_timestamp")
        result["server_response_time_ms"] = int(time.time() * 1000)
        return {"type": "query_result", "result": result}

    if qtype == "set_hand_world_alignment":
        enabled = bool(query.get("enabled", True))
        HAND_WORLD_ALIGN_RUNTIME_ENABLED = enabled
        return {
            "type": "query_result",
            "result": {
                "ok": True,
                "hand_world_alignment_enabled": bool(HAND_WORLD_ALIGN_RUNTIME_ENABLED),
            },
        }

    return {
        "type": "query_result",
        "result": {
            "ok": False,
            "message": f"Unknown query type: {qtype}"
        }
    }


def _merge_maps(base_points: list[dict] | None, extra_points: list[dict] | None, max_points: int = 1200) -> list[dict]:
    base_points = list(base_points or [])
    extra_points = list(extra_points or [])
    if not extra_points:
        return base_points
    merged = base_points + extra_points
    merged.sort(
        key=lambda item: (
            0 if str(item.get("status", "")).lower() == "visible" else 1,
            -float(item.get("quality", 0.0) or 0.0),
            -float(item.get("hits", 0.0) or 0.0),
        )
    )
    return merged[: max(120, int(max_points))]


async def handler(websocket):
    """Main websocket handler: process incoming frames and queries."""
    frame_counter = 0
    first_frame_ts = None
    async for message in websocket:
        data = json.loads(message)

        if data["type"] == "frame":
            frame_counter += 1
            ts_val = data.get("timestamp")
            if first_frame_ts is None and isinstance(ts_val, (int, float)):
                first_frame_ts = float(ts_val)
            t0 = time.perf_counter()
            frame = decode_data_url_image(data["image"])
            t1 = time.perf_counter()
            elapsed_s = (
                (float(ts_val) - float(first_frame_ts)) / 1000.0
                if isinstance(ts_val, (int, float)) and isinstance(first_frame_ts, (int, float))
                else None
            )

            objects, semantic_candidates = await asyncio.to_thread(detect_cup_and_semantic_candidates, frame)
            t2 = time.perf_counter()

            raw_depth_map = estimate_depth(frame)
            colmap_depth_prior_debug = {"enabled": False, "mode": "disabled"}
            if colmap_depth_prior is not None:
                dynamic_bboxes = []
                if COLMAP_PRIOR_EXCLUDE_DYNAMIC:
                    for obj in objects:
                        bbox = obj.get("bbox")
                        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            dynamic_bboxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
                    dynamic_bboxes.extend(build_hand_dynamic_bboxes(frame.shape[:2], getattr(state, "hands", [])))
                raw_depth_map, colmap_depth_prior_debug = colmap_depth_prior.align_depth(
                    raw_depth_map,
                    frame_counter,
                    elapsed_s=(
                        elapsed_s
                    ),
                    dynamic_bboxes=dynamic_bboxes,
                )
                colmap_depth_prior_debug["exclude_dynamic"] = bool(COLMAP_PRIOR_EXCLUDE_DYNAMIC)
                colmap_depth_prior_debug["dynamic_boxes"] = int(len(dynamic_bboxes))
            t_depth = time.perf_counter()
            intrinsics = infer_camera_intrinsics(frame.shape[1], frame.shape[0])

            semantic_info = {
                "stable_tracks": [],
                "dynamic_bboxes": [],
                "num_tracks": 0,
                "num_stable_tracks": 0,
                "num_dynamic_tracks": 0,
            }
            semantic_mask = None
            hand_dynamic_bboxes = build_hand_dynamic_bboxes(frame.shape[:2], getattr(state, "hands", []))
            if SEMANTIC_ENABLED:
                semantic_info = semantic_stabilizer.update(semantic_candidates)
                merged_dynamic_bboxes = list(semantic_info.get("dynamic_bboxes", [])) + hand_dynamic_bboxes
                semantic_mask = build_foreground_mask(frame.shape[:2], merged_dynamic_bboxes)
                semantic_info["hand_dynamic_bboxes"] = hand_dynamic_bboxes
                semantic_info["num_hand_dynamic_bboxes"] = len(hand_dynamic_bboxes)
            elif hand_dynamic_bboxes:
                semantic_mask = build_foreground_mask(frame.shape[:2], hand_dynamic_bboxes)
                semantic_info["hand_dynamic_bboxes"] = hand_dynamic_bboxes
                semantic_info["num_hand_dynamic_bboxes"] = len(hand_dynamic_bboxes)

            camera_pose = slam_backend.update(
                frame,
                depth_map=raw_depth_map,
                intrinsics=intrinsics,
                semantic_mask=semantic_mask,
            )
            t_pose = time.perf_counter()
            depth_map, depth_stabilization = stabilize_depth_with_anchors(raw_depth_map, camera_pose)
            camera_pose = slam_backend.refine_visible_landmarks(
                depth_map,
                intrinsics,
                camera_pose,
                semantic_mask=semantic_mask,
            )
            fusion_debug = {"enabled": False, "added_samples": 0, "voxels": 0}
            fused_map = []
            if isinstance(camera_pose, dict) and camera_pose:
                fusion_debug = depth_fusion.update(
                    depth_map,
                    intrinsics,
                    camera_pose,
                    dynamic_mask=semantic_mask,
                )
                fused_map = depth_fusion.export_points()
                if DEPTH_FUSION_MAX_SEND_POINTS > 0 and len(fused_map) > DEPTH_FUSION_MAX_SEND_POINTS:
                    fused_map = sorted(
                        fused_map,
                        key=lambda p: (float(p.get("quality", 0.0)), float(p.get("hits", 0))),
                        reverse=True,
                    )[:DEPTH_FUSION_MAX_SEND_POINTS]
                camera_pose["fused_map"] = fused_map
                if DEPTH_FUSION_BLEND_IN_PERSISTENT:
                    persistent_map = camera_pose.get("persistent_map", [])
                    camera_pose["persistent_map"] = _merge_maps(
                        persistent_map,
                        fused_map,
                        max_points=max(240, DEPTH_FUSION_MAX_SEND_POINTS + 420),
                    )
            hands, hand_debug = hand_tracker.detect(frame, depth_map, intrinsics, camera_pose)
            use_static_surface = (
                elapsed_s is None
                or float(elapsed_s) <= max(0.0, float(OBJECT_SURFACE_STATIC_SECONDS))
            )
            objects = enrich_detections_with_3d(
                objects,
                depth_map,
                camera_pose,
                intrinsics,
                extract_surface_points=bool(use_static_surface),
            )
            objects, hands, jepa_debug = attach_jepa_features(frame, objects, hands)
            if JEPA_USE_FOR_CONTACT:
                for hand in hands:
                    nearest = None
                    hand_emb = hand.get("jepa_embedding", [])
                    for obj in objects:
                        jepa = jepa_interaction_score(
                            hand_emb,
                            obj.get("jepa_embedding", []),
                            float(np.linalg.norm(
                                np.asarray(hand.get("center_3d", [0.0, 0.0, 0.0]), dtype=np.float32)
                                - np.asarray(obj.get("position_3d", [0.0, 0.0, 0.0]), dtype=np.float32)
                            )) if hand.get("center_3d") and obj.get("position_3d") else 9.9,
                            near_distance_m=0.22,
                        )
                        score = float(jepa.get("jepa_interaction_score", 0.0))
                        if nearest is None or score > nearest.get("jepa_interaction_score", 0.0):
                            nearest = {
                                "object_id": obj.get("id"),
                                "label": obj.get("label"),
                                **jepa,
                            }
                    if nearest is not None:
                        hand["jepa_nearest"] = nearest
            hands, hand_scale_debug = refine_hands_with_scene_scale(hands, camera_pose, intrinsics)
            hands, hand_alignment_debug = align_hands_to_objects(hands, objects)
            depth_debug = encode_depth_debug(depth_map)
            t_world = time.perf_counter()
            demo_gate = compute_demo_gate(
                camera_pose,
                hands,
                getattr(state, "hand_object_interactions", []),
            )
            world_debug = {
                **summarize_depth(depth_map),
                "raw_depth": summarize_depth(raw_depth_map),
                "depth_stabilization": depth_stabilization,
                "colmap_depth_prior": colmap_depth_prior_debug,
                "depth_fusion": fusion_debug,
                "demo_gate": demo_gate,
                "hand_tracking": hand_debug,
                "hand_world_scale": hand_scale_debug,
                "hand_world_alignment": hand_alignment_debug,
                "jepa": jepa_debug,
                "intrinsics": {
                    "fx": round(float(intrinsics["fx"]), 2),
                    "fy": round(float(intrinsics["fy"]), 2),
                    "cx": round(float(intrinsics["cx"]), 2),
                    "cy": round(float(intrinsics["cy"]), 2),
                    "source": intrinsics.get("source", "unknown"),
                    "fov_deg": round(float(intrinsics.get("fov_deg", 0.0)), 2),
                },
                "num_objects": len(objects),
                "object_labels": sorted(list({str(obj.get("label", "unknown")) for obj in objects})),
                "object_surface_static_seconds": float(OBJECT_SURFACE_STATIC_SECONDS),
                "object_surface_static_phase": bool(use_static_surface),
                "segmentation_associated_objects": int(
                    sum(1 for obj in objects if str(obj.get("segmentation_source", "bbox")) != "bbox")
                ),
                "perception": {
                    "detector_mode": PERCEPTION_DETECTOR_MODE,
                    "detector_model_path": detector_model_path,
                    "open_vocab_status": OPEN_VOCAB_STATUS,
                    "segmentation_enabled": segmentation_model is not None,
                    "segmentation_backend": PERCEPTION_SEGMENTATION_BACKEND,
                    "segmentation_model_path": segmentation_model_path if segmentation_model is not None else None,
                    "add_unmatched_seg_objects": bool(PERCEPTION_ADD_UNMATCHED_SEG_OBJECTS),
                    "unmatched_seg_max": int(max(0, PERCEPTION_UNMATCHED_SEG_MAX)),
                    "vlm_refiner_enabled": bool(VLM_REFINER_ENABLED),
                    "vlm_refiner_every_n_keyframes": int(max(VLM_REFINER_EVERY_N_KEYFRAMES, 1)),
                },
                "active_tracks": camera_pose.get("active_tracks", 0),
                "sparse_landmark_count": camera_pose.get("sparse_landmark_count", 0),
                "visible_landmark_count": camera_pose.get("visible_landmark_count", 0),
                "local_visible_landmark_count": camera_pose.get("local_visible_landmark_count", 0),
                "persistent_landmark_count": camera_pose.get("persistent_landmark_count", 0),
                "missing_landmark_count": camera_pose.get("missing_landmark_count", 0),
                "landmark_lifecycle": camera_pose.get("landmark_lifecycle", {}),
                "keyframes": camera_pose.get("keyframes", 0),
                "stable_landmark_count": camera_pose.get("stable_landmark_count", 0),
                "stable_2d_landmark_count": camera_pose.get("stable_2d_landmark_count", 0),
                "geometry_verified_landmark_count": camera_pose.get("geometry_verified_landmark_count", 0),
                "triangulated_landmark_count": camera_pose.get("triangulated_landmark_count", 0),
                "dynamic_landmark_count": camera_pose.get("dynamic_landmark_count", 0),
                "triangulation": camera_pose.get("triangulation", {}),
                "mean_stable_reprojection_error": camera_pose.get("mean_stable_reprojection_error"),
                "latest_keyframe": camera_pose.get("latest_keyframe"),
                "covisibility_edges": camera_pose.get("covisibility_edges", 0),
                "latest_covisible_keyframes": camera_pose.get("latest_covisible_keyframes", []),
                "local_keyframes": camera_pose.get("local_keyframes", []),
                "local_landmark_count": camera_pose.get("local_landmark_count", 0),
                "geometric_inlier_count": camera_pose.get("geometric_inlier_count", 0),
                "pnp_anchor_scope": camera_pose.get("pnp_anchor_scope", "none"),
                "local_keyframe_baseline": camera_pose.get("local_keyframe_baseline", 0.0),
                "ba_lite": camera_pose.get("ba_lite", {}),
                "sliding_ba": camera_pose.get("sliding_ba", {}),
                "pose_source": camera_pose.get("pose_source", "unknown"),
                "slam_backend": camera_pose.get("slam_backend", "unknown"),
                "pnp_inliers": camera_pose.get("pnp_inliers", 0),
                "pnp_reprojection_error": camera_pose.get("pnp_reprojection_error"),
                "camera_position_world": camera_pose.get("camera_position_world"),
                "semantic_stabilization_enabled": SEMANTIC_ENABLED,
                "semantic_candidates": len(semantic_candidates),
                "semantic_stable_tracks": semantic_info.get("num_stable_tracks", 0),
                "semantic_dynamic_tracks": semantic_info.get("num_dynamic_tracks", 0),
                "semantic_tracks": semantic_info.get("stable_tracks", []),
                "hand_dynamic_mask_enabled": bool(HAND_DYNAMIC_MASK_ENABLED),
                "hand_dynamic_mask_boxes": semantic_info.get("num_hand_dynamic_bboxes", 0),
                "hands_detected": len(hands),
                "hand_object_interactions": state.hand_object_interactions,
                "fused_map_points": len(fused_map),
                "object_landmark_support": [
                    {
                        "label": obj.get("label"),
                        "landmark_support": obj.get("landmark_support", 0),
                        "landmark_blend_weight": obj.get("landmark_blend_weight", 0.0),
                    }
                    for obj in objects
                ],
            }

            state.update(
                objects,
                camera_pose=camera_pose,
                hands=hands,
                world_debug=world_debug,
                sparse_map=camera_pose.get("sparse_map", []),
            )
            curr_state_vec = state.get_state_vector()
            saved = maybe_collect_transition(curr_state_vec)
            interaction_capture_saved = maybe_capture_interaction_frame(
                frame_counter,
                elapsed_s,
                state,
                world_debug,
            )
            t3 = time.perf_counter()

            try:
                world_state_export = state.export_world_state()
                payload = sanitize_for_json({
                    "type": "state_updated",
                    "objects": state.export_objects(),
                    "object_memory": world_state_export.get("object_memory", []),
                    "objects_3d": world_state_export["objects_3d"],
                    "camera_pose": world_state_export["camera_pose"],
                    "hands": world_state_export["hands"],
                    "hand_object_interactions": world_state_export.get("hand_object_interactions", []),
                    "manipulation_events": world_state_export.get("manipulation_events", []),
                    "learned_manipulation_events": world_state_export.get("learned_manipulation_events", []),
                    "hand_trajectories": world_state_export.get("hand_trajectories", []),
                    "static_targets": world_state_export.get("static_targets", []),
                    "world_debug": world_state_export["world_debug"],
                    "sparse_map": world_state_export["sparse_map"],
                    "local_sparse_map": camera_pose.get("local_sparse_map", []),
                    "depth_debug": depth_debug,
                    "frame_timestamp": data.get("timestamp"),
                    "frame_width": data.get("frame_width"),
                    "frame_height": data.get("frame_height"),
                    "capture_ms": data.get("capture_ms"),
                    "server_decode_ms": (t1 - t0) * 1000.0,
                    "server_detect_ms": (t2 - t1) * 1000.0,
                    "server_depth_ms": (t_depth - t2) * 1000.0,
                    "server_pose_ms": (t_pose - t_depth) * 1000.0,
                    "server_world_ms": (t_world - t_pose) * 1000.0,
                    "server_total_ms": (t3 - t0) * 1000.0,
                    "server_time": time.time(),
                    "transition_saved": saved,
                    "transitions_file": str(Path(TRANSITIONS_FILE).resolve()),
                    "interaction_capture_saved": interaction_capture_saved,
                    "interaction_capture_file": str(Path(INTERACTION_CAPTURE_FILE).resolve()),
                })
                await websocket.send(json.dumps(payload, allow_nan=False))
            except websockets.exceptions.ConnectionClosed:
                break

        elif data["type"] == "query":
            response = sanitize_for_json(answer_query(data))
            try:
                await websocket.send(json.dumps(response, allow_nan=False))
            except websockets.exceptions.ConnectionClosed:
                break


async def main():
    async with websockets.serve(
        handler,
        "localhost",
        8090,
        max_size=2**24,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=10,
    ):
        print("World model server running on ws://localhost:8090")
        print(f"Automatic transition collection: {'ON' if COLLECT_TRANSITIONS else 'OFF'}")
        print(f"Automatic interaction capture: {'ON' if AUTO_INTERACTION_CAPTURE else 'OFF'}")
        print(f"Transitions file: {Path(TRANSITIONS_FILE).resolve()}")
        print(f"Interaction capture file: {Path(INTERACTION_CAPTURE_FILE).resolve()}")
        print(f"YOLO device: {YOLO_DEVICE}")
        print(f"Perception detector mode: {PERCEPTION_DETECTOR_MODE} ({detector_model_path})")
        print(f"Open-vocab status: {OPEN_VOCAB_STATUS}")
        print(
            "Segmentation model: "
            + (
                f"ON [{PERCEPTION_SEGMENTATION_BACKEND}] ({segmentation_model_path})"
                if segmentation_model is not None
                else "OFF (bbox fallback)"
            )
        )
        print(f"DINO embeddings: {'ON' if USE_DINO_EMBEDDING else 'OFF'}")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWorld model server stopped.")

