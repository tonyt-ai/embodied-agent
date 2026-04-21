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

from camera_tracker import CameraTracker
from depth import (
    DEPTH_MAX_RANGE,
    DEPTH_MIN_RANGE,
    estimate_depth,
    stabilize_depth_with_anchors,
    summarize_depth,
)
from dino_encoder import encode_bbox
from lift_to_3d import infer_camera_intrinsics, lift_bbox_to_3d
from pathlib import Path
from planner import simulate_all_actions
from ultralytics import YOLO
from world_state import WorldState

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load object detector (YOLO) from local model file
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")
YOLO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(YOLO_MODEL_PATH)
model = model.to(YOLO_DEVICE)

# File to append collected transitions (JSON lines)
TRANSITIONS_FILE = os.path.join(DATA_DIR, "transitions.jsonl")

# Runtime flags and DINO configuration
parser = argparse.ArgumentParser()
parser.add_argument("--capture", action="store_true", help="Enable transition capture mode")
args = parser.parse_args()

COLLECT_TRANSITIONS = args.capture
USE_DINO_EMBEDDING = True
DINO_DIM = 32
DINO_UPDATE_EVERY = 8   # recompute embedding every 8 detected frames

# Simple cache for the most recent DINO embedding to avoid frequent encodes
last_dino_embedding = [0.0] * DINO_DIM
dino_frame_counter = 0

# State used for collecting transitions across frames
prev_state_vec = None
prev_move_vec = None
state = WorldState(collection_mode=COLLECT_TRANSITIONS)
camera_tracker = CameraTracker()


def save_transition(state_vec, action_vec, next_state_vec):
    """Append a single transition record to the transitions file."""
    record = {
        "state": state_vec,
        "action": action_vec,
        "next_state": next_state_vec,
    }
    with open(TRANSITIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


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


def detect_best_cup(frame):
    """Run YOLO on the frame and return one best 'cup' detection.

    Performs simple filtering on size and confidence and optionally
    computes a DINO embedding for the crop (cached every N frames).
    Returns an empty list when no suitable cup is found, or a list
    with a single object dict compatible with `WorldState.update()`.
    """
    global last_dino_embedding, dino_frame_counter

    results = model(frame, verbose=False, device=YOLO_DEVICE)
    h, w = frame.shape[:2]

    best = None
    best_conf = -1.0

    # Iterate over detection results and pick the highest-confidence cup
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = model.names[cls_id]

            if label != "cup":
                continue
            if conf < 0.10:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if bw < 0.02 or bh < 0.02:
                continue

            if conf > best_conf:
                best_conf = conf
                best = (x1, y1, x2, y2, conf)

    if best is None:
        return []

    x1, y1, x2, y2, conf = best
    bbox_norm = [
        round(x1 / w, 3),
        round(y1 / h, 3),
        round(x2 / w, 3),
        round(y2 / h, 3),
    ]
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h

    emb = last_dino_embedding
    if USE_DINO_EMBEDDING:
        dino_frame_counter += 1
        if dino_frame_counter % DINO_UPDATE_EVERY == 0:
            try:
                emb = encode_bbox(frame, bbox_norm, out_dim=DINO_DIM)
                last_dino_embedding = emb
            except Exception as e:
                print(f"DINO encode warning: {e}")
                emb = last_dino_embedding

    return [{
        "label": "cup",
        "x": round(cx, 3),
        "y": round(cy, 3),
        "bbox": bbox_norm,
        "confidence": round(conf, 3),
        "embedding": emb,
    }]


def enrich_detections_with_3d(detections, depth_map, camera_pose, intrinsics):
    """Attach approximate 3D position fields to 2D detections."""
    sparse_map = camera_pose.get("sparse_map", []) if camera_pose else []
    persistent_map = camera_pose.get("persistent_map", []) if camera_pose else []
    stabilization_points = sparse_map or persistent_map
    enriched = []
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
        enriched.append(item)

    return enriched
def encode_depth_debug(depth_map, max_width=160):
    """Encode a small false-color preview of the depth map for the UI."""
    if depth_map is None or depth_map.size == 0:
        return None

    source_height, source_width = depth_map.shape[:2]
    scale = min(1.0, float(max_width) / max(float(source_width), 1.0))
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    preview = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_AREA)
    normalized = (preview - DEPTH_MIN_RANGE) / max(DEPTH_MAX_RANGE - DEPTH_MIN_RANGE, 1e-6)
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
    qtype = query.get("query")

    if qtype == "simulate_actions":
        result = simulate_all_actions(state)
        return {"type": "query_result", "result": result}

    return {
        "type": "query_result",
        "result": {
            "ok": False,
            "message": f"Unknown query type: {qtype}"
        }
    }


async def handler(websocket):
    """Main websocket handler: process incoming frames and queries."""
    async for message in websocket:
        data = json.loads(message)

        if data["type"] == "frame":
            t0 = time.perf_counter()
            frame = decode_data_url_image(data["image"])
            t1 = time.perf_counter()

            objects = await asyncio.to_thread(detect_best_cup, frame)
            t2 = time.perf_counter()

            raw_depth_map = estimate_depth(frame)
            t_depth = time.perf_counter()
            intrinsics = infer_camera_intrinsics(frame.shape[1], frame.shape[0])
            camera_pose = camera_tracker.update(frame, depth_map=raw_depth_map, intrinsics=intrinsics)
            t_pose = time.perf_counter()
            depth_map, depth_stabilization = stabilize_depth_with_anchors(raw_depth_map, camera_pose)
            camera_pose = camera_tracker.refine_visible_landmarks(depth_map, intrinsics, camera_pose)
            objects = enrich_detections_with_3d(objects, depth_map, camera_pose, intrinsics)
            depth_debug = encode_depth_debug(depth_map)
            t_world = time.perf_counter()
            world_debug = {
                **summarize_depth(depth_map),
                "raw_depth": summarize_depth(raw_depth_map),
                "depth_stabilization": depth_stabilization,
                "intrinsics": {
                    "fx": round(float(intrinsics["fx"]), 2),
                    "fy": round(float(intrinsics["fy"]), 2),
                    "cx": round(float(intrinsics["cx"]), 2),
                    "cy": round(float(intrinsics["cy"]), 2),
                },
                "num_objects": len(objects),
                "active_tracks": camera_pose.get("active_tracks", 0),
                "sparse_landmark_count": camera_pose.get("sparse_landmark_count", 0),
                "visible_landmark_count": camera_pose.get("visible_landmark_count", 0),
                "persistent_landmark_count": camera_pose.get("persistent_landmark_count", 0),
                "missing_landmark_count": camera_pose.get("missing_landmark_count", 0),
                "landmark_lifecycle": camera_pose.get("landmark_lifecycle", {}),
                "keyframes": camera_pose.get("keyframes", 0),
                "stable_landmark_count": camera_pose.get("stable_landmark_count", 0),
                "mean_stable_reprojection_error": camera_pose.get("mean_stable_reprojection_error"),
                "latest_keyframe": camera_pose.get("latest_keyframe"),
                "covisibility_edges": camera_pose.get("covisibility_edges", 0),
                "latest_covisible_keyframes": camera_pose.get("latest_covisible_keyframes", []),
                "local_keyframes": camera_pose.get("local_keyframes", []),
                "local_landmark_count": camera_pose.get("local_landmark_count", 0),
                "ba_lite": camera_pose.get("ba_lite", {}),
                "pose_source": camera_pose.get("pose_source", "unknown"),
                "pnp_inliers": camera_pose.get("pnp_inliers", 0),
                "pnp_reprojection_error": camera_pose.get("pnp_reprojection_error"),
                "camera_position_world": camera_pose.get("camera_position_world"),
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
                hands=[],
                world_debug=world_debug,
                sparse_map=camera_pose.get("sparse_map", []),
            )
            curr_state_vec = state.get_state_vector()
            saved = maybe_collect_transition(curr_state_vec)
            t3 = time.perf_counter()

            try:
                world_state_export = state.export_world_state()
                await websocket.send(json.dumps({
                    "type": "state_updated",
                    "objects": state.export_objects(),
                    "objects_3d": world_state_export["objects_3d"],
                    "camera_pose": world_state_export["camera_pose"],
                    "hands": world_state_export["hands"],
                    "world_debug": world_state_export["world_debug"],
                    "sparse_map": world_state_export["sparse_map"],
                    "depth_debug": depth_debug,
                    "frame_timestamp": data.get("timestamp"),
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
                }))
            except websockets.exceptions.ConnectionClosed:
                break

        elif data["type"] == "query":
            response = answer_query(data)
            try:
                await websocket.send(json.dumps(response))
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
        print(f"Transitions file: {Path(TRANSITIONS_FILE).resolve()}")
        print(f"YOLO device: {YOLO_DEVICE}")
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

