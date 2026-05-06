from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from dino_encoder import encode_bbox
from perception_candidates import bbox_iou, build_semantic_candidates, dedupe_candidates_nms
from world_state import WorldState


def hsv_embedding(frame: np.ndarray, bbox_norm, out_dim: int = 32):
    h, w = frame.shape[:2]
    if not bbox_norm or len(bbox_norm) != 4:
        return [0.0] * out_dim
    x1 = int(max(0, min(w - 1, round(float(bbox_norm[0]) * w))))
    y1 = int(max(0, min(h - 1, round(float(bbox_norm[1]) * h))))
    x2 = int(max(0, min(w, round(float(bbox_norm[2]) * w))))
    y2 = int(max(0, min(h, round(float(bbox_norm[3]) * h))))
    if x2 <= x1 or y2 <= y1:
        return [0.0] * out_dim
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return [0.0] * out_dim
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    vec = np.concatenate([
        cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten(),
        cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten(),
        cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten(),
    ]).astype(np.float32)
    if vec.size < out_dim:
        vec = np.pad(vec, (0, out_dim - vec.size), mode="constant")
    else:
        vec = vec[:out_dim]
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec = vec / norm
    return vec.tolist()


def should_track(label: str) -> bool:
    label = str(label or "").lower()
    if label in {"person", "dining table", "chair", "tv", "couch", "potted plant"}:
        return False
    return True


def build_detections(frame, yolo, seg_model, device, args, mode: str, sample_no: int):
    candidates = build_semantic_candidates(
        frame,
        yolo,
        segmentation_model=seg_model,
        device=device,
        detector_conf_min=args.confidence,
        segmentation_conf_min=0.08,
        segmentation_source=args.segmentation_backend,
        add_unmatched=args.add_unmatched_seg_objects,
        unmatched_min_conf=args.unmatched_seg_min_confidence,
        unmatched_min_area=args.unmatched_seg_min_area,
        unmatched_max_area=args.unmatched_seg_max_area,
        unmatched_max_items=args.unmatched_seg_max,
    )
    ranked = dedupe_candidates_nms(
        sorted(candidates, key=lambda item: item.get("confidence", 0.0), reverse=True),
        iou_threshold=0.55,
    )
    detections = []
    dino_embeds_done = 0
    for cand in ranked:
        label = str(cand.get("label", "")).lower()
        if not should_track(label):
            continue
        bbox = cand.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x = round((float(bbox[0]) + float(bbox[2])) * 0.5, 3)
        y = round((float(bbox[1]) + float(bbox[3])) * 0.5, 3)
        hsv = hsv_embedding(frame, bbox, out_dim=args.dim)
        emb = hsv
        source = "hsv"
        dino = []
        if mode == "dino_sparse":
            interval = args.dino_bootstrap_update_every if sample_no <= args.bootstrap_frames else args.dino_update_every
            should_refresh = sample_no % max(interval, 1) == 0
            if should_refresh and dino_embeds_done < max(1, args.max_dino_embeds_per_update):
                try:
                    dino = encode_bbox(frame, bbox, out_dim=args.dim)
                    dino_embeds_done += 1
                except Exception:
                    dino = []
            if isinstance(dino, list) and any(abs(float(v)) > 1e-8 for v in dino):
                emb = dino
                source = "dino"
        detections.append({
            "label": label,
            "x": x,
            "y": y,
            "bbox": bbox,
            "confidence": round(float(cand.get("confidence", 0.0) or 0.0), 3),
            "embedding": emb,
            "embedding_source": source,
            "hsv_embedding": hsv,
            "dino_embedding": dino if source == "dino" else [],
            "mask_polygon": cand.get("mask_polygon"),
            "segmentation_source": cand.get("segmentation_source", "bbox"),
        })
        if len(detections) >= args.max_objects:
            break
    return detections


def run_mode(args, mode: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_path = Path(args.yolo_model)
    if not yolo_path.is_file():
        yolo_path = Path(__file__).parent / "models" / "yolov8n.pt"
    yolo = YOLO(str(yolo_path)).to(device)
    seg_model = None
    if args.enable_segmentation:
        seg_path = Path({
            "yolo-seg": args.yolo_seg_model,
            "fastsam-s": args.fastsam_s_model,
            "fastsam-x": args.fastsam_x_model,
        }[args.segmentation_backend])
        if not seg_path.is_file():
            fallback = Path(__file__).resolve().parent.parent / seg_path.name
            if fallback.is_file():
                seg_path = fallback
        if seg_path.is_file():
            seg_model = YOLO(str(seg_path)).to(device)

    cap = cv2.VideoCapture(str(Path(args.video)))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {args.video}")

    state = WorldState()
    previous = []
    id_switch_proxy = 0
    continuity_checks = 0
    frames_used = 0
    source_counts = {"hsv": 0, "dino": 0}

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.frame_stride > 1 and (frame_idx - 1) % args.frame_stride != 0:
                continue
            frames_used += 1
            if args.max_frames > 0 and frames_used > args.max_frames:
                break

            detections = build_detections(frame, yolo, seg_model, device, args, mode, frames_used)
            state.update(detections)
            exported = state.export_objects()
            curr = []
            for det in detections:
                best = None
                best_iou = 0.0
                for obj in exported:
                    if obj.get("label") != det.get("label"):
                        continue
                    iou = bbox_iou(obj.get("bbox", [0, 0, 0, 0]), det.get("bbox", [0, 0, 0, 0]))
                    if iou > best_iou:
                        best_iou = iou
                        best = obj
                if best is None:
                    continue
                source = str(best.get("embedding_source", det.get("embedding_source", "unknown")))
                if source in source_counts:
                    source_counts[source] += 1
                curr.append({"label": det.get("label"), "bbox": det.get("bbox"), "id": best.get("id")})

            for prev in previous:
                matches = [
                    cur for cur in curr
                    if cur["label"] == prev["label"] and bbox_iou(cur["bbox"], prev["bbox"]) >= args.switch_iou
                ]
                if matches:
                    continuity_checks += 1
                    if matches[0]["id"] != prev["id"]:
                        id_switch_proxy += 1
            previous = curr
    finally:
        cap.release()

    objects = list(state.objects.values())
    hit_counts = [int(o.get("observation_count", 0) or 0) for o in objects]
    label_ids = {}
    for obj in objects:
        label_ids.setdefault(str(obj.get("label", "unknown")), set()).add(str(obj.get("id")))
    return {
        "mode": mode,
        "frames_used": frames_used,
        "tracks_created": int(state.next_id - 1),
        "active_tracks": len(state.export_objects()),
        "labels_with_multiple_ids": {
            label: len(ids) for label, ids in sorted(label_ids.items()) if len(ids) > 1
        },
        "id_switch_proxy": id_switch_proxy,
        "continuity_checks": continuity_checks,
        "id_switch_proxy_rate": round(id_switch_proxy / max(1, continuity_checks), 4),
        "mean_track_hits": round(float(np.mean(hit_counts)) if hit_counts else 0.0, 2),
        "short_track_ratio": round(sum(1 for v in hit_counts if v <= 2) / max(1, len(hit_counts)), 4),
        "embedding_sources_seen": source_counts,
    }


def summarize_delta(results: list[dict]) -> dict:
    by_mode = {item.get("mode"): item for item in results}
    hsv = by_mode.get("hsv", {})
    dino = by_mode.get("dino_sparse", {})
    labels = set(hsv.get("labels_with_multiple_ids", {}).keys()) | set(dino.get("labels_with_multiple_ids", {}).keys())
    duplicate_id_delta = {}
    for label in sorted(labels):
        before = int(hsv.get("labels_with_multiple_ids", {}).get(label, 1))
        after = int(dino.get("labels_with_multiple_ids", {}).get(label, 1))
        if before != after:
            duplicate_id_delta[label] = after - before
    return {
        "tracks_created_delta_dino_minus_hsv": int(dino.get("tracks_created", 0)) - int(hsv.get("tracks_created", 0)),
        "mean_track_hits_delta_dino_minus_hsv": round(
            float(dino.get("mean_track_hits", 0.0)) - float(hsv.get("mean_track_hits", 0.0)),
            3,
        ),
        "short_track_ratio_delta_dino_minus_hsv": round(
            float(dino.get("short_track_ratio", 0.0)) - float(hsv.get("short_track_ratio", 0.0)),
            4,
        ),
        "id_switch_proxy_rate_delta_dino_minus_hsv": round(
            float(dino.get("id_switch_proxy_rate", 0.0)) - float(hsv.get("id_switch_proxy_rate", 0.0)),
            4,
        ),
        "duplicate_id_delta_by_label_dino_minus_hsv": duplicate_id_delta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="public/scene_hand.mp4")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--bootstrap-frames", type=int, default=600)
    parser.add_argument("--dino-bootstrap-update-every", type=int, default=4)
    parser.add_argument("--dino-update-every", type=int, default=16)
    parser.add_argument("--max-dino-embeds-per-update", type=int, default=3)
    parser.add_argument("--switch-iou", type=float, default=0.35)
    parser.add_argument("--enable-segmentation", action="store_true")
    parser.add_argument("--segmentation-backend", default="yolo-seg", choices=["yolo-seg", "fastsam-s", "fastsam-x"])
    parser.add_argument("--add-unmatched-seg-objects", action="store_true")
    parser.add_argument("--unmatched-seg-min-confidence", type=float, default=0.15)
    parser.add_argument("--unmatched-seg-min-area", type=float, default=0.004)
    parser.add_argument("--unmatched-seg-max-area", type=float, default=0.25)
    parser.add_argument("--unmatched-seg-max", type=int, default=8)
    parser.add_argument("--yolo-model", default="world_model/models/yolov8n.pt")
    parser.add_argument("--yolo-seg-model", default="world_model/models/yolov8n-seg.pt")
    parser.add_argument("--fastsam-s-model", default="world_model/models/FastSAM-s.pt")
    parser.add_argument("--fastsam-x-model", default="world_model/models/FastSAM-x.pt")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    results = [run_mode(args, "hsv"), run_mode(args, "dino_sparse")]
    report = {"results": results, "summary": summarize_delta(results)}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
