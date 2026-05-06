"""Extract detector crops in the UI bridge payload format."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from perception_candidates import build_semantic_candidates


def _resolve_model(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    root = Path(__file__).resolve().parent.parent / p.name
    return root if root.is_file() else p


def main():
    parser = argparse.ArgumentParser(description="Extract object crops for Gemini label bridge testing.")
    parser.add_argument("--video", default="public/scene_hand.mp4")
    parser.add_argument("--frame", type=int, default=700)
    parser.add_argument("--detector-model", default="world_model/models/yolov8n.pt")
    parser.add_argument("--segmentation-model", default="world_model/models/yolov8n-seg.pt")
    parser.add_argument("--out-json", default="world_model/data/gemini_label_crops.json")
    parser.add_argument("--out-debug-dir", default="world_model/data/gemini_label_crops")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(Path(args.video)))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(args.frame)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {args.frame}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(str(_resolve_model(args.detector_model))).to(device)
    seg_path = _resolve_model(args.segmentation_model)
    segmenter = YOLO(str(seg_path)).to(device) if seg_path.is_file() else None
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

    h, w = frame.shape[:2]
    out = []
    debug_dir = Path(args.out_debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    for idx, c in enumerate(candidates[:6]):
        bbox = c.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) >= 4):
            continue
        x1 = int(max(0, min(w - 1, round(float(bbox[0]) * w))))
        y1 = int(max(0, min(h - 1, round(float(bbox[1]) * h))))
        x2 = int(max(0, min(w, round(float(bbox[2]) * w))))
        y2 = int(max(0, min(h, round(float(bbox[3]) * h))))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        ok_enc, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok_enc:
            continue
        label = str(c.get("label", "unknown")).lower()
        obj_id = f"crop_{idx}_{label.replace(' ', '_')}"
        (debug_dir / f"{obj_id}.jpg").write_bytes(enc.tobytes())
        out.append({
            "id": obj_id,
            "label_hint": label,
            "mime_type": "image/jpeg",
            "image_base64": base64.b64encode(enc.tobytes()).decode("ascii"),
        })

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out_path), "objects": len(out)}, indent=2))


if __name__ == "__main__":
    main()
