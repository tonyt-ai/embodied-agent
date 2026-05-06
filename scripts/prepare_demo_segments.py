from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def write_segment(src: Path, dst: Path, start_s: float, end_s: float | None, max_width: int = 0) -> dict:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total_frames / max(fps, 1e-6)
    start_frame = max(0, int(round(float(start_s) * fps)))
    end_frame = total_frames if end_s is None else min(total_frames, int(round(float(end_s) * fps)))
    if end_frame <= start_frame:
        raise RuntimeError(f"empty segment {dst}: start={start_s}s end={end_s}s")

    out_w, out_h = width, height
    if max_width and width > max_width:
        scale = float(max_width) / max(width, 1)
        out_w = int(round(width * scale))
        out_h = int(round(height * scale))
    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"unable to write {dst}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    idx = start_frame
    try:
        while idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if (out_w, out_h) != (width, height):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
            idx += 1
    finally:
        writer.release()
        cap.release()

    return {
        "path": str(dst),
        "start_s": float(start_s),
        "end_s": round(float(start_s + written / max(fps, 1e-6)), 3),
        "frames": int(written),
        "fps": round(fps, 3),
        "width": int(out_w),
        "height": int(out_h),
        "source_duration_s": round(float(duration), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="public/scene_sophie.mp4")
    parser.add_argument("--static-out", default="public/scene_sophie_static_30s.mp4")
    parser.add_argument("--identity-out", default="public/scene_sophie_identity_30s.mp4")
    parser.add_argument("--interaction-out", default="public/scene_sophie_interactions.mp4")
    parser.add_argument("--static-seconds", type=float, default=30.0)
    parser.add_argument(
        "--identity-seconds",
        type=float,
        default=0.0,
        help="End time for identity/bootstrap segment. Defaults to static-seconds.",
    )
    parser.add_argument(
        "--interaction-start-seconds",
        type=float,
        default=0.0,
        help="Start time for interaction segment. Defaults to identity-seconds/static-seconds.",
    )
    parser.add_argument("--max-width", type=int, default=0)
    parser.add_argument("--manifest", default="world_model/data/demo_segments_manifest_sophie.json")
    args = parser.parse_args()

    src = Path(args.source)
    static_end = max(0.1, float(args.static_seconds))
    identity_end = max(static_end, float(args.identity_seconds) if float(args.identity_seconds) > 0.0 else static_end)
    interaction_start = max(identity_end, float(args.interaction_start_seconds) if float(args.interaction_start_seconds) > 0.0 else identity_end)
    static = write_segment(src, Path(args.static_out), 0.0, static_end, max_width=args.max_width)
    identity = write_segment(src, Path(args.identity_out), 0.0, identity_end, max_width=args.max_width)
    interaction = write_segment(src, Path(args.interaction_out), interaction_start, None, max_width=args.max_width)
    manifest = {
        "source": str(src),
        "timing": {
            "static_bootstrap_seconds": float(static_end),
            "identity_bootstrap_seconds": float(identity_end),
            "interaction_start_seconds": float(interaction_start),
        },
        "static_bootstrap": static,
        "identity_bootstrap": identity,
        "interaction_learning": interaction,
        "recommended_use": {
            "static_bootstrap": "COLMAP sparse prior, depth scale anchors, static target bootstrapping",
            "identity_bootstrap": "object detection, DINO/LLM labels, object identity memories, optional static target refinement",
            "interaction_learning": "continuous hand-object dynamics, geometry-teacher labels, JEPA temporal training, prediction overlays",
        },
    }
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
