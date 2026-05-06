from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "ok"
    return "warn" if warn else "missing"


def build_report(args) -> dict:
    static = _load(args.static_report)
    interaction = _load(args.interaction_report)
    tracking = _load(args.tracking_report)
    labels_seen = interaction.get("labels_seen", {}) or {}
    contact_labels = interaction.get("contact_labels", {}) or {}
    tracking_summary = tracking.get("summary", {}) or {}

    movable_labels = {
        "cup": int(labels_seen.get("cup", 0) or 0),
        "apple": int(labels_seen.get("apple", 0) or 0),
        "banana": int(labels_seen.get("banana", 0) or 0),
        "bottle": int(labels_seen.get("bottle", 0) or 0),
    }
    static_locked = int(static.get("static_targets_locked", 0) or interaction.get("static_targets_locked", 0) or 0)
    pick_place_count = int(interaction.get("pick_place_count", 0) or 0)
    contact_frames = int(interaction.get("contact_frames", 0) or 0)
    strict_touch_frames = int(interaction.get("strict_touch_frames", 0) or 0)
    hand_ratio = float(interaction.get("hand_frame_ratio", 0.0) or 0.0)
    dino_track_delta = float(tracking_summary.get("mean_track_hits_delta_dino_minus_hsv", 0.0) or 0.0)

    checklist = {
        "colmap_static_bootstrap": {
            "status": _status(static_locked >= args.min_static_targets, warn=static_locked > 0),
            "evidence": {
                "static_targets_locked": static_locked,
                "required": int(args.min_static_targets),
            },
        },
        "movable_objects_visible": {
            "status": _status(movable_labels["cup"] > 0 and (movable_labels["apple"] > 0 or movable_labels["banana"] > 0), warn=True),
            "evidence": movable_labels,
        },
        "named_targets": {
            "status": "warn",
            "evidence": {
                "current": "placement targets are locked mostly as unknown_seg; LLM relabeling is available but not yet backend-persistent",
            },
        },
        "right_hand_tracking": {
            "status": _status(hand_ratio >= args.min_hand_ratio, warn=hand_ratio > 0.25),
            "evidence": {
                "hand_frame_ratio": round(hand_ratio, 4),
                "required": float(args.min_hand_ratio),
            },
        },
        "geometry_contact_release": {
            "status": _status(contact_frames >= args.min_contact_frames and pick_place_count >= args.min_pick_place, warn=contact_frames > 0),
            "evidence": {
                "contact_frames": contact_frames,
                "strict_touch_frames": strict_touch_frames,
                "pick_place_count": pick_place_count,
                "contact_labels": contact_labels,
                "contact_distance_mean_m": interaction.get("contact_distance_mean_m"),
                "contact_distance_p90_m": interaction.get("contact_distance_p90_m"),
            },
        },
        "dino_tracking_gain": {
            "status": _status(dino_track_delta > 0.0, warn=True),
            "evidence": tracking_summary,
        },
        "jepa_future_prediction": {
            "status": "warn",
            "evidence": {
                "current": "temporal head is loaded and emits contact/place/motion predictions; training remains pseudo-labeled and early",
            },
        },
        "wow_visuals": {
            "status": "ok",
            "evidence": {
                "current": "UI renders COLMAP-centered maps, hand 3D, contact highlights, grab/place attention heatmaps",
            },
        },
    }

    missing = [name for name, item in checklist.items() if item.get("status") != "ok"]
    return {
        "inputs": {
            "static_report": args.static_report,
            "interaction_report": args.interaction_report,
            "tracking_report": args.tracking_report,
        },
        "overall_status": "ok" if not missing else "needs_attention",
        "needs_attention": missing,
        "checklist": checklist,
        "recommendations": [
            "Persist LLM-refined labels for static targets so coasters/dish are named, not only unknown_seg.",
            "Capture a new one-take video with two cups, two fruits, two coasters, one dish, plus one open-vocab object.",
            "Train the future dynamics model from interaction_capture.jsonl once new recordings exist.",
            "Keep geometry contact/release as teacher labels and use JEPA/DINO for prediction and track identity.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-report", default="world_model/data/static_targets_validation_static_segment_dense.json")
    parser.add_argument("--interaction-report", default="world_model/data/interaction_validation_right_hand_current.json")
    parser.add_argument("--tracking-report", default="world_model/data/tracking_embeddings_ab_latest.json")
    parser.add_argument("--output", default="world_model/data/demo_readiness_report.json")
    parser.add_argument("--min-static-targets", type=int, default=4)
    parser.add_argument("--min-hand-ratio", type=float, default=0.40)
    parser.add_argument("--min-contact-frames", type=int, default=12)
    parser.add_argument("--min-pick-place", type=int, default=2)
    args = parser.parse_args()

    report = build_report(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
