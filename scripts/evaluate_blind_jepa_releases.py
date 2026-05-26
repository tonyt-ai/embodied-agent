import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for JEPA release evaluation") from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "world_model"))

from temporal_interaction_head import TemporalInteractionHead  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in rows if isinstance(row, dict) and isinstance(row.get("feat"), list)]
    if not rows:
        raise RuntimeError(f"No feature rows found in {path}")
    return rows


def predict(rows: list[dict], model_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([row["feat"] for row in rows], dtype=np.float32)
    model = TemporalInteractionHead(in_dim=x.shape[1])
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"), strict=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x))
        contact = torch.sigmoid(out["contact_logit"]).cpu().numpy().reshape(-1)
        release = torch.sigmoid(out.get("release_logit", out["placement_logit"])).cpu().numpy().reshape(-1)
        target = torch.sigmoid(out.get("target_tray_logit", out["placement_logit"])).cpu().numpy().reshape(-1)
    return contact, release, target


def canonical_label(label: str) -> str:
    text = str(label or "").strip().lower().replace("_", " ")
    if text in {"donut", "mouse", "toy", "giraffe", "sophie", "sophie giraffe"}:
        return "toy giraffe"
    if text in {"bottle", "cup", "mug"}:
        return "baby bottle"
    return text or "object"


def visual_label_from_row(row: dict) -> str:
    latent_label = canonical_label(row.get("visual_identity_label") or "")
    if latent_label in {"baby bottle", "toy giraffe"}:
        return latent_label
    label = canonical_label(row.get("object_label") or row.get("label") or "")
    raw = str(row.get("object_label") or row.get("object_raw_label") or "").strip().lower().replace("_", " ")
    if label == "toy giraffe" or raw in {"donut", "mouse", "toy", "toy giraffe"}:
        return "toy giraffe"
    bbox = row.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) >= 4):
        return label
    try:
        bw = max(0.0, float(bbox[2]) - float(bbox[0]))
        bh = max(0.0, float(bbox[3]) - float(bbox[1]))
        area = bw * bh
        aspect = bw / max(bh, 1e-6)
    except (TypeError, ValueError):
        return label
    if label == "baby bottle" and area >= 0.006 and aspect >= 0.72:
        return "toy giraffe"
    if label in {"baby bottle", "bottle", "object"} and aspect <= 0.68:
        return "baby bottle"
    return label


def target_from_bbox(row: dict) -> str:
    bbox = row.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) >= 4):
        return ""
    try:
        cx = (float(bbox[0]) + float(bbox[2])) * 0.5
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5
    except (TypeError, ValueError):
        return ""
    regions = [
        ("tray", 0.31, 0.49, 0.36, 0.43),
        ("mat", 0.73, 0.62, 0.34, 0.38),
    ]
    score, label = min(
        (((cx - rx0) / max(rw, 1e-6)) ** 2 + ((cy - ry0) / max(rh, 1e-6)) ** 2, label)
        for label, rx0, ry0, rw, rh in regions
    )
    return label if score <= 1.85 else ""


def canonical_support(label: str) -> str:
    text = str(label or "").strip().lower().replace("_", " ")
    if text in {"black mat", "table mat", "placemat", "dish", "plate"}:
        return "mat"
    if text in {"plastic tray", "white tray"}:
        return "tray"
    return text


def grounded_transfer_target(row: dict) -> str:
    source = canonical_support(row.get("source_support_label") or row.get("support_target_label") or "")
    if source == "mat":
        return "tray"
    if source == "tray":
        return "mat"
    return ""


def decoded_target_from_row(row: dict, target_prob: float, target_tray_threshold: float) -> str:
    grounded = grounded_transfer_target(row)
    if grounded:
        return grounded
    return "tray" if float(target_prob) >= float(target_tray_threshold) else "mat"


def _row_time(row: dict) -> float:
    return float(row.get("video_time_s", row.get("frame", 0.0)) or 0.0)


def _event_from_row(row: dict, prob: float, target_prob: float, target_tray_threshold: float = 0.45) -> dict:
    label = visual_label_from_row(row)
    learned_target = decoded_target_from_row(row, float(target_prob), float(target_tray_threshold))
    teacher_target = str(row.get("future_episode_target") or row.get("episode_target") or target_from_bbox(row) or "")
    return {
        "video_time_s": round(_row_time(row), 3),
        "prob": round(float(prob), 4),
        "label": label,
        "target": learned_target,
        "teacher_target": teacher_target,
        "target_tray_prob": round(float(target_prob), 4),
        "object_id": row.get("object_id"),
    }


def decode_threshold_events(rows: list[dict], probs: np.ndarray, target_probs: np.ndarray, threshold: float, refractory_s: float, target_tray_threshold: float) -> list[dict]:
    events = []
    last_t = -1e9
    for row, prob, target_prob in zip(rows, probs, target_probs):
        t = _row_time(row)
        if float(prob) < threshold or t - last_t < refractory_s:
            continue
        events.append(_event_from_row(row, float(prob), float(target_prob), target_tray_threshold=target_tray_threshold))
        last_t = t
    return events


def decode_stateful_events(
    rows: list[dict],
    contact_probs: np.ndarray,
    release_probs: np.ndarray,
    target_probs: np.ndarray,
    contact_threshold: float,
    release_threshold: float,
    grab_refractory_s: float,
    min_hold_s: float,
    pre_next_grab_margin_s: float,
    target_tray_threshold: float,
    peak_lookahead_s: float,
) -> list[dict]:
    """Decode one learned release inside each learned hold episode.

    Release is not an independent visual class here: it is the closing edge of a
    hand-object episode. The contact head opens the episode, the release head
    chooses the most plausible closing moment before the next learned grab.
    """
    from evaluate_blind_jepa_grabs import decode_events as decode_grab_events

    grabs = decode_grab_events(
        rows,
        contact_probs,
        target_probs,
        threshold=contact_threshold,
        refractory_s=grab_refractory_s,
        target_tray_threshold=target_tray_threshold,
        peak_lookahead_s=peak_lookahead_s,
    )
    times = np.asarray([_row_time(row) for row in rows], dtype=np.float32)
    events = []
    for i, grab in enumerate(grabs):
        start_t = float(grab["video_time_s"]) + float(min_hold_s)
        end_t = float(grabs[i + 1]["video_time_s"]) - float(pre_next_grab_margin_s) if i + 1 < len(grabs) else float(times[-1]) + 1.0
        if end_t <= start_t:
            continue
        idx = np.where((times >= start_t) & (times <= end_t))[0]
        if len(idx) == 0:
            continue
        grab_object_id = str(grab.get("object_id") or "")
        grab_label = canonical_label(str(grab.get("label") or ""))
        same_object_idx = [
            int(j) for j in idx
            if grab_object_id and str(rows[int(j)].get("object_id") or "") == grab_object_id
        ]
        same_label_idx = [
            int(j) for j in idx
            if canonical_label(str(rows[int(j)].get("visual_identity_label") or rows[int(j)].get("object_label") or "")) == grab_label
        ]
        if same_object_idx:
            idx = np.asarray(same_object_idx, dtype=np.int64)
        elif same_label_idx:
            idx = np.asarray(same_label_idx, dtype=np.int64)
        above = idx[release_probs[idx] >= float(release_threshold)]
        if len(above) > 0:
            # Use the first confident release so the event is timely, not merely
            # the largest later peak in the same held interval.
            chosen = int(above[0])
        else:
            chosen = int(idx[np.argmax(release_probs[idx])])
        event = _event_from_row(rows[chosen], float(release_probs[chosen]), float(target_probs[chosen]), target_tray_threshold=target_tray_threshold)
        event["grab_time_s"] = round(float(grab["video_time_s"]), 3)
        event["grab_prob"] = grab["prob"]
        event["fallback_peak"] = bool(len(above) == 0)
        events.append(event)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind JEPA release-event count from temporal rows.")
    parser.add_argument("--rows", default="world_model/data/temporal_head_train_rows_sophie.json")
    parser.add_argument("--model", default="world_model/models/temporal_interaction_head_sophie.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-tray-threshold", type=float, default=0.45)
    parser.add_argument("--peak-lookahead-s", type=float, default=2.0)
    parser.add_argument("--refractory-s", type=float, default=5.0)
    parser.add_argument("--mode", choices=["stateful", "threshold"], default="stateful")
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--min-hold-s", type=float, default=2.0)
    parser.add_argument("--pre-next-grab-margin-s", type=float, default=0.75)
    parser.add_argument("--output", default="world_model/data/blind_jepa_releases_sophie_latest.json")
    args = parser.parse_args()

    rows = load_rows(REPO_ROOT / args.rows)
    contact_probs, release_probs, target_probs = predict(rows, REPO_ROOT / args.model)
    if args.mode == "stateful":
        events = decode_stateful_events(
            rows,
            contact_probs,
            release_probs,
            target_probs,
            contact_threshold=args.contact_threshold,
            release_threshold=args.threshold,
            grab_refractory_s=args.refractory_s,
            min_hold_s=args.min_hold_s,
            pre_next_grab_margin_s=args.pre_next_grab_margin_s,
            target_tray_threshold=args.target_tray_threshold,
            peak_lookahead_s=args.peak_lookahead_s,
        )
    else:
        events = decode_threshold_events(rows, release_probs, target_probs, threshold=args.threshold, refractory_s=args.refractory_s, target_tray_threshold=args.target_tray_threshold)
    report = {
        "rows": len(rows),
        "model": args.model,
        "mode": args.mode,
        "threshold": float(args.threshold),
        "refractory_s": float(args.refractory_s),
        "contact_threshold": float(args.contact_threshold),
        "target_tray_threshold": float(args.target_tray_threshold),
        "peak_lookahead_s": float(args.peak_lookahead_s),
        "release_count": len(events),
        "events": events,
    }
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
