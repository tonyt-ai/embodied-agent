from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for JEPA grab evaluation") from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "world_model"))

from temporal_interaction_head import TemporalInteractionHead  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in rows if isinstance(row, dict) and isinstance(row.get("feat"), list)]
    if not rows:
        raise RuntimeError(f"No feature rows found in {path}")
    return rows


def predict_temporal(rows: list[dict], model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["feat"] for row in rows], dtype=np.float32)
    model = TemporalInteractionHead(in_dim=x.shape[1])
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"), strict=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x))
        contact = torch.sigmoid(out["contact_logit"]).cpu().numpy().reshape(-1)
        target = torch.sigmoid(out.get("target_tray_logit", out["placement_logit"])).cpu().numpy().reshape(-1)
        return contact, target


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
    if 0.02 <= cx <= 0.47 and 0.12 <= cy <= 0.86:
        return "mat"
    if 0.48 <= cx <= 0.99 and 0.20 <= cy <= 0.98:
        return "tray"
    return ""


def decode_events(
    rows: list[dict],
    probs: np.ndarray,
    target_probs: np.ndarray,
    threshold: float,
    refractory_s: float,
) -> list[dict]:
    events = []
    last_t = -1e9
    for row, prob, target_prob in zip(rows, probs, target_probs):
        t = float(row.get("video_time_s", row.get("frame", 0.0)) or 0.0)
        if float(prob) < threshold or t - last_t < refractory_s:
            continue
        label = visual_label_from_row(row)
        learned_target = "tray" if float(target_prob) >= 0.5 else "mat"
        target = target_from_bbox(row) or str(row.get("future_episode_target") or row.get("episode_target") or "") or learned_target
        events.append(
            {
                "video_time_s": round(t, 3),
                "prob": round(float(prob), 4),
                "label": label,
                "target": target,
                "target_tray_prob": round(float(target_prob), 4),
                "object_id": row.get("object_id"),
            }
        )
        last_t = t
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind JEPA grab-start episode count from temporal rows.")
    parser.add_argument("--rows", default="world_model/data/temporal_head_train_rows_sophie.json")
    parser.add_argument("--model", default="world_model/models/temporal_interaction_head_sophie.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--refractory-s", type=float, default=10.0)
    parser.add_argument("--output", default="world_model/data/blind_jepa_grabs_sophie_latest.json")
    args = parser.parse_args()

    rows = load_rows(REPO_ROOT / args.rows)
    probs, target_probs = predict_temporal(rows, REPO_ROOT / args.model)
    events = decode_events(rows, probs, target_probs, threshold=args.threshold, refractory_s=args.refractory_s)
    report = {
        "rows": len(rows),
        "model": args.model,
        "threshold": float(args.threshold),
        "refractory_s": float(args.refractory_s),
        "grab_count": len(events),
        "events": events,
    }
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
