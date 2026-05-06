"""Build horizon-labeled interaction dynamics rows from auto-capture logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _vec3(value):
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        arr = np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _by_id(items):
    out = {}
    for item in items or []:
        item_id = item.get("id")
        if item_id:
            out[str(item_id)] = item
    return out


def _pad(values, n):
    out = [0.0] * n
    if not isinstance(values, list):
        return out
    for i, value in enumerate(values[:n]):
        try:
            out[i] = float(value)
        except (TypeError, ValueError):
            out[i] = 0.0
    return out


def load_records(path: Path):
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda item: int(item.get("frame", 0)))
    return records


def build_rows(records, horizon: int, emb_dim: int):
    rows = []
    for idx, rec in enumerate(records):
        fut = records[min(len(records) - 1, idx + max(1, horizon))]
        fut_objects = _by_id(fut.get("objects", []))
        fut_contact_ids = {
            str(item)
            for item in (fut.get("teacher_labels", {}) or {}).get("contact_object_ids", [])
            if item
        }
        release_events = [
            ev for ev in fut.get("new_events", [])
            if str(ev.get("event", "")) in {"pick_place", "contact_end"}
        ]
        release_target_ids = {str(ev.get("object_id")) for ev in release_events if ev.get("object_id")}
        objects = _by_id(rec.get("objects", []))
        hands = _by_id(rec.get("hands", []))
        if not hands:
            continue
        hand = next(iter(hands.values()))
        hand_pos = _vec3(hand.get("center_3d"))
        if hand_pos is None:
            continue
        hand_vel = _vec3(hand.get("velocity_3d"))
        if hand_vel is None:
            hand_vel = np.zeros(3, dtype=np.float32)
        hand_emb = _pad(hand.get("jepa_temporal_embedding", []), emb_dim)

        for obj_id, obj in objects.items():
            obj_pos = _vec3(obj.get("position_3d"))
            if obj_pos is None:
                continue
            obj_vel = _vec3(obj.get("velocity_3d"))
            if obj_vel is None:
                obj_vel = np.zeros(3, dtype=np.float32)
            obj_emb = _pad(obj.get("jepa_temporal_embedding", []), emb_dim)
            fut_obj = fut_objects.get(obj_id)
            fut_pos = _vec3(fut_obj.get("position_3d")) if fut_obj else None
            motion = (fut_pos - obj_pos).tolist() if fut_pos is not None else [0.0, 0.0, 0.0]
            rel = obj_pos - hand_pos
            dist = float(np.linalg.norm(rel))
            row = {
                "frame": int(rec.get("frame", 0)),
                "future_frame": int(fut.get("frame", rec.get("frame", 0))),
                "horizon_records": int(min(len(records) - 1, idx + max(1, horizon)) - idx),
                "hand_id": hand.get("id"),
                "object_id": obj_id,
                "object_label": obj.get("label", "unknown"),
                "feature": [
                    *hand_pos.tolist(),
                    *hand_vel.tolist(),
                    *obj_pos.tolist(),
                    *obj_vel.tolist(),
                    *rel.tolist(),
                    dist,
                    float(hand.get("confidence", 0.0) or 0.0),
                    float(obj.get("confidence", 0.0) or 0.0),
                    1.0 if hand.get("predicted") else 0.0,
                    *hand_emb,
                    *obj_emb,
                ],
                "target": {
                    "future_contact": 1.0 if obj_id in fut_contact_ids else 0.0,
                    "future_release": 1.0 if obj_id in release_target_ids else 0.0,
                    "future_motion_delta": [float(v) for v in motion],
                    "future_moved": 1.0 if float(np.linalg.norm(np.asarray(motion, dtype=np.float32))) > 0.03 else 0.0,
                },
                "teacher_quality": rec.get("quality", {}),
            }
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="world_model/data/interaction_capture.jsonl")
    parser.add_argument("--output", default="world_model/data/interaction_dynamics_rows.json")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--emb-dim", type=int, default=64)
    args = parser.parse_args()

    records = load_records(Path(args.input))
    rows = build_rows(records, horizon=args.horizon, emb_dim=args.emb_dim)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": args.input, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "rows": len(rows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
