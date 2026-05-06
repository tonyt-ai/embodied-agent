from __future__ import annotations

import argparse
import json
from pathlib import Path


def _delay(value, expected):
    if value is None:
        return None
    return round(float(value) - float(expected), 3)


def _best_correct_drop(window: dict):
    target = str(window.get("target", "")).strip().lower()
    expected = float(window.get("release_s", window.get("grab_start_s", 0.0)) or 0.0)
    best = None
    best_abs_delay = None
    for event in window.get("pick_place_events", []) or []:
        if str(event.get("target_label", "")).strip().lower() != target:
            continue
        if not bool(event.get("moved", False)):
            continue
        t = event.get("video_time_s")
        if t is None:
            continue
        abs_delay = abs(float(t) - expected)
        if best is None or abs_delay < best_abs_delay:
            best = float(t)
            best_abs_delay = abs_delay
    return round(best, 3) if best is not None else None


def summarize(path: Path, grab_tolerance_s: float, drop_tolerance_s: float):
    data = json.loads(path.read_text(encoding="utf-8"))
    windows = data.get("timeline_validation", [])
    rows = []
    for idx, w in enumerate(windows, 1):
        expected_grab = float(w.get("grab_start_s", 0.0) or 0.0)
        expected_drop = float(w.get("release_s", expected_grab) or expected_grab)
        first_contact = w.get("first_contact_s")
        first_temporal_contact = w.get("first_temporal_contact_s")
        first_temporal_place = w.get("first_temporal_place_s")
        best_correct_drop = _best_correct_drop(w)
        row = {
            "index": idx,
            "id": w.get("id"),
            "object": w.get("object"),
            "source": w.get("source"),
            "target": w.get("target"),
            "expected_grab_s": round(expected_grab, 3),
            "expected_drop_s": round(expected_drop, 3),
            "first_geometry_contact_s": first_contact,
            "geometry_contact_delay_s": _delay(first_contact, expected_grab),
            "first_temporal_contact_s": first_temporal_contact,
            "temporal_contact_delay_s": _delay(first_temporal_contact, expected_grab),
            "first_temporal_place_s": first_temporal_place,
            "temporal_place_lead_to_drop_s": (
                round(expected_drop - float(first_temporal_place), 3)
                if first_temporal_place is not None else None
            ),
            "best_correct_drop_s": best_correct_drop,
            "correct_drop_delay_s": _delay(best_correct_drop, expected_drop),
            "geometry_grab_detected": bool(w.get("geometry_grab_detected", False)),
            "geometry_transfer_detected": bool(w.get("geometry_transfer_detected", False)),
            "labels_seen": w.get("labels_seen", {}),
            "contact_labels": w.get("contact_labels", {}),
            "place_target_labels": w.get("place_target_labels", {}),
        }
        row["grab_timing_ok"] = (
            row["temporal_contact_delay_s"] is not None
            and row["temporal_contact_delay_s"] <= grab_tolerance_s
        ) or (
            row["geometry_contact_delay_s"] is not None
            and row["geometry_contact_delay_s"] <= grab_tolerance_s
        )
        row["drop_timing_ok"] = (
            row["correct_drop_delay_s"] is not None
            and abs(row["correct_drop_delay_s"]) <= drop_tolerance_s
        )
        row["prediction_before_drop"] = (
            row["temporal_place_lead_to_drop_s"] is not None
            and row["temporal_place_lead_to_drop_s"] >= 0.0
        )
        rows.append(row)

    def count(key):
        return sum(1 for row in rows if bool(row.get(key)))

    summary = {
        "source": str(path),
        "windows": len(rows),
        "geometry_grabs": count("geometry_grab_detected"),
        "geometry_transfers": count("geometry_transfer_detected"),
        "grab_timing_ok": count("grab_timing_ok"),
        "drop_timing_ok": count("drop_timing_ok"),
        "prediction_before_drop": count("prediction_before_drop"),
        "grab_tolerance_s": grab_tolerance_s,
        "drop_tolerance_s": drop_tolerance_s,
        "rows": rows,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", default="world_model/data/interaction_validation_sophie_timeline_stride8.json")
    parser.add_argument("--out", default="world_model/data/sophie_timing_report_latest.json")
    parser.add_argument("--grab-tolerance-s", type=float, default=2.0)
    parser.add_argument("--drop-tolerance-s", type=float, default=2.0)
    args = parser.parse_args()

    report = summarize(Path(args.timeline), args.grab_tolerance_s, args.drop_tolerance_s)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    for row in report["rows"]:
        print(
            f"{row['index']:02d} {row['id']}: "
            f"grab geom {row['geometry_contact_delay_s']}s, "
            f"temporal {row['temporal_contact_delay_s']}s, "
            f"place lead {row['temporal_place_lead_to_drop_s']}s, "
            f"drop {row['correct_drop_delay_s']}s, "
            f"ok(grab/drop/pred)={row['grab_timing_ok']}/{row['drop_timing_ok']}/{row['prediction_before_drop']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
