from __future__ import annotations

import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "world_model"))

from interaction_guidance import build_interaction_guidance


class MockState:
    def __init__(self):
        self.objects = {}
        self.static_targets = {}
        self.hand_object_interactions = []
        self.hand_contact_events = []
        self.manipulation_events = []


def bridge_payload(text: str | None) -> dict | None:
    if not text:
        return None
    return {"type": "world_model_explanation", "text": text}


def load_static_targets(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    targets = {}
    for item in data.get("targets", []):
        tid = str(item.get("id") or "")
        if not tid:
            continue
        targets[tid] = dict(item)
    return targets


def main():
    validation_path = ROOT / "world_model" / "data" / "interaction_validation_right_hand_current.json"
    static_path = ROOT / "world_model" / "data" / "static_targets_validation_static_segment_dense.json"
    out_path = ROOT / "world_model" / "data" / "interaction_guidance_replay_latest.json"

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    state = MockState()
    state.static_targets = load_static_targets(static_path)

    state.objects = {
        "obj_cup": {"id": "obj_cup", "label": "cup"},
        "obj_apple": {"id": "obj_apple", "label": "apple"},
    }
    # Pretend one Gemini refinement has already named the top static target as dish.
    if "target_22" in state.static_targets:
        state.static_targets["target_22"]["label"] = "dish"

    scenarios = []

    state.hand_object_interactions = [{
        "hand_id": "hand_right",
        "nearest_object_id": "obj_apple",
        "nearest_object_label": "apple",
        "distance_m": 0.085,
        "is_contacting": False,
        "intent_grab_candidates": [{
            "object_id": "obj_apple",
            "label": "apple",
            "pred_contact_prob": 0.76,
        }],
        "intent_place_candidates": [],
    }]
    state.hand_contact_events = []
    state.manipulation_events = []
    scenarios.append({"name": "approach_apple", "guidance": build_interaction_guidance(state)})

    state.hand_object_interactions = [{
        "hand_id": "hand_right",
        "nearest_object_id": "obj_cup",
        "nearest_object_label": "cup",
        "distance_m": 0.032,
        "is_contacting": True,
        "intent_grab_candidates": [{
            "object_id": "obj_cup",
            "label": "cup",
            "pred_contact_prob": 0.88,
        }],
        "intent_place_candidates": [{
            "object_id": "target_22",
            "label": "unknown_seg",
            "pred_placement_prob": 0.82,
        }],
    }]
    state.hand_contact_events = [{"event": "contact_start", "object_id": "obj_cup", "label": "cup"}]
    state.manipulation_events = []
    scenarios.append({"name": "grab_cup_predict_dish", "guidance": build_interaction_guidance(state)})

    state.objects = {
        "obj_bottle": {"id": "obj_bottle", "label": "baby bottle", "position_3d": [0.0, 0.0, 0.0]},
    }
    state.static_targets = {
        "target_mat": {"id": "target_mat", "label": "dish", "locked": True, "position_3d": [0.02, 0.0, 0.0]},
        "target_coaster": {"id": "target_coaster", "label": "coaster", "locked": True, "position_3d": [0.6, 0.0, 0.0]},
    }
    state.hand_object_interactions = [{
        "hand_id": "hand_right",
        "nearest_object_id": "obj_bottle",
        "nearest_object_label": "baby bottle",
        "distance_m": 0.032,
        "is_contacting": True,
        "intent_grab_candidates": [],
        "intent_place_candidates": [],
    }]
    state.hand_contact_events = [{"event": "contact_start", "object_id": "obj_bottle", "label": "baby bottle"}]
    state.manipulation_events = []
    scenarios.append({"name": "grab_baby_bottle_from_mat_predict_coaster", "guidance": build_interaction_guidance(state)})

    state.objects["obj_bottle"]["position_3d"] = [0.6, 0.0, 0.0]
    scenarios.append({"name": "grab_baby_bottle_from_coaster_predict_mat", "guidance": build_interaction_guidance(state)})

    state.objects = {
        "obj_bottle": {"id": "obj_bottle", "label": "baby bottle", "position_3d": [0.6, 0.0, 0.0]},
        "obj_giraffe": {"id": "obj_giraffe", "label": "toy giraffe", "position_3d": [0.02, 0.0, 0.0]},
    }
    state.static_targets = {
        "target_mat": {"id": "target_mat", "label": "mat", "locked": True, "position_3d": [0.02, 0.0, 0.0]},
        "target_tray": {"id": "target_tray", "label": "tray", "locked": True, "position_3d": [0.6, 0.0, 0.0]},
    }
    state.hand_object_interactions = [{
        "hand_id": "hand_right",
        "nearest_object_id": "obj_bottle",
        "nearest_object_label": "baby bottle",
        "held_object_id": "obj_giraffe",
        "held_object_label": "toy giraffe",
        "learned_object_id": "obj_giraffe",
        "learned_target_label": "tray",
        "learned_is_held": True,
        "distance_m": 0.04,
        "is_contacting": True,
        "intent_grab_candidates": [],
        "intent_place_candidates": [],
    }]
    state.hand_contact_events = [{"event": "contact_start", "object_id": "obj_giraffe", "label": "toy giraffe"}]
    state.manipulation_events = []
    owner_guidance = build_interaction_guidance(state)
    if owner_guidance.get("details", {}).get("object") != "toy giraffe":
        raise AssertionError(f"held owner regression: {owner_guidance}")
    scenarios.append({"name": "held_owner_beats_nearest_object", "guidance": owner_guidance})

    pick_place_events = validation.get("pick_place_events", [])
    if pick_place_events:
        state.hand_object_interactions = []
        state.hand_contact_events = [{"event": "contact_end", "object_id": pick_place_events[-1].get("object_id")}]
        state.objects[str(pick_place_events[-1].get("object_id"))] = {
            "id": str(pick_place_events[-1].get("object_id")),
            "label": pick_place_events[-1].get("label", "object"),
        }
        state.manipulation_events = [pick_place_events[-1]]
        scenarios.append({"name": "release_from_validation_last_event", "guidance": build_interaction_guidance(state)})

    report = {
        "source_validation": str(validation_path),
        "validation_summary": {
            "processed_frames": validation.get("processed_frames"),
            "contact_frames": validation.get("contact_frames"),
            "strict_touch_frames": validation.get("strict_touch_frames"),
            "contact_labels": validation.get("contact_labels"),
            "pick_place_count": validation.get("pick_place_count"),
        },
        "scenarios": [
            {
                **item,
                "bridge_payload": bridge_payload(item["guidance"].get("explanation")),
            }
            for item in scenarios
        ],
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
