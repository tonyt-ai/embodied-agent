"""Grounded hand-object guidance text for avatar speech."""

from __future__ import annotations

import os
import time

from semantic_labels import normalize_label

MOVABLE_LABELS = {"cup", "mug", "bottle", "baby bottle", "toy giraffe", "apple", "banana", "orange", "fruit"}
SOPHIE_RAW_OBJECT_LABELS = {"mouse", "donut", "toy"}
TARGET_LABELS = {"dish", "coaster", "mat", "black mat", "table mat", "placemat", "tray", "plastic tray", "white tray"}
FRUIT_LABELS = {"apple", "banana", "orange", "fruit"}
CUP_LABELS = {"cup", "mug", "bottle", "baby bottle"}
BOTTLE_LABELS = {"bottle", "baby bottle", "toy giraffe"}
MAT_LABELS = {"mat", "black mat", "table mat", "placemat", "dish"}
TRAY_LABELS = {"tray", "plastic tray", "white tray"}


def pretty_label(label):
    text = normalize_label(label or "object")
    if os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() == "sophie" and text in SOPHIE_RAW_OBJECT_LABELS:
        # These are detector-class artifacts for the Sophie toy before VLM refinement.
        return "toy giraffe"
    scene_movables = {
        normalize_label(item)
        for item in os.environ.get("SCENE_MOVABLE_LABELS", "").split(",")
        if item.strip()
    }
    scene_forbidden = {
        normalize_label(item)
        for item in os.environ.get("SCENE_FORBIDDEN_LABELS", "").split(",")
        if item.strip()
    }
    if scene_movables and text in scene_forbidden:
        return "object"
    if text in {"sophie", "sophie giraffe", "sophie the giraffe", "giraffe", "giraffe toy", "rubber giraffe", "plush"}:
        return "toy giraffe"
    if text in {"plastic tray", "white tray", "serving tray"}:
        return "tray"
    if not text or text in {"unknown", "unknown seg", "unknown object"}:
        return "object"
    return text


def is_movable_label(label):
    text = pretty_label(label)
    return text in MOVABLE_LABELS or text in SOPHIE_RAW_OBJECT_LABELS


def is_target_label(label):
    return pretty_label(label) in TARGET_LABELS


def _as_vec3(values):
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return None
    try:
        xyz = [float(values[0]), float(values[1]), float(values[2])]
    except (TypeError, ValueError):
        return None
    if not all(v == v for v in xyz):
        return None
    return xyz


def _distance3(a, b):
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)) ** 0.5


def _target_label_alias(label, preferred=None):
    text = pretty_label(label)
    if text in {"black mat", "table mat", "placemat"}:
        return "mat"
    if text in {"plastic tray", "white tray", "serving tray"}:
        return "tray"
    if preferred == "mat" and text in {"dish", "plate", "platter", "coaster"}:
        # In the biberon scene the large static placement regions are table mats.
        return "mat"
    if preferred == "tray" and text in {"dish", "plate", "platter", "coaster"}:
        return "tray"
    return text


def nearest_static_target_label(world_state, object_id):
    obj = getattr(world_state, "objects", {}).get(str(object_id or ""))
    obj_pos = _as_vec3(obj.get("position_3d")) if obj else None
    if obj_pos is None:
        return None
    best_label = None
    best_dist = None
    for target in getattr(world_state, "static_targets", {}).values():
        if not bool(target.get("locked", False)):
            continue
        label = _target_label_alias(target.get("label", "target"))
        if label not in TARGET_LABELS and label not in MAT_LABELS:
            continue
        tpos = _as_vec3(target.get("position_3d"))
        if tpos is None:
            continue
        dist = _distance3(obj_pos, tpos)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def has_static_target_label(world_state, desired):
    wanted = _target_label_alias(desired, desired)
    for target in getattr(world_state, "static_targets", {}).values():
        if not bool(target.get("locked", False)):
            continue
        if _target_label_alias(target.get("label", "target"), wanted) == wanted:
            return True
    return False


def configured_transfer_targets():
    labels = [
        _target_label_alias(item.strip())
        for item in os.environ.get("DEMO_TRANSFER_TARGETS", "").split(",")
        if item.strip()
    ]
    return [label for label in labels if label in TARGET_LABELS or label in MAT_LABELS]


def default_target_for_object(label, world_state=None, object_id=None):
    obj = pretty_label(label)
    transfer_targets = configured_transfer_targets()
    if len(transfer_targets) >= 2 and obj in (BOTTLE_LABELS | SOPHIE_RAW_OBJECT_LABELS | CUP_LABELS):
        if world_state is not None:
            source = nearest_static_target_label(world_state, object_id)
            if source in transfer_targets:
                idx = transfer_targets.index(source)
                return transfer_targets[(idx + 1) % len(transfer_targets)]
        return next((target for target in transfer_targets if has_static_target_label(world_state, target)), transfer_targets[0])
    if obj in FRUIT_LABELS:
        return "dish"
    if (obj in BOTTLE_LABELS or obj in SOPHIE_RAW_OBJECT_LABELS) and world_state is not None:
        source = nearest_static_target_label(world_state, object_id)
        if source in {"coaster", "tray"}:
            return "mat"
        if source in MAT_LABELS:
            return "tray" if has_static_target_label(world_state, "tray") else "coaster"
        return "tray" if has_static_target_label(world_state, "tray") else "coaster"
    if obj in CUP_LABELS:
        return "coaster"
    return "target"


def learned_target_from_item(item):
    label = _target_label_alias(item.get("pred_target_label") or item.get("target_label") or "")
    if not is_target_label(label):
        return None
    try:
        motion_score = float(item.get("pred_target_motion_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        motion_score = 0.0
    if motion_score >= 0.15:
        return label
    try:
        prob = float(item.get("pred_target_tray_prob", item.get("target_tray_prob", 0.5)) or 0.5)
    except (TypeError, ValueError):
        prob = 0.5
    if prob >= 0.56 or prob <= 0.44:
        return label
    return None


def preferred_target_for_cue(label, world_state=None, object_id=None, *items):
    for item in items:
        if isinstance(item, dict):
            learned = learned_target_from_item(item)
            if learned:
                return learned
    return default_target_for_object(label, world_state, object_id)


def cue(label, state):
    obj = pretty_label(label)
    if state == "near":
        return f"{obj}: close."
    if state == "held":
        return f"{obj}: held."
    if state == "released":
        return f"{obj}: released."
    if state == "touched":
        return f"{obj}: contact."
    return f"{obj}."


def object_label_by_id(world_state, object_id, fallback="object"):
    if not object_id:
        return pretty_label(fallback)
    obj = getattr(world_state, "objects", {}).get(str(object_id))
    if obj is None:
        obj = getattr(world_state, "static_targets", {}).get(str(object_id))
    if obj is None:
        return pretty_label(fallback)
    label = pretty_label(obj.get("label", fallback))
    if label != "object":
        return label
    for key in ("vlm_label", "raw_label"):
        alt = pretty_label(obj.get(key, ""))
        if alt != "object":
            return alt
    return pretty_label(fallback)


def top_candidate(candidates, score_key):
    best = None
    best_score = -1.0
    for item in candidates or []:
        try:
            score = float(item.get(score_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if best is None or score > best_score:
            best = item
            best_score = score
    return best, best_score


def top_movable_candidate(world_state, candidates, score_key):
    best = None
    best_score = -1.0
    best_label = None
    for item in candidates or []:
        try:
            score = float(item.get(score_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        label = object_label_by_id(world_state, item.get("object_id"), item.get("label", "object"))
        if not is_movable_label(label):
            continue
        if best is None or score > best_score:
            best = item
            best_score = score
            best_label = label
    return best, best_score, best_label


def top_target_candidate(world_state, candidates, score_key, preferred=None):
    best = None
    best_score = -1.0
    best_label = None
    for item in candidates or []:
        try:
            score = float(item.get(score_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        label = _target_label_alias(object_label_by_id(world_state, item.get("object_id"), item.get("label", "target")), preferred)
        if not is_target_label(label):
            continue
        if best is None or score > best_score:
            best = item
            best_score = score
            best_label = pretty_label(label)
    return best, best_score, best_label


def event_age(now, event):
    if not event:
        return None
    try:
        return max(0.0, now - float(event.get("time", now)))
    except (TypeError, ValueError):
        return None


def build_interaction_guidance(world_state):
    """Create a short, avatar-safe description from grounded interaction state."""
    interactions = list(getattr(world_state, "hand_object_interactions", []) or [])
    events = list(getattr(world_state, "manipulation_events", []) or [])
    contact_events = list(getattr(world_state, "hand_contact_events", []) or [])
    now = time.time()
    release_speech_enabled = os.environ.get(
        "GUIDANCE_RELEASE_SPEECH_ENABLED",
        "0" if os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() == "sophie" else "1",
    ).lower() in {"1", "true", "yes"}

    recent_event = events[-1] if events else None
    recent_contact = contact_events[-1] if contact_events else None
    event_age_s = event_age(now, recent_event)
    contact_age_s = event_age(now, recent_contact)
    if event_age_s is not None and event_age_s > 2.8:
        recent_event = None
    if contact_age_s is not None and contact_age_s > 1.8:
        recent_contact = None
    active = None
    for item in interactions:
        try:
            temporal_contact = float(item.get("pred_contact_prob", 0.0) or 0.0) >= 0.50
        except (TypeError, ValueError):
            temporal_contact = False
        if bool(item.get("learned_is_held", False)) or bool(item.get("is_contacting", False)) or bool(item.get("is_touching_strict", False)) or temporal_contact:
            active = item
            break

    explanation = None
    mode = "idle"
    details = {}

    if active is not None:
        obj_label = object_label_by_id(
            world_state,
            active.get("nearest_object_id"),
            active.get("nearest_object_label", "object"),
        )
        if not is_movable_label(obj_label):
            active = None
        else:
            preferred_target = preferred_target_for_cue(
                obj_label,
                world_state,
                active.get("nearest_object_id"),
                active,
            )
            release_prob = 0.0
            try:
                release_prob = float(active.get("pred_release_prob", 0.0) or 0.0)
            except (TypeError, ValueError):
                release_prob = 0.0
            place, place_score, target_label = top_target_candidate(
                world_state,
                active.get("intent_place_candidates", []),
                "pred_placement_prob",
                preferred_target,
            )
            if preferred_target in TARGET_LABELS and target_label != preferred_target:
                place = None
                place_score = -1.0
                target_label = preferred_target
            if bool(active.get("learned_releasing", False)) or release_prob >= 0.55:
                target_label = target_label or preferred_target
                explanation = f"{obj_label}: releasing near {target_label}."
                mode = "releasing"
                details["release_score"] = round(float(release_prob), 4)
                details["place_target"] = target_label
                details["learned_release_age_s"] = active.get("learned_release_age_s")
            elif place is not None and place_score >= 0.35:
                explanation = f"{obj_label}: held. Target: {target_label}."
                details["place_target"] = target_label
                details["place_target_id"] = place.get("object_id")
                details["place_score"] = round(float(place_score), 4)
            else:
                target_label = preferred_target
                explanation = f"{obj_label}: held. Target: {target_label}."
                details["place_target"] = target_label
            if mode != "releasing":
                mode = "grabbed"
            details["object"] = obj_label
            details["object_id"] = active.get("nearest_object_id")

    if explanation is None and release_speech_enabled and recent_event is not None:
        obj_label = object_label_by_id(world_state, recent_event.get("object_id"), recent_event.get("label", "object"))
        if not is_movable_label(obj_label):
            recent_event = None
        else:
            relation = recent_event.get("place_relation") or {}
            target_id = relation.get("target_id") or relation.get("nearest_object_id")
            target_hint = relation.get("target_label") or relation.get("nearest_object_label") or "target"
            target_label = object_label_by_id(world_state, target_id, target_hint)
            support_score = float(relation.get("support_score", 0.0) or 0.0)
            support_confident = bool(
                relation.get("is_on_support", False)
                or relation.get("bbox_center_inside", False)
                or float(relation.get("bbox_overlap", 0.0) or 0.0) >= 0.20
                or support_score >= 1.20
                or relation.get("support_inferred_from_transfer_memory", False)
            )
            moved = bool(recent_event.get("moved", False))
            preferred_target = default_target_for_object(obj_label, world_state, recent_event.get("object_id"))
            if moved and support_confident and is_target_label(target_label) and target_label == preferred_target:
                explanation = f"{obj_label}: released near {target_label}."
                mode = "released"
                details["object"] = obj_label
                details["object_id"] = recent_event.get("object_id")
                details["place_target"] = target_label
                details["place_target_id"] = target_id
                details["support_score"] = round(float(support_score), 4)
            else:
                recent_event = None
    if explanation is None and recent_contact is not None and str(recent_contact.get("event")) == "contact_start":
        obj_label = object_label_by_id(world_state, recent_contact.get("object_id"), recent_contact.get("label", "object"))
        if is_movable_label(obj_label):
            target_label = preferred_target_for_cue(obj_label, world_state, recent_contact.get("object_id"), recent_contact)
            explanation = f"{obj_label}: held. Target: {target_label}."
            mode = "grabbed"
            details["object"] = obj_label
            details["object_id"] = recent_contact.get("object_id")
            details["place_target"] = target_label
    if explanation is None:
        nearest = None
        nearest_dist = None
        for item in interactions:
            try:
                dist = float(item.get("distance_m", 9.9) or 9.9)
            except (TypeError, ValueError):
                continue
            if nearest is None or dist < nearest_dist:
                nearest = item
                nearest_dist = dist
        if nearest is not None:
            obj_label = object_label_by_id(
                world_state,
                nearest.get("nearest_object_id"),
                nearest.get("nearest_object_label", "object"),
            )
            explanation = None
            mode = "approach"
            details["object"] = obj_label
            details["object_id"] = nearest.get("nearest_object_id")
            details["distance_m"] = round(float(nearest_dist), 4) if nearest_dist is not None else None
            grab, grab_score, grab_label = top_movable_candidate(
                world_state,
                nearest.get("intent_grab_candidates", []),
                "pred_contact_prob",
            )
            intent_threshold = float(os.environ.get("GUIDANCE_INTENT_CONTACT_THRESHOLD", "0.35"))
            if grab is not None and grab_score >= intent_threshold:
                target_label = preferred_target_for_cue(grab_label, world_state, grab.get("object_id"), grab, nearest)
                explanation = f"{grab_label}: likely next. Target: {target_label}."
                details["object"] = grab_label
                details["object_id"] = grab.get("object_id")
                details["grab_score"] = round(float(grab_score), 4)
                details["place_target"] = target_label
            if explanation is None and is_movable_label(obj_label) and nearest_dist is not None and nearest_dist <= 0.38:
                target_label = preferred_target_for_cue(obj_label, world_state, nearest.get("nearest_object_id"), nearest)
                explanation = f"{obj_label}: close. Target: {target_label}."
                details["place_target"] = target_label

    if explanation is None and mode in {"idle", "approach"}:
        hands = list(getattr(world_state, "hand_tracks", {}).values() or [])
        visible_hands = [
            hand for hand in hands
            if int(hand.get("missing_frames", 0) or 0) <= 2
            and float(hand.get("confidence", 0.0) or 0.0) >= 0.35
        ]
        if visible_hands:
            explanation = "hand detected."
            mode = "hand_detected"
            details["hand_count"] = int(len(visible_hands))

    return {
        "ok": True,
        "mode": mode,
        "explanation": explanation,
        "message": explanation or "No grounded interaction cue yet.",
        "details": details,
        "event_age_s": round(float(event_age_s), 3) if event_age_s is not None else None,
        "contact_age_s": round(float(contact_age_s), 3) if contact_age_s is not None else None,
        "interaction_count": int(len(interactions)),
        "event_count": int(len(events)),
    }
