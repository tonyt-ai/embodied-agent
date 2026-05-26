"""Train interaction-conditioned temporal heads from pseudo-labeled trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep optional native runtime telemetry/log noise out of training output. Set
# MEDIAPIPE_LOG_LEVEL=0 to debug MediaPipe itself.
_native_log_level = os.environ.get("MEDIAPIPE_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", _native_log_level)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", _native_log_level)

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from depth import estimate_depth, stabilize_depth_with_anchors
from hands import HandTracker
from jepa_encoder import JepaFeatureEncoder
from lift_to_3d import infer_camera_intrinsics, lift_bbox_to_3d
from perception_candidates import build_semantic_candidates
from semantic_labels import normalize_label
from slam_backend import BuiltinSparseSlamBackend
from temporal_interaction_head import TemporalInteractionHead, build_feature_vector
from world_state import WorldState
from ultralytics import YOLO

try:
    from dino_encoder import encode_bbox as encode_dino_bbox
except Exception:
    encode_dino_bbox = None


def _resolve_model(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    root = Path(__file__).resolve().parent.parent / p.name
    return root if root.is_file() else p


def _pad_embedding(values, n=64):
    out = np.zeros((n,), dtype=np.float32)
    if isinstance(values, (list, tuple)):
        m = min(len(values), n)
        if m > 0:
            out[:m] = np.asarray(values[:m], dtype=np.float32)
    return out


def _latent_from_row(row, emb_dim=64):
    if isinstance(row.get("obj_emb"), list) and row.get("obj_emb"):
        return _pad_embedding(row.get("obj_emb"), emb_dim)
    feat = row.get("feat")
    if isinstance(feat, list) and len(feat) >= emb_dim * 2:
        return _pad_embedding(feat[emb_dim:emb_dim * 2], emb_dim)
    return np.zeros((emb_dim,), dtype=np.float32)


def add_future_latent_targets(rows, horizon=10, emb_dim=64):
    for i, row in enumerate(rows):
        j = min(len(rows) - 1, i + max(1, int(horizon)))
        future = rows[j]
        for cand in rows[i + 1:j + 1]:
            if cand.get("object_id") == row.get("object_id"):
                future = cand
        row["y_future_latent"] = _latent_from_row(future, emb_dim).tolist()
    return rows


MOVABLE_EPISODE_LABELS = {
    "bottle",
    "baby bottle",
    "cup",
    "toy",
    "toy giraffe",
    "donut",
    "mouse",
}

DINO_STATE_DIM = 32


def _bbox_area_norm(bbox) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0.0
    try:
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    except (TypeError, ValueError):
        return 0.0


def _nonzero_embedding(emb) -> bool:
    if not isinstance(emb, list) or not emb:
        return False
    try:
        return any(abs(float(v)) > 1e-8 for v in emb)
    except (TypeError, ValueError):
        return False


def _cosine_embedding(a, b) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    try:
        av = np.asarray(a[:n], dtype=np.float32)
        bv = np.asarray(b[:n], dtype=np.float32)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom <= 1e-8:
            return 0.0
        return float(np.clip(np.dot(av, bv) / denom, -1.0, 1.0))
    except Exception:
        return 0.0


def _encode_dino_identity(frame, bbox, *, out_dim=DINO_STATE_DIM):
    if encode_dino_bbox is None:
        return []
    if _bbox_area_norm(bbox) <= 0.0:
        return []
    try:
        emb = encode_dino_bbox(frame, bbox, out_dim=out_dim)
    except Exception:
        return []
    return emb if _nonzero_embedding(emb) else []


def _sophie_identity_prototypes(world_state: WorldState) -> dict[str, list[float]]:
    if os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() != "sophie":
        return {}
    prototypes: dict[str, list[list[float]]] = {}
    for obj in getattr(world_state, "objects", {}).values():
        label = normalize_label(
            obj.get("scene_memory_label")
            or obj.get("semantic_label")
            or obj.get("label")
            or obj.get("raw_label")
            or ""
        )
        visual = str(obj.get("visual_identity_class") or "").strip().lower()
        if visual == "baby_bottle":
            label = "baby bottle"
        elif visual == "toy_giraffe":
            label = "toy giraffe"
        if label not in {"baby bottle", "toy giraffe"}:
            continue
        emb = obj.get("dino_embedding") or (obj.get("embedding") if str(obj.get("embedding_source", "")).lower() == "dino" else [])
        if _nonzero_embedding(emb):
            prototypes.setdefault(label, []).append(emb)
    out = {}
    for label, embeds in prototypes.items():
        arrs = []
        for emb in embeds[-8:]:
            try:
                arr = np.asarray(emb, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 1e-8:
                    arrs.append(arr / norm)
            except Exception:
                continue
        if arrs:
            proto = np.mean(np.stack(arrs, axis=0), axis=0)
            norm = float(np.linalg.norm(proto))
            if norm > 1e-8:
                out[label] = (proto / norm).astype(np.float32).tolist()
    return out


def _target_label_for_support(label: str) -> str:
    norm = normalize_label(label)
    if norm in {"mat", "black mat", "table mat", "placemat", "dish", "plate"}:
        return "tray"
    if norm == "tray":
        return "mat"
    return ""


def _sophie_region_label_from_bbox(bbox) -> str:
    if os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() != "sophie":
        return ""
    if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
        return ""
    try:
        cx = (float(bbox[0]) + float(bbox[2])) * 0.5
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5
    except (TypeError, ValueError):
        return ""
    # The Sophie scene has two stable support regions from the COLMAP/static
    # phase: the tray/plate on the left and the black mat on the right. Use
    # soft elliptical regions instead of a hard vertical split; the plate reaches
    # into the center of the image, so an x cutoff incorrectly teaches tray
    # releases as mat.
    regions = [
        ("tray", 0.31, 0.49, 0.36, 0.43),
        ("mat", 0.73, 0.62, 0.34, 0.38),
    ]
    scored = []
    for label, rx0, ry0, rw, rh in regions:
        score = ((cx - rx0) / max(rw, 1e-6)) ** 2 + ((cy - ry0) / max(rh, 1e-6)) ** 2
        scored.append((score, label))
    score, label = min(scored, key=lambda item: item[0])
    return label if score <= 1.85 else ""


def _grounded_source_region(row: dict) -> str:
    """Prefer persistent support memory; keep the Sophie bbox as a weak fallback."""
    if not isinstance(row, dict):
        return ""
    support = normalize_label(row.get("source_support_label", ""))
    if support in {"mat", "tray"}:
        return support
    support = normalize_label(row.get("support_target_label", ""))
    if support in {"mat", "tray"}:
        return support
    return _sophie_region_label_from_bbox(row.get("bbox"))


def _infer_grounded_support_label(world_state, object_id: str, obj: dict, fallback: str = "") -> str:
    label = normalize_label(fallback)
    if label in {"mat", "tray"}:
        return label
    if not isinstance(obj, dict):
        return ""
    try:
        relation = world_state._find_place_relation(
            object_id,
            world_state._as_vec3(obj.get("position_camera_3d")),
            object_bbox=obj.get("bbox"),
        )
        if relation and world_state._support_relation_confident(relation):
            label = normalize_label(relation.get("nearest_object_label", ""))
            if label in {"mat", "tray"}:
                return label
    except Exception:
        pass
    try:
        nearest = world_state._find_best_static_target(None, object_bbox=obj.get("bbox"))
        if nearest and (
            bool(nearest.get("bbox_center_inside", False))
            or float(nearest.get("bbox_overlap", 0.0) or 0.0) >= 0.04
            or float(nearest.get("support_score", 0.0) or 0.0) >= 1.35
        ):
            label = normalize_label(nearest.get("label", ""))
            if label in {"mat", "tray"}:
                return label
    except Exception:
        pass
    return _sophie_region_label_from_bbox(obj.get("bbox"))


def _episode_label(label: str) -> str:
    norm = normalize_label(label)
    if norm in {"mouse", "donut", "toy"}:
        return "toy giraffe"
    if norm in {"cup", "bottle"}:
        return "baby bottle" if os.environ.get("DEMO_SCENE_PROFILE", "").lower() == "sophie" else norm
    return norm


def _visual_episode_label(row: dict) -> str:
    latent_label = normalize_label(row.get("visual_identity_label", ""))
    if latent_label in {"baby bottle", "toy giraffe"}:
        return latent_label
    label = _episode_label(row.get("object_label", ""))
    raw = _episode_label(row.get("object_raw_label", ""))
    if label == "toy giraffe" or raw == "toy giraffe":
        return "toy giraffe"
    bbox = row.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            bw = max(0.0, float(bbox[2]) - float(bbox[0]))
            bh = max(0.0, float(bbox[3]) - float(bbox[1]))
            aspect = bw / max(bh, 1e-6)
            area = bw * bh
            if label == "baby bottle" and area >= 0.006 and aspect >= 0.72:
                return "toy giraffe"
            if label in {"baby bottle", "bottle", "cup"} and aspect <= 0.68:
                return "baby bottle"
        except (TypeError, ValueError):
            pass
    return label


def apply_latent_identity_labels(rows: list[dict], emb_dim=64, margin=0.08) -> list[dict]:
    """Use object latents to stabilize bottle-vs-giraffe identity.

    Detector labels are noisy in the Sophie scene, especially under hand
    occlusion. The object embeddings are much more stable, so we bootstrap two
    prototypes from high-confidence raw labels and then relabel ambiguous rows
    by cosine similarity.
    """
    bottle = []
    giraffe = []
    for row in rows:
        emb = _latent_from_row(row, emb_dim)
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-6:
            continue
        emb = emb / norm
        raw = normalize_label(row.get("object_raw_label") or row.get("object_label") or "")
        bbox = row.get("bbox")
        aspect = None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                bw = max(0.0, float(bbox[2]) - float(bbox[0]))
                bh = max(0.0, float(bbox[3]) - float(bbox[1]))
                aspect = bw / max(bh, 1e-6)
            except (TypeError, ValueError):
                aspect = None
        if raw in {"donut", "mouse", "toy", "toy giraffe"}:
            giraffe.append(emb)
        elif raw in {"bottle", "baby bottle", "cup", "mug"} and (aspect is None or aspect <= 0.72):
            bottle.append(emb)
    if len(bottle) < 4 or len(giraffe) < 4:
        return rows
    bottle_c = np.mean(np.stack(bottle, axis=0), axis=0)
    giraffe_c = np.mean(np.stack(giraffe, axis=0), axis=0)
    bottle_c = bottle_c / max(float(np.linalg.norm(bottle_c)), 1e-6)
    giraffe_c = giraffe_c / max(float(np.linalg.norm(giraffe_c)), 1e-6)
    for row in rows:
        emb = _latent_from_row(row, emb_dim)
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-6:
            continue
        emb = emb / norm
        b = float(np.dot(emb, bottle_c))
        g = float(np.dot(emb, giraffe_c))
        row["visual_identity_bottle_cos"] = round(b, 4)
        row["visual_identity_giraffe_cos"] = round(g, 4)
        if b >= g + float(margin):
            row["visual_identity_label"] = "baby bottle"
            row["object_label"] = "baby bottle"
        elif g >= b + float(margin):
            row["visual_identity_label"] = "toy giraffe"
            row["object_label"] = "toy giraffe"
    return rows


def add_episode_targets(rows, horizon=10, min_contact_rows=4, gap_rows=8):
    """Convert raw grounded contact rows into stable self-supervised episodes.

    Raw hand-object contact flickers because detection labels and object ids can
    switch while the same physical grab is happening. This teacher keeps rows in
    an episode when contact recurs within a short gap, then trains the temporal
    head to predict the future episode state instead of the raw per-frame edge.
    """
    rows = [r for r in rows if isinstance(r, dict)]
    episodes = []
    active = None
    last_contact_i = None
    for i, row in enumerate(rows):
        label = _visual_episode_label(row)
        is_movable = label in {"baby bottle", "toy giraffe", "bottle", "cup", "toy"}
        try:
            effective_distance = float(row.get("effective_distance_m", row.get("distance_m", 9.0)) or 9.0)
            distance_m = float(row.get("distance_m", effective_distance) or effective_distance)
        except (TypeError, ValueError):
            effective_distance = 9.0
            distance_m = 9.0
        strict_or_close = bool(row.get("is_touching_strict", False)) or effective_distance <= 0.105 or distance_m <= 0.12
        is_contact = bool(row.get("is_contacting", False)) and is_movable and strict_or_close
        source_region = _grounded_source_region(row)
        target = _target_label_for_support(source_region)
        if is_contact:
            should_split_label = (
                active is not None
                and label in {"baby bottle", "toy giraffe"}
                and active.get("last_label") in {"baby bottle", "toy giraffe"}
                and label != active.get("last_label")
            )
            if should_split_label:
                active["end"] = last_contact_i if last_contact_i is not None else active.get("last", i)
                episodes.append(active)
                active = None
            if active is None:
                active = {
                    "start": i,
                    "end": i,
                    "last": i,
                    "last_label": label,
                    "labels": {},
                    "targets": {},
                    "regions": [],
                    "contact_rows": 0,
                    "object_ids": {},
                }
            elif last_contact_i is not None and i - last_contact_i > max(1, int(gap_rows)):
                active["end"] = last_contact_i
                episodes.append(active)
                active = {
                    "start": i,
                    "end": i,
                    "last": i,
                    "last_label": label,
                    "labels": {},
                    "targets": {},
                    "regions": [],
                    "contact_rows": 0,
                    "object_ids": {},
                }
            active["last"] = i
            active["last_label"] = label
            active["end"] = i
            active["contact_rows"] += 1
            active["labels"][label] = active["labels"].get(label, 0) + 1
            object_id = str(row.get("object_id") or "")
            if object_id:
                active["object_ids"][object_id] = active["object_ids"].get(object_id, 0) + 1
            current_region = source_region
            if current_region in {"mat", "tray"}:
                active["regions"].append(current_region)
            if target:
                active["targets"][target] = active["targets"].get(target, 0) + 1
            last_contact_i = i
        elif active is not None and last_contact_i is not None and i - last_contact_i > max(1, int(gap_rows)):
            active["end"] = last_contact_i
            episodes.append(active)
            active = None
            last_contact_i = None
    if active is not None:
        active["end"] = active.get("last", active["start"])
        episodes.append(active)

    stable = []
    if os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() == "sophie":
        physical_clusters = []
        current = []
        last_t = None
        for i, row in enumerate(rows):
            label = _visual_episode_label(row)
            if label not in {"baby bottle", "toy giraffe", "bottle", "cup", "toy"}:
                continue
            try:
                t = float(row.get("video_time_s", 0.0) or 0.0)
                effective_distance = float(row.get("effective_distance_m", row.get("distance_m", 9.0)) or 9.0)
                distance_m = float(row.get("distance_m", effective_distance) or effective_distance)
            except (TypeError, ValueError):
                continue
            strict_or_close = bool(row.get("is_touching_strict", False)) or effective_distance <= 0.105 or distance_m <= 0.12
            if not (bool(row.get("is_contacting", False)) and strict_or_close):
                continue
            if current and last_t is not None and t - last_t > 0.95:
                physical_clusters.append(current)
                current = []
            current.append(i)
            last_t = t
        if current:
            physical_clusters.append(current)

        for cluster in physical_clusters:
            by_object = {}
            for i in cluster:
                row = rows[i]
                label = _visual_episode_label(row)
                object_id = str(row.get("object_id") or "")
                key = object_id or label
                if not key:
                    continue
                feat = row.get("feat") if isinstance(row.get("feat"), list) else []
                try:
                    obj_speed = float(feat[132]) if len(feat) > 132 else 0.0
                    effective_distance = float(row.get("effective_distance_m", row.get("distance_m", 9.0)) or 9.0)
                except (TypeError, ValueError):
                    obj_speed = 0.0
                    effective_distance = 9.0
                item = by_object.setdefault(key, {
                    "indices": [],
                    "labels": {},
                    "regions": [],
                    "score": 0.0,
                    "object_id": object_id,
                })
                item["indices"].append(i)
                item["labels"][label] = item["labels"].get(label, 0) + 1
                region = _grounded_source_region(row)
                if region in {"mat", "tray"}:
                    item["regions"].append(region)
                item["score"] += 0.05
                item["score"] += min(max(obj_speed, 0.0), 0.5) * 5.0
                item["score"] += 0.8 if bool(row.get("is_touching_strict", False)) else 0.0
                item["score"] += max(0.0, 0.12 - effective_distance) * 2.0
            if not by_object:
                continue
            owner = max(by_object.values(), key=lambda item: item["score"])
            if len(owner["indices"]) < max(1, int(min_contact_rows)):
                continue
            label = max(owner["labels"].items(), key=lambda kv: kv[1])[0] if owner.get("labels") else "object"
            if label not in {"baby bottle", "toy giraffe"}:
                continue
            regions = list(owner.get("regions") or [])
            source = regions[0] if regions else ""
            target = _target_label_for_support(source)
            if not target and regions:
                tail = regions[max(0, len(regions) // 2):]
                target = max({r: tail.count(r) for r in set(tail)}.items(), key=lambda kv: kv[1])[0]
            stable.append({
                "start": min(owner["indices"]),
                "end": max(owner["indices"]),
                "label": label,
                "target": target,
                "object_ids": {owner.get("object_id", ""): len(owner["indices"])},
            })

    if not stable:
        for ep in episodes:
            if int(ep.get("contact_rows", 0)) < max(1, int(min_contact_rows)):
                continue
            label = max(ep["labels"].items(), key=lambda kv: kv[1])[0] if ep.get("labels") else "object"
            if ep.get("object_ids"):
                dominant_object_count = max(ep["object_ids"].values())
                if dominant_object_count / max(1, int(ep.get("contact_rows", 0))) < 0.62:
                    continue
            regions = list(ep.get("regions") or [])
            tail = regions[max(0, len(regions) // 2):] if regions else []
            if tail:
                target = max({r: tail.count(r) for r in set(tail)}.items(), key=lambda kv: kv[1])[0]
            elif regions:
                source = regions[0]
                target = _target_label_for_support(source) or ""
            else:
                target = max(ep["targets"].items(), key=lambda kv: kv[1])[0] if ep.get("targets") else ""
            ep["label"] = label
            ep["target"] = target
            stable.append(ep)

    for row in rows:
        row.pop("episode_id", None)
        row.pop("snapshot_episode_target", None)
        row.pop("release_target_time_s", None)
        row["episode_active"] = 0.0
        row["episode_starts"] = 0.0
        row["episode_ends"] = 0.0
        row["episode_label"] = ""
        row["episode_target"] = ""
        row["y_target_tray"] = 0.0
        row["y_release"] = 0.0
        row["future_episode_label"] = ""
        row["future_episode_target"] = ""

    for ep_idx, ep in enumerate(stable):
        start = int(ep["start"])
        end = int(ep["end"])
        target = str(ep.get("target") or "")
        label = str(ep.get("label") or "")
        for i in range(start, min(end + 1, len(rows))):
            rows[i]["episode_active"] = 1.0
            rows[i]["episode_id"] = ep_idx
            rows[i]["episode_label"] = label
            rows[i]["episode_target"] = target
            rows[i]["y_target_tray"] = 1.0 if target == "tray" else 0.0
        if 0 <= start < len(rows):
            rows[start]["episode_starts"] = 1.0
        if 0 <= end < len(rows):
            rows[end]["episode_ends"] = 1.0

    refresh_episode_future_targets(rows, horizon=horizon)
    return rows


def refresh_episode_future_targets(rows, horizon=10):
    for i, row in enumerate(rows):
        j = min(len(rows) - 1, i + max(1, int(horizon)))
        future_slice = rows[i:j + 1]
        row_label = _visual_episode_label(row)
        row_object_id = str(row.get("object_id") or "")

        def same_candidate_episode(candidate: dict) -> bool:
            episode_label = _episode_label(candidate.get("episode_label") or candidate.get("object_label") or "")
            candidate_object_id = str(candidate.get("object_id") or "")
            if row_object_id and candidate_object_id and row_object_id == candidate_object_id:
                return True
            return bool(row_label and episode_label and row_label == episode_label)

        future_start = next((
            r for r in future_slice
            if float(r.get("episode_starts", 0.0) or 0.0) >= 0.5 and same_candidate_episode(r)
        ), None)
        future_end = next((
            r for r in future_slice
            if float(r.get("episode_ends", 0.0) or 0.0) >= 0.5 and same_candidate_episode(r)
        ), None)
        future_episode = future_start or next((
            r for r in future_slice
            if float(r.get("episode_active", 0.0) or 0.0) >= 0.5 and same_candidate_episode(r)
        ), None)
        row["y_contact_raw"] = float(row.get("y_contact", 0.0) or 0.0)
        row["y_episode_active"] = float(row.get("episode_active", 0.0) or 0.0)
        row["y_contact"] = 1.0 if future_start is not None else 0.0
        row["y_release"] = 1.0 if (
            future_end is not None
            and float(row.get("episode_active", 0.0) or 0.0) >= 0.5
            and same_candidate_episode(row)
        ) else 0.0
        if future_episode is not None:
            row["y_target_tray"] = float(future_episode.get("y_target_tray", row.get("y_target_tray", 0.0)) or 0.0)
            row["future_episode_label"] = future_episode.get("episode_label", "")
            row["future_episode_target"] = future_episode.get("episode_target", "")
        elif not row.get("future_episode_target"):
            source_region = _grounded_source_region(row)
            target = _target_label_for_support(source_region)
            if target:
                row["future_episode_target"] = target
                row["y_target_tray"] = 1.0 if target == "tray" else 0.0
    return rows


def add_snapshot_target_supervision(rows, snapshots, horizon=10, settle_window_s=4.0):
    """Use post-contact object observations to label the destination support."""
    by_episode = {}
    for row in rows:
        if "episode_id" not in row:
            continue
        try:
            ep_id = int(row.get("episode_id"))
            t = float(row.get("video_time_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        ep = by_episode.setdefault(ep_id, {"rows": [], "start_t": t, "end_t": t, "labels": {}})
        ep["rows"].append(row)
        ep["start_t"] = min(ep["start_t"], t)
        ep["end_t"] = max(ep["end_t"], t)
        label = _episode_label(row.get("episode_label") or row.get("object_label", ""))
        if label:
            ep["labels"][label] = ep["labels"].get(label, 0) + 1
    for ep in by_episode.values():
        if not ep["labels"]:
            continue
        label = max(ep["labels"].items(), key=lambda kv: kv[1])[0]
        regions = []
        for snap in snapshots or []:
            snap_label = _episode_label(snap.get("label", ""))
            if snap_label != label:
                continue
            try:
                t = float(snap.get("video_time_s", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if ep["end_t"] - 0.25 <= t <= ep["end_t"] + float(settle_window_s):
                region = _grounded_source_region(snap)
                if region in {"mat", "tray"}:
                    regions.append(region)
        if not regions:
            continue
        target = max({r: regions.count(r) for r in set(regions)}.items(), key=lambda kv: kv[1])[0]
        for row in ep["rows"]:
            row["episode_target"] = target
            row["snapshot_episode_target"] = target
            row["y_target_tray"] = 1.0 if target == "tray" else 0.0
    refresh_episode_future_targets(rows, horizon=horizon)
    return rows


def _release_target_label(event: dict) -> str:
    relation = event.get("place_relation") or {}
    label = normalize_label(
        relation.get("nearest_object_label")
        or relation.get("support_target_label")
        or relation.get("target_label")
        or ""
    )
    if label in {"black mat", "table mat", "placemat", "dish", "plate"}:
        return "mat"
    if label in {"mat", "tray"}:
        return label
    return ""


def build_release_target_events(events, cluster_gap_s=4.0):
    """Build blind release/support events for target supervision."""
    candidates = []
    for ev in events or []:
        relation = ev.get("place_relation") or {}
        target = _release_target_label(ev)
        if target not in {"mat", "tray"}:
            continue
        try:
            t = float(ev.get("video_time_s", 0.0) or 0.0)
            moved = bool(ev.get("moved", False))
            move_distance = float(ev.get("move_distance_m", 0.0) or 0.0)
            support_score = float(relation.get("support_score", 0.0) or 0.0)
            overlap = float(relation.get("bbox_overlap", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        support_ok = bool(
            relation.get("is_on_support", False)
            or relation.get("bbox_center_inside", False)
            or relation.get("support_inferred_from_transfer_memory", False)
            or overlap >= 0.18
            or support_score >= 1.0
        )
        if not moved or move_distance < 0.05 or not support_ok:
            continue
        score = (2.0 * move_distance) + (0.25 * support_score) + (2.0 * overlap)
        candidates.append({
            "time": t,
            "target": target,
            "label": _episode_label(ev.get("label", "")),
            "object_id": str(ev.get("object_id") or ""),
            "score": float(score),
            "event": ev,
        })
    candidates.sort(key=lambda item: item["time"])
    clusters = []
    cur = []
    last_t = None
    for item in candidates:
        if last_t is None or item["time"] - last_t <= float(cluster_gap_s):
            cur.append(item)
        else:
            clusters.append(cur)
            cur = [item]
        last_t = item["time"]
    if cur:
        clusters.append(cur)
    out = []
    for cluster in clusters:
        best = max(cluster, key=lambda item: item["score"])
        votes = {}
        for item in cluster:
            votes[item["target"]] = votes.get(item["target"], 0.0) + item["score"]
        target = max(votes.items(), key=lambda kv: kv[1])[0] if votes else best["target"]
        out.append({
            "time": float(best["time"]),
            "target": target,
            "label": best.get("label", ""),
            "object_id": best.get("object_id", ""),
            "score": float(best["score"]),
            "cluster_size": int(len(cluster)),
        })
    return out


def add_release_target_supervision(rows, release_events, lookback_s=14.0, lookahead_s=2.0):
    """Use future grounded release/support events to supervise mat/tray target.

    Release events are noisy in this webcam setup, so never let one object's
    release globally relabel every candidate row in the same time window. The
    event must match the row's visual identity, and object ids are used when the
    tracker keeps them stable.
    """
    if (
        os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() == "sophie"
        and os.environ.get("SOPHIE_USE_RELEASE_TARGET_TEACHER", "0").strip().lower() not in {"1", "true", "yes"}
    ):
        return rows
    releases = build_release_target_events(release_events)
    if not releases:
        return rows
    for row in rows:
        try:
            t = float(row.get("video_time_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        row_label = _visual_episode_label(row)
        row_object_id = str(row.get("object_id") or "")
        future = [
            ev for ev in releases
            if t - float(lookahead_s) <= ev["time"] <= t + float(lookback_s)
            and (
                (row_object_id and ev.get("object_id") and row_object_id == ev.get("object_id"))
                or (row_label and ev.get("label") and row_label == ev.get("label"))
            )
        ]
        if not future:
            continue
        ev = min(future, key=lambda item: abs(item["time"] - t))
        row["future_episode_target"] = ev["target"]
        row["release_target_time_s"] = round(float(ev["time"]), 3)
        row["y_target_tray"] = 1.0 if ev["target"] == "tray" else 0.0
    return rows


def _bbox_from_hand_landmarks(landmarks_px, image_shape):
    if not isinstance(landmarks_px, list) or not landmarks_px:
        return None
    h, w = image_shape[:2]
    xs, ys = [], []
    for pt in landmarks_px:
        if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
            continue
        xs.append(float(pt[0])); ys.append(float(pt[1]))
    if len(xs) < 4:
        return None
    x1 = max(0.0, min(float(w - 1), min(xs)))
    y1 = max(0.0, min(float(h - 1), min(ys)))
    x2 = max(0.0, min(float(w - 1), max(xs)))
    y2 = max(0.0, min(float(h - 1), max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 / max(w, 1), y1 / max(h, 1), x2 / max(w, 1), y2 / max(h, 1)]


def collect_examples(video_path: str, horizon: int = 10):
    os.environ["JEPA_ENABLED"] = "1"
    os.environ["JEPA_USE_FOR_CONTACT"] = "1"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    model = YOLO(str(_resolve_model(os.environ.get("YOLO_MODEL_PATH", str(Path(__file__).parent / "models" / "yolov8n.pt")))))
    seg_backend = os.environ.get("PERCEPTION_SEGMENTATION_BACKEND", "yolo-seg").strip().lower()
    seg_model = None
    if seg_backend == "fastsam-s":
        seg_path = _resolve_model(os.environ.get("FASTSAM_S_MODEL_PATH", str(Path(__file__).parent / "models" / "FastSAM-s.pt")))
    elif seg_backend == "fastsam-x":
        seg_path = _resolve_model(os.environ.get("FASTSAM_X_MODEL_PATH", str(Path(__file__).parent / "models" / "FastSAM-x.pt")))
    else:
        seg_path = _resolve_model(os.environ.get("YOLO_SEG_MODEL_PATH", str(Path(__file__).parent / "models" / "yolov8n-seg.pt")))
    if seg_path.is_file():
        seg_model = YOLO(str(seg_path))
    backend = BuiltinSparseSlamBackend()
    hand_tracker = HandTracker()
    world_state = WorldState(collection_mode=False)
    jepa = JepaFeatureEncoder()
    allowed = {
        "cup", "mug", "apple", "orange", "banana", "bowl", "vase", "bottle",
        "book", "cell phone", "toy", "box",
        "mouse", "donut",
        "tray", "mat", "black mat", "table mat", "placemat", "dish", "plate", "unknown_seg", "unknown seg",
    }

    rows = []
    release_events = []
    object_snapshots = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % 2 != 0:
            continue

        intr = infer_camera_intrinsics(frame.shape[1], frame.shape[0])
        raw_depth = estimate_depth(frame)
        pose = backend.update(frame, depth_map=raw_depth, intrinsics=intr)
        depth_map, _ = stabilize_depth_with_anchors(raw_depth, pose)
        pose = backend.refine_visible_landmarks(depth_map, intr, pose)

        detections = []
        candidates = build_semantic_candidates(
            frame,
            model,
            segmentation_model=seg_model,
            detector_conf_min=0.10,
            segmentation_conf_min=0.08,
            segmentation_source=seg_backend,
            add_unmatched=True,
            unmatched_min_conf=0.12,
            unmatched_min_area=0.003,
            unmatched_max_area=0.30,
            unmatched_max_items=8,
        )
        dino_reid_enabled = (
            encode_dino_bbox is not None
            and os.environ.get("USE_DINO_EMBEDDING", "1").lower() in {"1", "true", "yes"}
            and os.environ.get("DEMO_SCENE_PROFILE", "").strip().lower() == "sophie"
        )
        dino_prototypes = _sophie_identity_prototypes(world_state) if dino_reid_enabled else {}
        for candidate in candidates:
            conf = float(candidate.get("confidence", 0.0) or 0.0)
            if conf < 0.12:
                continue
            raw_label = str(candidate.get("label", "")).lower()
            label = normalize_label(raw_label)
            if label not in allowed:
                continue
            bbox = candidate.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            bbox = [float(v) for v in bbox]
            dino_emb = []
            visual_identity_label = ""
            visual_identity_score = 0.0
            visual_identity_margin = 0.0
            if dino_reid_enabled and label not in {
                "tray", "mat", "black mat", "table mat", "placemat", "dish", "plate"
            } and 0.001 <= _bbox_area_norm(bbox) <= 0.34:
                dino_emb = _encode_dino_identity(frame, bbox)
                if dino_emb and dino_prototypes:
                    sims = {name: _cosine_embedding(dino_emb, proto) for name, proto in dino_prototypes.items()}
                    if sims:
                        best_label, best_sim = max(sims.items(), key=lambda kv: kv[1])
                        other = max([v for k, v in sims.items() if k != best_label] or [0.0])
                        margin = float(best_sim) - float(other)
                        if best_sim >= 0.70 and margin >= 0.04:
                            label = best_label
                            visual_identity_label = best_label
                            visual_identity_score = float(best_sim)
                            visual_identity_margin = float(margin)
            lifted = lift_bbox_to_3d(bbox, depth_map, pose, intr, sparse_points=pose.get("sparse_map", []))
            item = {
                "label": label,
                "raw_label": raw_label,
                "confidence": conf,
                "bbox": bbox,
                "x": float((bbox[0] + bbox[2]) * 0.5),
                "y": float((bbox[1] + bbox[3]) * 0.5),
                "position_3d": lifted.get("position_world_3d", [0.0, 0.0, 0.0]),
                "position_camera_3d": lifted.get("position_camera_3d", [0.0, 0.0, 0.0]),
                "embedding": dino_emb if dino_emb else [0.0] * DINO_STATE_DIM,
                "embedding_source": "dino" if dino_emb else "none",
                "dino_embedding": dino_emb,
                "jepa_embedding": jepa.encode_bbox(frame, bbox) if jepa.ready else [],
                "mask_polygon": candidate.get("mask_polygon"),
                "segmentation_source": candidate.get("segmentation_source", "bbox"),
            }
            if visual_identity_label:
                item["semantic_label"] = visual_identity_label
                item["visual_identity_label"] = visual_identity_label
                item["visual_identity_score"] = round(visual_identity_score, 4)
                item["visual_identity_margin"] = round(visual_identity_margin, 4)
                item["visual_identity_class"] = "baby_bottle" if visual_identity_label == "baby bottle" else "toy_giraffe"
            detections.append(item)
            if label in MOVABLE_EPISODE_LABELS:
                object_snapshots.append({
                    "video_time_s": float((frame_idx - 1) / max(fps, 1e-6)),
                    "video_frame": int(frame_idx),
                    "label": label,
                    "object_id": "",
                    "bbox": bbox,
                })

        hands, _ = hand_tracker.detect(frame, depth_map, intr, pose)
        for hand in hands:
            hbbox = _bbox_from_hand_landmarks(hand.get("landmarks_px"), frame.shape[:2])
            hand["jepa_embedding"] = jepa.encode_bbox(frame, hbbox) if (jepa.ready and hbbox is not None) else []

        world_state.update(
            detections,
            camera_pose=pose,
            hands=hands,
            world_debug={"intrinsics": intr},
            sparse_map=pose.get("sparse_map", []),
        )
        t_s = float((frame_idx - 1) / max(fps, 1e-6))
        while len(release_events) < len(world_state.manipulation_events):
            event = dict(world_state.manipulation_events[len(release_events)])
            event["video_time_s"] = round(t_s, 3)
            release_events.append(event)
        for it in world_state.hand_object_interactions:
            hand = world_state.hand_tracks.get(it.get("hand_id"))
            object_id = it.get("held_object_id") or it.get("learned_object_id") or it.get("nearest_object_id")
            obj = world_state.objects.get(object_id)
            if hand is None or obj is None:
                continue
            support_label = _infer_grounded_support_label(
                world_state,
                str(object_id or ""),
                obj,
                fallback=it.get("source_support_label") or obj.get("support_target_label", ""),
            )
            if support_label:
                obj["support_target_label"] = support_label
            hand_speed = float(np.linalg.norm(np.asarray(hand.get("velocity_3d", [0.0, 0.0, 0.0]), dtype=np.float32)))
            obj_speed = float(np.linalg.norm(np.asarray(obj.get("velocity_3d", [0.0, 0.0, 0.0]), dtype=np.float32)))
            semantic_identity_label = normalize_label(obj.get("semantic_label", ""))
            visual_identity_label = obj.get("visual_identity_label", "")
            if not visual_identity_label and semantic_identity_label in {"baby bottle", "toy giraffe"}:
                visual_identity_label = semantic_identity_label
            feat = build_feature_vector(
                hand.get("jepa_temporal_embedding", hand.get("jepa_embedding", [])),
                obj.get("jepa_temporal_embedding", obj.get("jepa_embedding", [])),
                float(it.get("distance_m", 0.0)),
                float(it.get("effective_distance_m", it.get("distance_m", 0.0))),
                float(it.get("jepa_similarity", 0.0)),
                hand_speed,
                obj_speed,
                int(hand.get("contact_streak", 0)),
                obj_pos_3d=obj.get("position_3d", [0.0, 0.0, 0.0]),
                bbox=obj.get("bbox", []),
                support_label=support_label,
            )
            rows.append({
                "frame": len(rows),
                "video_frame": int(frame_idx),
                "video_time_s": t_s,
                "hand_id": it.get("hand_id"),
                "object_id": object_id,
                "object_label": obj.get("scene_memory_label") or obj.get("semantic_label") or obj.get("label"),
                "object_raw_label": obj.get("raw_label"),
                "scene_memory_id": obj.get("scene_memory_id", ""),
                "scene_memory_label": obj.get("scene_memory_label", ""),
                "scene_memory_score": float(obj.get("scene_memory_score", 0.0) or 0.0),
                "visual_identity_label": visual_identity_label,
                "visual_identity_score": float(obj.get("visual_identity_score", 0.0) or 0.0),
                "visual_identity_margin": float(obj.get("visual_identity_margin", 0.0) or 0.0),
                "source_support_label": support_label,
                "bbox": obj.get("bbox", []),
                "feat": feat.tolist(),
                "is_contacting": bool(it.get("is_contacting", False)),
                "is_touching_strict": bool(it.get("is_touching_strict", False)),
                "distance_m": float(it.get("distance_m", 9.0) or 9.0),
                "effective_distance_m": float(it.get("effective_distance_m", it.get("distance_m", 9.0)) or 9.0),
                "penetration_m": float(it.get("penetration_m", 0.0) or 0.0),
                "obj_pos": obj.get("position_3d", [0.0, 0.0, 0.0]),
                "obj_emb": obj.get("jepa_temporal_embedding", obj.get("jepa_embedding", [])),
            })

    cap.release()
    backend.close()

    # pseudo-label future targets
    for i, row in enumerate(rows):
        j = min(len(rows) - 1, i + horizon)
        future = rows[j]
        for cand in rows[i + 1:j + 1]:
            if cand.get("object_id") == row.get("object_id"):
                future = cand
        row["y_contact"] = 1.0 if future.get("is_contacting", False) else 0.0
        p0 = np.asarray(row.get("obj_pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        p1 = np.asarray(future.get("obj_pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        d = (p1 - p0).astype(np.float32)
        row["y_motion"] = d.tolist()
        row["y_place"] = 1.0 if float(np.linalg.norm(d)) > 0.03 and not future.get("is_contacting", False) else 0.0
        row["y_future_latent"] = _pad_embedding(future.get("obj_emb", []), 64).tolist()
    rows = apply_latent_identity_labels(rows)
    rows = add_episode_targets(
        rows,
        horizon=horizon,
        min_contact_rows=int(os.environ.get("TEMPORAL_EPISODE_MIN_CONTACT_ROWS", "4")),
        gap_rows=int(os.environ.get("TEMPORAL_EPISODE_GAP_ROWS", "8")),
    )
    rows = add_snapshot_target_supervision(rows, object_snapshots, horizon=horizon)
    rows = add_release_target_supervision(rows, release_events)
    return rows


def train(rows, model_out: str, epochs: int = 15, lr: float = 1e-3):
    if torch is None:
        raise RuntimeError("PyTorch is required")
    feats = np.asarray([r["feat"] for r in rows], dtype=np.float32)
    y_contact = np.asarray([r["y_contact"] for r in rows], dtype=np.float32).reshape(-1, 1)
    y_place = np.asarray([r["y_place"] for r in rows], dtype=np.float32).reshape(-1, 1)
    y_release = np.asarray([r.get("y_release", 0.0) for r in rows], dtype=np.float32).reshape(-1, 1)
    y_target_tray = np.asarray([r.get("y_target_tray", 0.0) for r in rows], dtype=np.float32).reshape(-1, 1)
    y_motion = np.asarray([r["y_motion"] for r in rows], dtype=np.float32)
    y_future_latent = np.asarray([r.get("y_future_latent", [0.0] * 64) for r in rows], dtype=np.float32)

    x = torch.from_numpy(feats)
    yc = torch.from_numpy(y_contact)
    yp = torch.from_numpy(y_place)
    yr = torch.from_numpy(y_release)
    yt = torch.from_numpy(y_target_tray)
    ym = torch.from_numpy(y_motion)
    yz = torch.from_numpy(y_future_latent)

    model = TemporalInteractionHead(in_dim=feats.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    def weighted_bce(target_tensor):
        positives = float(target_tensor.sum().item())
        negatives = float(target_tensor.numel() - positives)
        if positives <= 0.0:
            return nn.BCEWithLogitsLoss()
        pos_weight = torch.tensor([max(1.0, min(8.0, negatives / positives))], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    bce_contact = weighted_bce(yc)
    bce_place = weighted_bce(yp)
    bce_release = weighted_bce(yr)
    bce_target = weighted_bce(yt)
    mse = nn.MSELoss()

    for ep in range(epochs):
        out = model(x)
        lc = bce_contact(out["contact_logit"], yc)
        lp = bce_place(out["placement_logit"], yp)
        lr_ = bce_release(out.get("release_logit", out["placement_logit"]), yr)
        lt = bce_target(out["target_tray_logit"], yt)
        lm = mse(out["motion_delta"], ym)
        lz = mse(out["future_latent"], yz)
        loss = lc + lp + lr_ + 0.5 * lt + 0.5 * lm + 0.2 * lz
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            print(f"epoch={ep} loss={loss.item():.4f} contact={lc.item():.4f} place={lp.item():.4f} release={lr_.item():.4f} target={lt.item():.4f} motion={lm.item():.4f} latent={lz.item():.4f}")

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"saved {model_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="public/scene_hand.mp4")
    ap.add_argument("--out-model", default="world_model/models/temporal_interaction_head.pt")
    ap.add_argument("--out-json", default="world_model/data/temporal_head_train_rows.json")
    ap.add_argument("--rows-in", default="", help="Reuse existing rows JSON instead of recollecting video.")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--episode-min-contact-rows", type=int, default=int(os.environ.get("TEMPORAL_EPISODE_MIN_CONTACT_ROWS", "4")))
    ap.add_argument("--episode-gap-rows", type=int, default=int(os.environ.get("TEMPORAL_EPISODE_GAP_ROWS", "8")))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-train", action="store_true", help="Collect/write rows without updating a model checkpoint.")
    ap.add_argument("--vjepa2-repo", default="")
    ap.add_argument("--vjepa2-model", default="")
    args = ap.parse_args()

    if args.vjepa2_repo:
        os.environ["VJEPA2_REPO"] = args.vjepa2_repo
    if args.vjepa2_model:
        os.environ["VJEPA2_MODEL"] = args.vjepa2_model

    if args.rows_in:
        rows = json.loads(Path(args.rows_in).read_text(encoding="utf-8"))
        rows = add_future_latent_targets(rows, horizon=args.horizon)
        rows = add_episode_targets(
            rows,
            horizon=args.horizon,
            min_contact_rows=args.episode_min_contact_rows,
            gap_rows=args.episode_gap_rows,
        )
    else:
        rows = collect_examples(args.video, horizon=args.horizon)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rows), encoding="utf-8")
    print(f"rows={len(rows)}")
    if len(rows) < 20:
        raise RuntimeError("Not enough rows")
    if args.no_train:
        print("skipped model training (--no-train)")
        return
    train(rows, args.out_model, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
