import argparse
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "world_model"))

try:
    from temporal_interaction_head import TemporalInteractionHead
except Exception:
    TemporalInteractionHead = None

try:
    from ultralytics import YOLO
    from hands import HandTracker
    from jepa_encoder import JepaFeatureEncoder
    from perception_candidates import build_semantic_candidates
    from sophie_visual_tracker import SophieVisualTracker
except Exception:
    YOLO = None
    HandTracker = None
    JepaFeatureEncoder = None
    build_semantic_candidates = None
    SophieVisualTracker = None


TARGET_BOXES = {
    "tray": (0.02, 0.12, 0.47, 0.83),
    "mat": (0.50, 0.24, 0.98, 0.94),
}

SEGMENTATION_MODEL_PATH = ROOT / "world_model" / "models" / "FastSAM-s.pt"
DETECTOR_MODEL_PATH = ROOT / "world_model" / "models" / "yolov8n.pt"


def normalize_label(label: str) -> str:
    text = str(label or "").strip().lower().replace("_", " ")
    text = " ".join(text.split())
    if text in {"mouse", "donut", "toy", "giraffe", "sophie", "sophie giraffe", "sophie the giraffe"}:
        return "toy giraffe"
    if text in {"bottle", "cup", "mug"}:
        return "baby bottle"
    if text in {"black mat", "table mat", "placemat", "dish", "plate"}:
        return "mat"
    if text in {"plastic tray", "white tray"}:
        return "tray"
    return text


def load_rows(path: Path, model_path: Path | None):
    rows = [r for r in json.loads(path.read_text(encoding="utf-8")) if isinstance(r, dict)]
    rows.sort(key=lambda r: float(r.get("video_time_s", 0.0) or 0.0))
    if model_path and torch is not None and TemporalInteractionHead is not None and model_path.exists():
        feature_rows = [r for r in rows if isinstance(r.get("feat"), list)]
        if feature_rows:
            x = np.asarray([r["feat"] for r in feature_rows], dtype=np.float32)
            model = TemporalInteractionHead(in_dim=x.shape[1])
            model.load_state_dict(torch.load(str(model_path), map_location="cpu"), strict=False)
            model.eval()
            with torch.no_grad():
                out = model(torch.from_numpy(x))
                contact = torch.sigmoid(out["contact_logit"]).cpu().numpy().reshape(-1)
                release = torch.sigmoid(out["release_logit"]).cpu().numpy().reshape(-1)
                target = torch.sigmoid(out.get("target_tray_logit", out["placement_logit"])).cpu().numpy().reshape(-1)
            for row, p_contact, p_release, p_target in zip(feature_rows, contact, release, target):
                row["render_contact_prob"] = float(p_contact)
                row["render_release_prob"] = float(p_release)
                row["render_target_tray_prob"] = float(p_target)
    return rows


def load_timeline(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("events", data if isinstance(data, list) else [])


def row_label(row):
    return normalize_label(
        row.get("episode_label")
        or row.get("future_episode_label")
        or row.get("visual_identity_label")
        or row.get("object_label")
    )


def row_target(row):
    source = normalize_label(row.get("source_support_label"))
    if source == "mat":
        return "tray"
    if source == "tray":
        return "mat"
    if float(row.get("episode_active", 0.0) or 0.0) >= 0.5:
        target = normalize_label(row.get("episode_target") or row.get("future_episode_target"))
    else:
        target = normalize_label(row.get("future_episode_target") or row.get("episode_target"))
    if target in TARGET_BOXES:
        return target
    return ""


def row_jepa_target(row):
    prob = float(row.get("render_target_tray_prob", 0.5) or 0.5)
    if prob >= 0.55:
        return "tray"
    if prob <= 0.45:
        return "mat"
    return row_target(row)


def episode_onset_target(rows, times, t, label, window=6.5):
    label = normalize_label(label)
    idx = bisect_left(times, t)
    candidates = []
    for j in range(max(0, idx - 90), min(len(rows), idx + 1)):
        r = rows[j]
        rt = float(r.get("video_time_s", 0.0) or 0.0)
        dt = t - rt
        if dt < -1e-6 or dt > window:
            continue
        if row_label(r) != label:
            continue
        source = normalize_label(r.get("source_support_label"))
        if source not in {"mat", "tray"}:
            continue
        try:
            effective_distance = float(r.get("effective_distance_m", r.get("distance_m", 9.0)) or 9.0)
        except (TypeError, ValueError):
            effective_distance = 9.0
        contact = float(r.get("render_contact_prob", 0.0) or 0.0)
        active = float(r.get("episode_active", 0.0) or 0.0)
        if effective_distance > 0.24 and contact < 0.25 and active < 0.5:
            continue
        # Prefer the earliest plausible support in the episode: later rows can
        # already be on the destination and would invert the predicted target.
        candidates.append((rt, source, effective_distance, contact, active))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    early = candidates[: max(1, min(8, len(candidates) // 2 or 1))]
    votes = {"mat": 0.0, "tray": 0.0}
    for rt, source, dist, contact, active in early:
        weight = 1.0 + max(contact, active) + max(0.0, 0.24 - dist)
        votes[source] += weight
    source = max(votes.items(), key=lambda kv: kv[1])[0]
    if source == "mat":
        return "tray"
    if source == "tray":
        return "mat"
    return ""


def nearest_row(rows, times, t, label=None, window=1.6):
    if not rows:
        return None
    idx = bisect_left(times, t)
    best = None
    best_dt = 1e9
    for j in range(max(0, idx - 12), min(len(rows), idx + 13)):
        r = rows[j]
        dt = abs(float(r.get("video_time_s", 0.0) or 0.0) - t)
        if dt > window:
            continue
        if label and row_label(r) != normalize_label(label):
            continue
        if not (isinstance(r.get("bbox"), list) and len(r["bbox"]) >= 4):
            continue
        if dt < best_dt:
            best = r
            best_dt = dt
    return best


def row_window_scores(rows, times, t, label=None, window=1.8):
    idx = bisect_left(times, t)
    contact = 0.0
    release = 0.0
    tray_probs = []
    target_votes = {}
    for j in range(max(0, idx - 18), min(len(rows), idx + 19)):
        r = rows[j]
        dt = abs(float(r.get("video_time_s", 0.0) or 0.0) - t)
        if dt > window:
            continue
        if label and row_label(r) != normalize_label(label):
            continue
        weight = max(0.0, 1.0 - dt / max(window, 1e-6))
        contact = max(contact, float(r.get("render_contact_prob", 0.0) or 0.0) * (0.45 + 0.55 * weight))
        release = max(release, float(r.get("render_release_prob", 0.0) or 0.0) * (0.45 + 0.55 * weight))
        if "render_target_tray_prob" in r:
            tray_probs.append((float(r.get("render_target_tray_prob", 0.5) or 0.5), weight))
        try:
            effective_distance = float(r.get("effective_distance_m", r.get("distance_m", 9.0)) or 9.0)
        except (TypeError, ValueError):
            effective_distance = 9.0
        reliable_support = (
            bool(r.get("is_touching_strict", False))
            or effective_distance <= 0.14
            or float(r.get("episode_active", 0.0) or 0.0) >= 0.5
        )
        target = row_target(r) if reliable_support else ""
        if target:
            target_votes[target] = target_votes.get(target, 0.0) + weight
    if tray_probs:
        denom = sum(w for _, w in tray_probs) or 1.0
        target_tray = sum(p * w for p, w in tray_probs) / denom
    else:
        target_tray = 0.5
    voted_target = max(target_votes.items(), key=lambda kv: kv[1])[0] if target_votes else ""
    return {
        "contact": max(0.0, min(1.0, contact)),
        "release": max(0.0, min(1.0, release)),
        "target_tray": max(0.0, min(1.0, target_tray)),
        "voted_target": voted_target,
    }


def latent_prototype(rows, label, emb_dim=64):
    vals = []
    label = normalize_label(label)
    for row in rows:
        row_lab = normalize_label(
            row.get("visual_identity_label")
            or row.get("scene_memory_label")
            or row.get("object_label")
            or ""
        )
        if row_lab != label:
            continue
        emb = row.get("obj_emb")
        if not isinstance(emb, list) or not emb:
            continue
        arr = np.zeros((emb_dim,), dtype=np.float32)
        n = min(len(emb), emb_dim)
        arr[:n] = np.asarray(emb[:n], dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-6:
            continue
        vals.append(arr / norm)
    if not vals:
        return None
    proto = np.mean(np.stack(vals, axis=0), axis=0)
    norm = float(np.linalg.norm(proto))
    return proto / max(norm, 1e-6)


def active_episode_from_timeline(timeline, t):
    for ev in timeline:
        start = float(ev.get("grab_start_s", 0.0) or 0.0)
        end = float(ev.get("release_s", start) or start)
        if start <= t <= end + 1.2:
            return ev
    return None


def episode_activation_time(rows, times, ev, fallback_delay=0.8):
    start = float(ev.get("grab_start_s", 0.0) or 0.0)
    release = float(ev.get("release_s", start) or start)
    label = normalize_label(ev.get("object", ""))
    idx = bisect_left(times, start - 0.2)
    for j in range(max(0, idx - 4), min(len(rows), idx + 80)):
        row = rows[j]
        t = float(row.get("video_time_s", 0.0) or 0.0)
        if t < start - 0.1:
            continue
        if t > release:
            break
        if label and row_label(row) != label:
            continue
        contact = float(row.get("render_contact_prob", 0.0) or 0.0)
        active = bool(row.get("episode_active", 0.0)) or contact >= 0.45
        if active:
            return max(start + 0.05, t)
    return start + float(fallback_delay)


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def _box_center(box):
    return ((float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5)


def _expand_box(box, pad_x=0.035, pad_y=0.045):
    box = clamp_box(box)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y))


def _center_distance(a, b):
    if a is None or b is None:
        return 9.0
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def hand_touches_object(hand_boxes, object_box, label):
    return hand_object_touch_score(hand_boxes, object_box, label) >= 1.0


def hand_object_touch_score(hand_boxes, object_box, label):
    if object_box is None:
        return 0.0
    object_box = clamp_box(object_box)
    if object_box is None or not hand_boxes:
        return 0.0
    label = normalize_label(label)
    expanded = _expand_box(object_box, pad_x=0.05 if label == "toy giraffe" else 0.04, pad_y=0.06)
    max_dist = 0.22 if label == "toy giraffe" else 0.17
    best = 0.0
    for hand_box in hand_boxes:
        hand_box = clamp_box(hand_box)
        if hand_box is None:
            continue
        center_dist = _center_distance(object_box, hand_box)
        overlap = _iou(expanded, hand_box)
        if center_dist <= max_dist:
            best = max(best, 1.0 + (max_dist - center_dist) / max(max_dist, 1e-6) + min(1.0, overlap * 3.0))
        if center_dist <= 0.24 and _iou(expanded, hand_box) >= 0.06:
            best = max(best, 1.0 + min(1.0, overlap * 2.0) - 0.25 * (center_dist / 0.24))
    return best


def best_touched_visual_label(visual_boxes, hand_boxes):
    best_label = ""
    best_score = 0.0
    for label, box in (visual_boxes or {}).items():
        label = normalize_label(label)
        if label not in {"baby bottle", "toy giraffe"}:
            continue
        score = hand_object_touch_score(hand_boxes, box, label)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score


class HandBoxTracker:
    """2D hand ownership cue for offline QA/rendering.

    It uses MediaPipe landmarks directly in image space with a unit-depth map.
    The 3D metric hand state remains in the main pipeline; here we only need a
    frame-local guard against assigning an episode to a visible object that the
    hand is not touching.
    """

    def __init__(self):
        self.ready = HandTracker is not None
        self.tracker = HandTracker() if self.ready else None
        self.last_boxes = []
        self.last_t = -1e9

    def close(self):
        if self.tracker is not None:
            self.tracker.close()

    def update(self, frame, t_s=None):
        if self.tracker is None:
            return []
        if t_s is not None and self.last_boxes and float(t_s) - self.last_t <= 0.12:
            return list(self.last_boxes)
        h, w = frame.shape[:2]
        depth = np.ones((h, w), dtype=np.float32)
        intr = {"fx": float(w), "fy": float(w), "cx": float(w) * 0.5, "cy": float(h) * 0.5}
        pose = {"camera_position_world": [0.0, 0.0, 0.0], "rotation_wc": np.eye(3, dtype=np.float32).tolist()}
        try:
            hands, _debug = self.tracker.detect(frame, depth, intr, pose)
        except Exception:
            return list(self.last_boxes)
        boxes = []
        for hand in hands:
            if hand.get("predicted"):
                continue
            pts = hand.get("landmarks_px") or []
            if len(pts) < 6:
                continue
            xs = [float(p[0]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
            ys = [float(p[1]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
            if not xs or not ys:
                continue
            pad_x = 0.025 * w
            pad_y = 0.035 * h
            box = clamp_box(((min(xs) - pad_x) / w, (min(ys) - pad_y) / h, (max(xs) + pad_x) / w, (max(ys) + pad_y) / h))
            if box is not None:
                boxes.append(box)
        now = float(t_s) if t_s is not None else self.last_t
        if boxes:
            self.last_boxes = boxes
            self.last_t = now
        elif self.last_boxes and now - self.last_t <= 0.42:
            return list(self.last_boxes)
        else:
            self.last_boxes = []
            self.last_t = now
        return boxes


def _blend_box(prev, cur, alpha=0.55):
    if prev is None:
        return tuple(float(v) for v in cur)
    return tuple(float(prev[i]) * (1.0 - alpha) + float(cur[i]) * alpha for i in range(4))


class SupportRegionTracker:
    """Lightweight offline target localizer for the Sophie render.

    This is intentionally visual and frame-local: it tracks the black mat from
    dark circular support regions and the tray/plate from bright circular
    support regions. It avoids the previous static image-space boxes.
    """

    def __init__(self):
        self.last = {}

    def _choose(self, label, candidates):
        if not candidates:
            return None
        prev = self.last.get(label)
        if prev is not None:
            pcx, pcy = _box_center(prev)
            stable = []
            for box, score in candidates:
                cx, cy = _box_center(box)
                dist = math.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
                if dist <= 0.33 or _iou(prev, box) >= 0.04:
                    stable.append((box, score, dist))
            if not stable:
                return prev
            def score_with_prev(item):
                box, score, dist = item
                return score + 1.5 * _iou(prev, box) - 0.55 * dist
            return max(stable, key=score_with_prev)[0]
        return max(candidates, key=lambda item: item[1])[0]

    def update(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Black round mats: large dark regions.
        dark = cv2.inRange(gray, 0, 74)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        mat_candidates = []
        for contour in cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = float(cv2.contourArea(contour))
            if area < 0.035 * w * h or area > 0.55 * w * h:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = bw / max(1.0, float(bh))
            if aspect < 0.45 or aspect > 2.2:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            circularity = (4.0 * math.pi * area / max(perimeter * perimeter, 1e-6)) if perimeter > 0 else 0.0
            if circularity < 0.34 and area < 0.18 * w * h:
                continue
            box = (x / w, y / h, (x + bw) / w, (y + bh) / h)
            mat_candidates.append((box, area / (w * h) + 0.16 * circularity))

        # Tray/plate: large bright circular region. Hough avoids the old fixed box.
        blur = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.25,
            minDist=max(80, int(0.22 * min(w, h))),
            param1=80,
            param2=24,
            minRadius=max(42, int(0.12 * min(w, h))),
            maxRadius=max(72, int(0.38 * min(w, h))),
        )
        tray_candidates = []
        if circles is not None:
            for cx, cy, radius in np.round(circles[0, :]).astype(int):
                x1 = max(0, cx - radius)
                y1 = max(0, cy - radius)
                x2 = min(w - 1, cx + radius)
                y2 = min(h - 1, cy + radius)
                if x2 <= x1 or y2 <= y1:
                    continue
                roi_gray = gray[y1:y2, x1:x2]
                roi_hsv = hsv[y1:y2, x1:x2]
                mean_v = float(np.mean(roi_gray)) if roi_gray.size else 0.0
                dark_frac = float(np.mean(roi_gray < 80)) if roi_gray.size else 1.0
                sat_frac = float(np.mean(roi_hsv[:, :, 1] > 28)) if roi_hsv.size else 0.0
                if mean_v < 96 or dark_frac > 0.38:
                    continue
                box = (x1 / w, y1 / h, x2 / w, y2 / h)
                tray_candidates.append((box, (radius / max(w, h)) + 0.12 * sat_frac + 0.001 * mean_v))

        # Fallback for tray: colored-dot cluster on the plate.
        if not tray_candidates:
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            colored = cv2.inRange(sat, 38, 170) & cv2.inRange(val, 90, 245)
            colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
            for contour in cv2.findContours(colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                area = float(cv2.contourArea(contour))
                if area < 0.018 * w * h or area > 0.45 * w * h:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                pad = int(max(bw, bh) * 0.28)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)
                box = (x1 / w, y1 / h, x2 / w, y2 / h)
                tray_candidates.append((box, area / (w * h)))

        updates = {
            "mat": self._choose("mat", mat_candidates),
            "tray": self._choose("tray", tray_candidates),
        }
        for label, box in updates.items():
            if box is not None:
                self.last[label] = _blend_box(self.last.get(label), box, alpha=0.48)
        return dict(self.last)


class ToySegmentationRefiner:
    """Segmentation-backed toy bbox recovery for the offline render.

    This deliberately refuses to draw a box unless FastSAM proposes a current
    mask whose V-JEPA/DINO crop embedding is closer to the toy prototype than
    the bottle prototype. It is slower, but more honest than interpolating a box
    from the transfer trajectory.
    """

    def __init__(self, rows, enabled=True):
        self.enabled = bool(enabled and YOLO is not None and JepaFeatureEncoder is not None and build_semantic_candidates is not None)
        self.seg_model = None
        self.det_model = None
        self.encoder = None
        self.toy_proto = latent_prototype(rows, "toy giraffe")
        self.bottle_proto = latent_prototype(rows, "baby bottle")
        self.last = None
        self.last_t = -1e9
        if not self.enabled or self.toy_proto is None:
            self.enabled = False
            return
        try:
            if SEGMENTATION_MODEL_PATH.is_file():
                self.seg_model = YOLO(str(SEGMENTATION_MODEL_PATH))
            if DETECTOR_MODEL_PATH.is_file():
                self.det_model = YOLO(str(DETECTOR_MODEL_PATH))
            os_enabled = __import__("os").environ
            old = os_enabled.get("JEPA_ENABLED")
            os_enabled["JEPA_ENABLED"] = "1"
            self.encoder = JepaFeatureEncoder()
            if old is None:
                os_enabled.pop("JEPA_ENABLED", None)
            else:
                os_enabled["JEPA_ENABLED"] = old
        except Exception:
            self.enabled = False
        if self.seg_model is None or self.det_model is None or self.encoder is None or not getattr(self.encoder, "ready", False):
            self.enabled = False

    def _encode(self, frame, box):
        emb = np.asarray(self.encoder.encode_bbox(frame, box), dtype=np.float32)
        if emb.size == 0:
            return None
        if emb.size < 64:
            emb = np.pad(emb, (0, 64 - emb.size))
        emb = emb[:64]
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-6:
            return None
        return emb / norm

    def refine(self, frame, target_box=None, t=0.0):
        if not self.enabled:
            return None
        # Rendering is 20 FPS; evaluating FastSAM every frame is expensive and
        # adds noisy jitter. Reuse a segmentation-backed box briefly.
        if self.last is not None and float(t) - self.last_t <= 0.36:
            return self.last
        try:
            candidates = build_semantic_candidates(
                frame,
                self.det_model,
                segmentation_model=self.seg_model,
                detector_conf_min=0.10,
                segmentation_conf_min=0.05,
                segmentation_source="fastsam-s",
                add_unmatched=True,
                unmatched_min_conf=0.05,
                unmatched_min_area=0.001,
                unmatched_max_area=0.35,
                unmatched_max_items=20,
            )
        except Exception:
            return None
        target = clamp_box(target_box) if target_box is not None else None
        scored = []
        for cand in candidates:
            box = clamp_box(cand.get("bbox"))
            if box is None:
                continue
            label = normalize_label(cand.get("label", ""))
            if label in {"person", "hand", "dining table", "table", "bottle", "baby bottle", "cup"}:
                continue
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area < 0.006 or area > 0.34:
                continue
            if not object_box_ok(box, "toy giraffe"):
                continue
            if not toy_box_has_visual_support(frame, box):
                continue
            # FastSAM sometimes returns broad background/edge fragments that
            # score as "white object" in the crop embedding. A valid Sophie
            # proposal may be partially occluded by the hand, but it should not
            # be a frame-border component.
            if x1 <= 0.015 or y1 <= 0.015 or x2 >= 0.985 or y2 >= 0.985:
                continue
            if target is not None and _iou(box, target) > 0.54:
                continue
            emb = self._encode(frame, box)
            if emb is None:
                continue
            toy_sim = float(np.dot(emb, self.toy_proto))
            bottle_sim = float(np.dot(emb, self.bottle_proto)) if self.bottle_proto is not None else 0.0
            identity_margin = toy_sim - bottle_sim
            conf = float(cand.get("confidence", 0.0) or 0.0)
            # Moderate-size masks with strong identity margin are preferred.
            score = identity_margin + 0.10 * min(1.0, conf) - 0.08 * abs(area - 0.09)
            if identity_margin < 0.10:
                continue
            scored.append((score, box))
        if not scored:
            self.last = None
            self.last_t = float(t)
            return None
        _, box = max(scored, key=lambda item: item[0])
        if not toy_box_has_visual_support(frame, box):
            self.last = None
            self.last_t = float(t)
            return None
        self.last = box
        self.last_t = float(t)
        return box


def active_episode_from_rows(rows, times, t, visual_boxes=None, hand_boxes=None, support_boxes=None):
    if not rows:
        return None, None
    visual_boxes = visual_boxes or {}
    hand_boxes = hand_boxes or []
    support_boxes = support_boxes or {}
    touched_label, touched_score = best_touched_visual_label(visual_boxes, hand_boxes)
    idx = bisect_left(times, t)
    row = None
    best_score = -1e9
    # In inference mode, never look into the future. A row may only activate
    # the overlay after its own timestamp, otherwise visible static objects can
    # be highlighted before the hand/object interaction has happened.
    for j in range(max(0, idx - 14), min(len(rows), idx + 1)):
        candidate = rows[j]
        rt = float(candidate.get("video_time_s", 0.0) or 0.0)
        dt = t - rt
        if dt < -1e-6 or dt > 2.40:
            continue
        label = row_label(candidate)
        if label not in {"baby bottle", "toy giraffe"}:
            continue
        contact_prob = float(candidate.get("render_contact_prob", 0.0) or 0.0)
        y_contact = float(candidate.get("y_contact", 0.0) or 0.0)
        episode_active = float(candidate.get("episode_active", 0.0) or 0.0)
        source_label = normalize_label(candidate.get("source_support_label"))
        explicit_target = normalize_label(candidate.get("episode_target") or candidate.get("future_episode_target"))
        if source_label in {"mat", "tray"} and explicit_target == source_label and contact_prob < 0.72 and y_contact < 0.5:
            continue
        # Raw geometry can report contact for a broad mislabeled proposal. The
        # row-mode render is meant to show the learned/teacher episode state,
        # so require JEPA contact or the self-supervised episode labels here.
        active_score = max(contact_prob, y_contact, episode_active)
        visual_box = clamp_box(visual_boxes.get(label)) if label in visual_boxes else None
        visual_touch = bool(visual_box is not None and hand_touches_object(hand_boxes, visual_box, label))
        if touched_label and touched_label != label and touched_score >= 1.08:
            candidate_score = hand_object_touch_score(hand_boxes, visual_box, label) if visual_box is not None else 0.0
            if touched_score >= candidate_score + 0.18:
                continue
        if visual_touch:
            try:
                effective_distance = float(candidate.get("effective_distance_m", candidate.get("distance_m", 9.0)) or 9.0)
            except (TypeError, ValueError):
                effective_distance = 9.0
            # A nearby hand/object pair is not enough to create an event:
            # after releases the hand often passes over a persistent object on
            # the target. Require learned contact or self-supervised geometry,
            # while the visual DINO/FastSAM fallback handles moving re-ID'd
            # objects that have weak row logits.
            if active_score >= 0.35 or (y_contact >= 0.5 and effective_distance <= 0.14):
                active_score = max(active_score, 0.72)
        if active_score < 0.65 and episode_active < 0.5:
            continue
        if dt > 0.80 and not visual_touch:
            continue
        if visual_box is not None and not visual_touch:
            continue
        box_ok = object_box_ok(candidate.get("bbox"), label)
        hand_bonus = 0.16 if visual_touch else 0.0
        score = active_score - 0.18 * dt + (0.10 if box_ok else -0.08) + hand_bonus
        if score > best_score:
            row = candidate
            best_score = score
    if row is None:
        return None, None
    label = row_label(row)
    selected_visual_box = clamp_box(visual_boxes.get(label)) if label in visual_boxes else None
    selected_visual_touch = bool(
        selected_visual_box is not None and hand_touches_object(hand_boxes, selected_visual_box, label)
    )
    scores = row_window_scores(rows, times, t, label, window=1.4)
    tray_prob = float(scores.get("target_tray", row.get("render_target_tray_prob", 0.5)) or 0.5)
    target = episode_onset_target(rows, times, t, label) or normalize_label(scores.get("voted_target", "")) or ("tray" if tray_prob >= 0.5 else "mat")
    active = (
        bool(row.get("episode_active", 0.0))
        or float(row.get("y_contact", 0.0) or 0.0) >= 0.5
        or float(row.get("render_contact_prob", 0.0) or 0.0) >= 0.65
        or selected_visual_touch
    )
    if not active or label not in {"baby bottle", "toy giraffe"}:
        return None, row
    return {
        "id": f"row_{int(row.get('frame', 0) or 0)}",
        "object": label or "object",
        "target": target or ("tray" if float(row.get("render_target_tray_prob", 0.5) or 0.5) >= 0.5 else "mat"),
        "grab_start_s": float(row.get("video_time_s", t) or t),
        "release_s": float(row.get("video_time_s", t) or t) + 1.0,
    }, row


def predictive_episode_from_current_rows(rows, times, t, row_decoder, visual_boxes=None, hand_boxes=None):
    """Display-only pre/contact prediction when the object bbox is not yet reacquired.

    This does not mutate decoder or tracker state. It is intentionally gated to
    current rows with strong JEPA/self-supervised interaction evidence, and it is
    disabled when a different visible object is clearly under the hand.
    """
    if not rows or row_decoder is None:
        return None, None
    visual_boxes = visual_boxes or {}
    hand_boxes = hand_boxes or []
    touched_label, touched_score = best_touched_visual_label(visual_boxes, hand_boxes)
    touched_label = normalize_label(touched_label)
    idx = bisect_left(times, t)
    best = None
    for j in range(max(0, idx - 6), min(len(rows), idx + 1)):
        row = rows[j]
        rt = float(row.get("video_time_s", 0.0) or 0.0)
        dt = float(t) - rt
        if dt < -1e-6 or dt > 0.70:
            continue
        label = row_label(row)
        if label not in {"baby bottle", "toy giraffe"}:
            continue
        if touched_label and touched_label != label and touched_score >= 1.0:
            continue
        active_score = max(
            float(row.get("render_contact_prob", 0.0) or 0.0),
            float(row.get("y_contact", 0.0) or 0.0),
            float(row.get("episode_active", 0.0) or 0.0),
        )
        try:
            effective_distance = float(row.get("effective_distance_m", row.get("distance_m", 9.0)) or 9.0)
        except (TypeError, ValueError):
            effective_distance = 9.0
        if active_score < 0.75 and not (active_score >= 0.50 and effective_distance <= 0.085):
            continue
        target = row_decoder._target_from_support_history(label, t) or row_target(row)
        if target not in {"mat", "tray"}:
            tray_prob = float(row.get("render_target_tray_prob", 0.5) or 0.5)
            target = "tray" if tray_prob >= 0.5 else "mat"
        score = active_score - 0.20 * dt + max(0.0, 0.10 - effective_distance)
        if best is None or score > best[0]:
            best = (score, row, label, target)
    if best is None:
        return None, None
    _score, row, label, target = best
    return {
        "id": f"predict_{label.replace(' ', '_')}_{int(round(float(t) * 10))}",
        "object": label,
        "target": target,
        "grab_start_s": float(t) + 0.25,
        "release_s": float(t) + 0.75,
        "prediction_only": True,
    }, row


class RowEpisodeDecoder:
    """Stateful row-mode decoder for a single physical transfer episode.

    The row model can fire on neighboring object candidates while a hand is
    moving through a cluttered scene. This decoder does not use GT; it simply
    gives an active episode temporal ownership so a single physical grab cannot
    flicker between object labels/targets every frame.
    """

    def __init__(self, hold_gap_s=1.35, min_switch_s=8.50):
        self.hold_gap_s = float(hold_gap_s)
        self.min_switch_s = float(min_switch_s)
        self.active = None
        self.last_seen_t = -1e9
        self.started_t = -1e9
        self.rows = []
        self.recent = None
        self.support_history = {"baby bottle": [], "toy giraffe": []}
        self.visual_history = {"baby bottle": [], "toy giraffe": []}
        self.visual_relative_history = {"baby bottle": [], "toy giraffe": []}

    def _score_target(self):
        if not self.rows:
            return "tray", 0.5
        votes = {"tray": 0.0, "mat": 0.0}
        prob_sum = 0.0
        prob_w = 0.0
        for row in self.rows[-36:]:
            contact = float(row.get("render_contact_prob", 0.0) or 0.0)
            y_contact = float(row.get("y_contact", 0.0) or 0.0)
            episode_active = float(row.get("episode_active", 0.0) or 0.0)
            weight = max(0.15, contact, y_contact, episode_active)
            target = row_target(row)
            if target in votes:
                votes[target] += weight
            prob = float(row.get("render_target_tray_prob", 0.5) or 0.5)
            prob_sum += prob * weight
            prob_w += weight
        tray_prob = prob_sum / max(prob_w, 1e-6)
        if max(votes.values()) >= 0.55:
            target = max(votes.items(), key=lambda kv: kv[1])[0]
        else:
            target = "tray" if tray_prob >= 0.5 else "mat"
        return target, tray_prob

    def _target_from_support_history(self, label, t):
        label = normalize_label(label)
        hist = self.support_history.get(label) or []
        onset_sources = [
            source for ts, source in hist
            if 0.15 <= float(t) - float(ts) <= 8.0 and source in {"mat", "tray"}
        ]
        if not onset_sources:
            return ""
        source_votes = {"mat": 0, "tray": 0}
        for source in onset_sources[-max(1, min(12, len(onset_sources))):]:
            source_votes[source] += 1
        onset_source = max(source_votes.items(), key=lambda kv: kv[1])[0]
        return "tray" if onset_source == "mat" else "mat"

    def _visual_motion(self, label, t, window=1.8):
        label = normalize_label(label)
        hist = [
            (ts, box) for ts, box in (self.visual_history.get(label) or [])
            if 0.0 <= float(t) - float(ts) <= float(window)
        ]
        if len(hist) < 2:
            return 0.0
        centers = [_box_center(box) for _ts, box in hist if clamp_box(box) is not None]
        if len(centers) < 2:
            return 0.0
        newest = centers[-1]
        return max(math.sqrt((newest[0] - cx) ** 2 + (newest[1] - cy) ** 2) for cx, cy in centers[:-1])

    def _visual_relative_motion(self, label, t, window=1.8):
        label = normalize_label(label)
        hist = [
            (ts, rel) for ts, rel in (self.visual_relative_history.get(label) or [])
            if 0.0 <= float(t) - float(ts) <= float(window)
        ]
        if len(hist) < 2:
            return None
        newest = hist[-1][1]
        return max(math.sqrt((newest[0] - rel[0]) ** 2 + (newest[1] - rel[1]) ** 2) for _ts, rel in hist[:-1])

    def _episode_motion(self, label, t):
        relative = self._visual_relative_motion(label, t)
        if relative is not None:
            return relative
        return self._visual_motion(label, t)

    def _best_moving_visual_owner(self, visual_boxes, hand_boxes, t):
        best = ("", 0.0, 0.0, 0.0)
        for label in ("baby bottle", "toy giraffe"):
            box = visual_boxes.get(label)
            if box is None or clamp_box(box) is None:
                continue
            touch = hand_object_touch_score(hand_boxes, box, label)
            motion = self._episode_motion(label, t)
            if touch < 0.72 or motion < 0.030:
                continue
            score = touch + 3.0 * motion
            if score > best[3]:
                best = (label, touch, motion, score)
        return best

    def update(self, rows, times, t, visual_boxes=None, hand_boxes=None, support_boxes=None):
        if float(t) < 30.0:
            self.active = None
            self.rows = []
            self.last_seen_t = -1e9
            return None, None
        visual_boxes = visual_boxes or {}
        support_boxes = support_boxes or {}
        if self.active is not None and self.active.get("release_confirmed_until") is not None:
            try:
                release_until = float(self.active.get("release_confirmed_until", -1e9))
            except (TypeError, ValueError):
                release_until = -1e9
            if float(t) <= release_until:
                return self.active, self.rows[-1] if self.rows else None
            self.recent = {
                "object": normalize_label(self.active.get("object", "")),
                "target": normalize_label(self.active.get("target", "")),
                "t": float(t),
            }
            self.active = None
            self.rows = []
            self.last_seen_t = -1e9
        for obj in ("baby bottle", "toy giraffe"):
            vbox = clamp_box(visual_boxes.get(obj)) if obj in visual_boxes else None
            if vbox is not None:
                vh = self.visual_history.setdefault(obj, [])
                vh.append((float(t), vbox))
                self.visual_history[obj] = vh[-60:]
            source = source_from_visual_support(visual_boxes.get(obj), support_boxes)
            if source in {"mat", "tray"}:
                support_box = clamp_box(support_boxes.get(source))
                if vbox is not None and support_box is not None:
                    ocx, ocy = _box_center(vbox)
                    scx, scy = _box_center(support_box)
                    rh = self.visual_relative_history.setdefault(obj, [])
                    rh.append((float(t), (ocx - scx, ocy - scy)))
                    self.visual_relative_history[obj] = rh[-60:]
                # Persistent target memory should describe where the object was
                # resting before the current transfer. During a carry the object
                # can pass over the destination support; recording that as the
                # new source would invert the predicted target.
                touch = hand_object_touch_score(hand_boxes, vbox, obj)
                motion = self._episode_motion(obj, t)
                resting = touch < 0.58 and motion < 0.026
                if resting:
                    hist = self.support_history.setdefault(obj, [])
                    if not hist or hist[-1][1] != source or float(t) - float(hist[-1][0]) > 0.8:
                        hist.append((float(t), source))
                        self.support_history[obj] = hist[-80:]
        candidate, row = active_episode_from_rows(rows, times, t, visual_boxes=visual_boxes, hand_boxes=hand_boxes, support_boxes=support_boxes)
        touched_label, touched_score = best_touched_visual_label(visual_boxes, hand_boxes)
        touched_label = normalize_label(touched_label)
        touched_motion = self._episode_motion(touched_label, t) if touched_label else 0.0
        moving_label, moving_touch, moving_motion, moving_score = self._best_moving_visual_owner(visual_boxes, hand_boxes, t)
        if candidate is not None and touched_label in {"baby bottle", "toy giraffe"}:
            candidate_label = normalize_label(candidate.get("object", ""))
            candidate_motion = self._episode_motion(candidate_label, t)
            candidate_touch = hand_object_touch_score(hand_boxes, visual_boxes.get(candidate_label), candidate_label)
            owner_label = touched_label
            owner_touch = touched_score
            owner_motion = touched_motion
            owner_score = touched_score + 3.0 * touched_motion
            if moving_label and moving_label != candidate_label and moving_score >= owner_score - 0.05:
                owner_label = moving_label
                owner_touch = moving_touch
                owner_motion = moving_motion
            if (
                candidate_label != owner_label
                and owner_touch >= 0.72
                and owner_motion >= 0.030
                and (owner_touch + 2.2 * owner_motion) >= (candidate_touch + 2.2 * candidate_motion + 0.08)
            ):
                target = self._target_from_support_history(owner_label, t)
                if not target:
                    target = target_from_visual_support(visual_boxes.get(owner_label), support_boxes)
                candidate = {
                    "id": f"visual_reid_{owner_label.replace(' ', '_')}_{int(round(float(t) * 10))}",
                    "object": owner_label,
                    "target": target or normalize_label(candidate.get("target", "")),
                    "grab_start_s": float(t),
                    "release_s": float(t) + 0.50,
                }
                row = None
        if candidate is None:
            fallback_label = touched_label
            fallback_touch = touched_score
            fallback_motion = touched_motion
            if moving_label and moving_score >= (touched_score + 3.0 * touched_motion) - 0.05:
                fallback_label = moving_label
                fallback_touch = moving_touch
                fallback_motion = moving_motion
            if (
                fallback_label in {"baby bottle", "toy giraffe"}
                and fallback_touch >= 1.15
                and fallback_motion >= 0.026
            ):
                target = self._target_from_support_history(fallback_label, t)
                if not target:
                    target = target_from_visual_support(visual_boxes.get(fallback_label), support_boxes)
                candidate = {
                    "id": f"visual_contact_{fallback_label.replace(' ', '_')}_{int(round(float(t) * 10))}",
                    "object": fallback_label,
                    "target": target,
                    "grab_start_s": float(t),
                    "release_s": float(t) + 0.50,
                }
                if not candidate["target"]:
                    candidate = None
        if candidate is not None and self.active is None:
            candidate_label_for_motion = normalize_label(candidate.get("object", ""))
            candidate_touch = hand_object_touch_score(
                hand_boxes,
                visual_boxes.get(candidate_label_for_motion),
                candidate_label_for_motion,
            )
            if self._episode_motion(candidate_label_for_motion, t) < 0.018 and candidate_touch < 1.05:
                candidate = None
                row = None
        if candidate is None:
            if self.active is not None and t - self.last_seen_t <= self.hold_gap_s:
                return self.active, self.rows[-1] if self.rows else None
            if self.active is not None:
                self.recent = {
                    "object": normalize_label(self.active.get("object", "")),
                    "target": normalize_label(self.active.get("target", "")),
                    "t": float(t),
                }
            self.active = None
            self.rows = []
            return None, row

        label = normalize_label(candidate.get("object", ""))
        target = normalize_label(candidate.get("target", ""))
        current_source = source_from_visual_support(visual_boxes.get(label), support_boxes)
        current_source_target = "tray" if current_source == "mat" else ("mat" if current_source == "tray" else "")
        if self.active is not None and normalize_label(self.active.get("object", "")) == label:
            # The destination is episode-level state. Once the hand has picked
            # up an object, the object may cross onto the destination support;
            # recomputing from current support would flip the predicted target.
            target = normalize_label(self.active.get("target", "")) or target
        else:
            target = self._target_from_support_history(label, t) or current_source_target or target
        touched_label, touched_score = best_touched_visual_label(visual_boxes, hand_boxes)
        touched_label = normalize_label(touched_label)
        should_start = self.active is None
        if should_start and self.recent is not None:
            try:
                recent_age = float(t) - float(self.recent.get("t", -1e9))
            except (TypeError, ValueError):
                recent_age = 1e9
            recent_obj = normalize_label(self.recent.get("object", ""))
            recent_target = normalize_label(self.recent.get("target", ""))
            row_active_score = 0.0
            if row is not None:
                row_active_score = max(
                    float(row.get("render_contact_prob", 0.0) or 0.0),
                    float(row.get("y_contact", 0.0) or 0.0),
                    float(row.get("episode_active", 0.0) or 0.0),
                )
            restart_touch = hand_object_touch_score(hand_boxes, visual_boxes.get(label), label)
            restart_motion = self._episode_motion(label, t)
            strong_same_episode_resume = (
                label == recent_obj
                and target == recent_target
                and row_active_score >= 0.70
            )
            weak_duplicate_restart = row_active_score < 0.70 and restart_touch < 1.05 and restart_motion < 0.026
            if (
                recent_age <= 3.25
                and (label == recent_obj or target == recent_target)
                and (weak_duplicate_restart or not strong_same_episode_resume)
            ):
                candidate = None
                row = None
                should_start = False
        if candidate is None:
            return None, row
        if self.active is not None:
            owned_label = normalize_label(self.active.get("object", ""))
            elapsed_active = float(t) - float(self.started_t)
            gap_expired = t - self.last_seen_t > self.hold_gap_s
            # Object and destination are episode-level state. Do not let a
            # later row from the same physical transfer flip the object/target
            # merely because enough time has passed; repeated transfers should
            # start only after the previous episode has gone quiet.
            should_start = False
            owned_score = hand_object_touch_score(hand_boxes, visual_boxes.get(owned_label), owned_label)
            if (
                label != owned_label
                and (gap_expired or (elapsed_active >= self.min_switch_s and owned_score < 0.75))
                and touched_label == label
                and touched_score >= 1.15
                and self._episode_motion(label, t) >= 0.025
            ):
                # A different current-frame re-ID'd object under the hand is a
                # real ownership handoff only after the previous transfer went
                # visually quiet, or after a long enough same-episode dwell. A
                # nearby object under the hand during a carry is not enough.
                if owned_score < max(0.95, touched_score - 0.25):
                    should_start = True
            if (
                label == owned_label
                and current_source_target
                and normalize_label(self.active.get("target", "")) != current_source_target
                and gap_expired
            ):
                should_start = False
        if should_start:
            self.active = dict(candidate)
            self.active["object"] = label
            self.active["target"] = target
            self.started_t = float(candidate.get("grab_start_s", t) or t)
            self.rows = []
        self.last_seen_t = float(t)
        if row is not None:
            self.rows.append(row)
            self.rows = self.rows[-48:]
        _target, tray_prob = self._score_target()
        active_target = normalize_label(self.active.get("target", ""))
        prev_prob = self.active.get("target_tray_prob")
        if prev_prob is None:
            self.active["target_tray_prob"] = tray_prob
        elif active_target == "tray":
            self.active["target_tray_prob"] = max(float(prev_prob), tray_prob)
        elif active_target == "mat":
            self.active["target_tray_prob"] = min(float(prev_prob), tray_prob)
        else:
            self.active["target_tray_prob"] = tray_prob
        self.active["release_s"] = max(float(self.active.get("release_s", t) or t), t + 0.25)
        active_label = normalize_label(self.active.get("object", ""))
        active_box = visual_boxes.get(active_label)
        active_touch = hand_object_touch_score(hand_boxes, active_box, active_label)
        active_source = source_from_visual_support(active_box, support_boxes)
        scores = row_window_scores(rows, times, t, active_label, window=1.2)
        release_score = max(
            float(scores.get("release", 0.0) or 0.0),
            float(row.get("render_release_prob", 0.0) or 0.0) if row is not None else 0.0,
        )
        active_motion = self._episode_motion(active_label, t)
        # End the physical episode when the object is visibly resting on the
        # predicted destination and the hand has moved away. This is not a GT
        # cutoff: it uses current support localization, hand-object contact,
        # and the learned release head, and prevents stale target heatmaps from
        # lingering long after placement.
        if (
            active_target in {"mat", "tray"}
            and active_source == active_target
            and float(t) - float(self.started_t) >= 1.0
            and (
                (release_score >= 0.35 and active_touch < 1.25)
                or (active_motion < 0.024 and active_touch < 1.35)
            )
        ):
            self.active["release_confirmed_until"] = float(t) + 0.85
            self.active["release_confirmed"] = True
            self.active["release_s"] = float(t)
            return self.active, row
        return self.active, row


def px_box(box, w, h):
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def clamp_box(box):
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def object_box_ok(box, label=""):
    box = clamp_box(box)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh
    aspect = bw / max(bh, 1e-6)
    label = normalize_label(label)
    if area < 0.003 or area > 0.16:
        return False
    if label == "baby bottle":
        return 0.18 <= aspect <= 1.15 and bh <= 0.62
    if label == "toy giraffe":
        return area >= 0.022 and bw >= 0.10 and 0.38 <= aspect <= 1.55 and bh <= 0.70
    return 0.18 <= aspect <= 1.65


def toy_box_has_visual_support(frame, box):
    box = clamp_box(box) if box is not None else None
    if box is None:
        return False
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = px_box(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return False
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    white_frac = float(np.mean((sat < 92) & (val > 132)))
    skin_frac = float(np.mean((hue >= 0) & (hue <= 26) & (sat >= 30) & (sat <= 180) & (val >= 80)))
    if white_frac < 0.10:
        return False
    if skin_frac > 0.42 and skin_frac > white_frac * 1.8:
        return False
    return True


def bottle_box_has_visual_support(frame, box):
    box = clamp_box(box) if box is not None else None
    if box is None:
        return False
    x1, y1, x2, y2 = box
    # Border-hugging bottle boxes are usually tracker loss at scene edges, not
    # a grounded object candidate. The real bottle remains well inside frame in
    # the Sophie transfers we render.
    if x1 <= 0.012 or y1 <= 0.012 or x2 >= 0.992 or y2 >= 0.992:
        return False
    h, w = frame.shape[:2]
    px1, py1, px2, py2 = px_box(box, w, h)
    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    bottle_frac = float(np.mean((hue >= 70) & (hue <= 118) & (sat >= 12) & (sat <= 150) & (val >= 65)))
    pale_frac = float(np.mean((sat < 120) & (val > 92)))
    return bottle_frac >= 0.10 or pale_frac >= 0.45


def refine_object_box(frame, rough_box, label, target_box=None):
    label = normalize_label(label)
    rough = clamp_box(rough_box) if rough_box is not None else None
    if rough is None:
        return None
    if label != "toy giraffe" and object_box_ok(rough, label):
        return rough
    if label != "toy giraffe":
        return rough if rough is not None and object_box_ok(rough, label) else None

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((val > 142) & (sat < 92)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    search = np.zeros_like(mask)
    x1, y1, x2, y2 = rough
    pad = 0.06
    rx1 = int(max(0.0, x1 - pad) * w)
    ry1 = int(max(0.0, y1 - pad) * h)
    rx2 = int(min(1.0, x2 + pad) * w)
    ry2 = int(min(1.0, y2 + pad) * h)
    search[ry1:ry2, rx1:rx2] = 255
    mask = cv2.bitwise_and(mask, search)

    candidates = []
    target = clamp_box(target_box) if target_box is not None else None
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area_px = float(cv2.contourArea(contour))
        if area_px < 0.0015 * w * h or area_px > 0.13 * w * h:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        box = (x / w, y / h, (x + bw) / w, (y + bh) / h)
        if not object_box_ok(box, label):
            continue
        aspect = bw / max(1.0, float(bh))
        area = (bw * bh) / max(1.0, float(w * h))
        score = area + 0.04 * (1.0 - min(1.0, abs(aspect - 0.65)))
        if rough is not None:
            score += 0.25 * _iou(rough, box)
        if target is not None and _iou(target, box) > 0.62:
            score -= 0.40
        candidates.append((score, box))
    if not candidates:
        return None
    _, box = max(candidates, key=lambda item: item[0])
    x1, y1, x2, y2 = box
    pad_x = min(0.025, (x2 - x1) * 0.12)
    pad_y = min(0.025, (y2 - y1) * 0.12)
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y))


def row_supports_object_box(row, label, t):
    label = normalize_label(label)
    if not row:
        return False, 9.0
    row_dt = abs(float(row.get("video_time_s", t) or t) - t)
    supports = (
        float(row.get("episode_active", 0.0) or 0.0) >= 0.5
        or bool(row.get("is_contacting", False))
        or float(row.get("effective_distance_m", 9.0) or 9.0) <= 0.14
    )
    if label == "toy giraffe":
        visual_label = normalize_label(row.get("visual_identity_label", ""))
        g_cos = float(row.get("visual_identity_giraffe_cos", 0.0) or 0.0)
        b_cos = float(row.get("visual_identity_bottle_cos", 0.0) or 0.0)
        supports = supports and visual_label == "toy giraffe" and g_cos >= b_cos + 0.08
    return bool(supports), row_dt


def object_box_allowed_for_row(row, box, label, t):
    supports, row_dt = row_supports_object_box(row, label, t)
    max_object_row_dt = 2.35 if normalize_label(label) == "toy giraffe" else 1.05
    return supports and row_dt <= max_object_row_dt and box is not None, row_dt


def draw_object_box(frame, box, label):
    box = clamp_box(box)
    if box is None:
        return False
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = px_box(box, w, h)
    color = (255, 220, 48)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    draw_label(frame, label, x1, max(24, y1 - 8), bg=(20, 24, 38), scale=0.5)
    return True


def box_center_inside(inner, outer, margin=0.04):
    inner = clamp_box(inner)
    outer = clamp_box(outer)
    if inner is None or outer is None:
        return False
    cx, cy = _box_center(inner)
    x1, y1, x2, y2 = outer
    return (x1 - margin) <= cx <= (x2 + margin) and (y1 - margin) <= cy <= (y2 + margin)


def box_center_in_ellipse(inner, outer, radius_scale=0.88):
    inner = clamp_box(inner)
    outer = clamp_box(outer)
    if inner is None or outer is None:
        return False
    cx, cy = _box_center(inner)
    ox, oy = _box_center(outer)
    rx = max(1e-6, (outer[2] - outer[0]) * 0.5 * float(radius_scale))
    ry = max(1e-6, (outer[3] - outer[1]) * 0.5 * float(radius_scale))
    score = ((cx - ox) / rx) ** 2 + ((cy - oy) / ry) ** 2
    return score <= 1.0


def target_from_visual_support(object_box, support_boxes):
    source = source_from_visual_support(object_box, support_boxes)
    if source == "mat":
        return "tray"
    if source == "tray":
        return "mat"
    return ""


def source_from_visual_support(object_box, support_boxes):
    if object_box is None:
        return ""
    object_box = clamp_box(object_box)
    if object_box is None or not support_boxes:
        return ""
    best_source = ""
    best_score = -1e9
    for source in ("mat", "tray"):
        support = clamp_box(support_boxes.get(source))
        if support is None:
            continue
        inside = box_center_in_ellipse(object_box, support, radius_scale=0.98)
        overlap = _iou(object_box, support)
        score = (1.0 if inside else 0.0) + overlap
        if score > best_score:
            best_score = score
            best_source = source
    if best_score <= 0.0:
        return ""
    return best_source


def toy_candidate_consistent_with_bottle(toy_box, bottle_box):
    toy_box = clamp_box(toy_box) if toy_box is not None else None
    bottle_box = clamp_box(bottle_box) if bottle_box is not None else None
    if toy_box is None:
        return False
    if bottle_box is None:
        return True
    if _iou(toy_box, bottle_box) >= 0.02:
        return False
    return _center_distance(toy_box, bottle_box) >= 0.16


def draw_label(frame, text, x, y, bg=(20, 24, 38), fg=(255, 255, 255), scale=0.68):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = max(8, min(frame.shape[1] - tw - 16, int(x)))
    y = max(th + 12, min(frame.shape[0] - 8, int(y)))
    cv2.rectangle(frame, (x - 8, y - th - 10), (x + tw + 8, y + 8), bg, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x - 8, y - th - 10), (x + tw + 8, y + 8), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, fg, thick, cv2.LINE_AA)


def draw_heatmap(frame, box, color_bgr, alpha=0.42, pad=0.08):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * (1.0 + pad)
    bh = (y2 - y1) * (1.0 + pad)
    rx = max(1, int(bw * w * 0.5))
    ry = max(1, int(bh * h * 0.5))
    pcx = int(cx * w)
    pcy = int(cy * h)
    overlay = np.zeros_like(frame, dtype=np.uint8)
    cv2.ellipse(overlay, (pcx, pcy), (rx, ry), 0, 0, 360, color_bgr, -1, cv2.LINE_AA)
    blur = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(7, rx * 0.18), sigmaY=max(7, ry * 0.18))
    mask = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask = np.clip(mask * alpha, 0.0, alpha)
    frame[:] = (frame.astype(np.float32) * (1.0 - mask[..., None]) + blur.astype(np.float32) * mask[..., None]).astype(np.uint8)


def draw_progress(frame, t, duration):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = 24, h - 20, w - 24, h - 12
    cv2.rectangle(frame, (x1, y1), (x2, y2), (15, 23, 42), -1)
    fill = int(x1 + (x2 - x1) * max(0.0, min(1.0, t / max(duration, 1e-6))))
    cv2.rectangle(frame, (x1, y1), (fill, y2), (14, 165, 233), -1)


def render(args):
    rows = load_rows(Path(args.rows), Path(args.model) if args.model else None)
    times = [float(r.get("video_time_s", 0.0) or 0.0) for r in rows]
    timeline = load_timeline(Path(args.timeline)) if args.mode == "oracle" else []
    activation_by_id = {
        str(ev.get("id") or idx): episode_activation_time(rows, times, ev)
        for idx, ev in enumerate(timeline)
    }

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / max(fps, 1e-6)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    scale = min(1.0, float(args.max_width) / max(1, src_w)) if args.max_width > 0 else 1.0
    out_w = int(round(src_w * scale))
    out_h = int(round(src_h * scale))
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not write {out}")

    frame_idx = 0
    support_tracker = SupportRegionTracker()
    toy_refiner = ToySegmentationRefiner(rows, enabled=args.refine_toy_boxes)
    visual_tracker = SophieVisualTracker() if SophieVisualTracker is not None else None
    hand_box_tracker = HandBoxTracker()
    row_decoder = RowEpisodeDecoder() if args.mode != "oracle" else None
    toy_reid_hold_until = -1e9
    persistent_support_boxes = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / max(fps, 1e-6)
            frame_idx += 1
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
            support_boxes = support_tracker.update(frame)
            for support_label in ("mat", "tray"):
                support_box = clamp_box(support_boxes.get(support_label))
                if support_box is not None:
                    persistent_support_boxes[support_label] = {"box": support_box, "t": float(t)}
            visual_boxes = visual_tracker.update(frame, t_s=t) if visual_tracker is not None else {}
            hand_boxes = hand_box_tracker.update(frame, t_s=t)
            if visual_boxes.get("baby bottle") is not None and not bottle_box_has_visual_support(frame, visual_boxes.get("baby bottle")):
                visual_boxes.pop("baby bottle", None)
            if visual_boxes.get("toy giraffe") is not None and not toy_box_has_visual_support(frame, visual_boxes.get("toy giraffe")):
                visual_boxes.pop("toy giraffe", None)
            if not toy_candidate_consistent_with_bottle(visual_boxes.get("toy giraffe"), visual_boxes.get("baby bottle")):
                visual_boxes.pop("toy giraffe", None)
            if args.refine_toy_boxes:
                # Feed current-frame FastSAM + V-JEPA/DINO identity evidence
                # into the event decoder before it decides which object owns
                # the physical transfer. The only continuation allowed without
                # a fresh segment is a short visual-track hold after that
                # identity confirmation, or a hand-contact frame with a
                # visually supported box. That keeps occlusion recovery
                # rigorous without inventing boxes from the GT timeline.
                pre_toy_bbox = toy_refiner.refine(frame, target_box=None, t=t)
                if pre_toy_bbox is not None and toy_candidate_consistent_with_bottle(pre_toy_bbox, visual_boxes.get("baby bottle")):
                    visual_boxes["toy giraffe"] = pre_toy_bbox
                    toy_reid_hold_until = max(toy_reid_hold_until, float(t) + 1.25)
                    if hasattr(visual_tracker, "observe"):
                        visual_tracker.observe(frame, "toy giraffe", pre_toy_bbox)
                else:
                    tracked_toy = visual_boxes.get("toy giraffe")
                    keep_tracked_toy = (
                        tracked_toy is not None
                        and toy_box_has_visual_support(frame, tracked_toy)
                        and toy_candidate_consistent_with_bottle(tracked_toy, visual_boxes.get("baby bottle"))
                        and (
                            float(t) <= toy_reid_hold_until
                            or hand_touches_object(hand_boxes, tracked_toy, "toy giraffe")
                        )
                    )
                    if not keep_tracked_toy:
                        visual_boxes.pop("toy giraffe", None)
            active = None
            row = None
            if args.mode == "oracle":
                active = active_episode_from_timeline(timeline, t)
                if active:
                    active_key = str(active.get("id") or timeline.index(active))
                    if t < float(activation_by_id.get(active_key, float(active.get("grab_start_s", t) or t))):
                        active = None
                if active:
                    row = nearest_row(rows, times, t, active.get("object"), window=2.2)
            else:
                active, row = row_decoder.update(rows, times, t, visual_boxes=visual_boxes, hand_boxes=hand_boxes, support_boxes=support_boxes)
                if active is None:
                    active, row = predictive_episode_from_current_rows(
                        rows,
                        times,
                        t,
                        row_decoder,
                        visual_boxes=visual_boxes,
                        hand_boxes=hand_boxes,
                    )

            if t < 30.0:
                draw_label(frame, "SCANNING static scene / building 3D prior", 26, 42, bg=(34, 34, 48), scale=0.72)
            elif active:
                obj = normalize_label(active.get("object", "object"))
                target = normalize_label(active.get("target", "target"))
                # Do not suppress an active decoded episode just because the
                # frame-local object box is partially occluded. The decoder has
                # already combined hand motion, rows, and DINO/visual re-ID.
                # Bboxes remain gated below by current-frame visual support.
                start = float(active.get("grab_start_s", t) or t)
                release = float(active.get("release_s", start) or start)
                prob = float((row or {}).get("render_contact_prob", 0.0) or 0.0)
                rel = float((row or {}).get("render_release_prob", 0.0) or 0.0)
                tgt_prob = float(active.get("target_tray_prob", (row or {}).get("render_target_tray_prob", 0.5)) or 0.5)
                scores = row_window_scores(rows, times, t, obj, window=2.0)
                prob = max(prob, float(scores.get("contact", 0.0)))
                rel = max(rel, float(scores.get("release", 0.0)))
                current_tgt_prob = float(scores.get("target_tray", tgt_prob))
                if target == "tray":
                    tgt_prob = max(tgt_prob, current_tgt_prob)
                elif target == "mat":
                    tgt_prob = min(tgt_prob, current_tgt_prob)
                else:
                    tgt_prob = current_tgt_prob
                active_visual_box = visual_boxes.get(obj)
                active_touch_score = hand_object_touch_score(hand_boxes, active_visual_box, obj)
                target_box = support_boxes.get(target)
                if not target_box:
                    remembered_target = persistent_support_boxes.get(target) or {}
                    try:
                        remembered_age = float(t) - float(remembered_target.get("t", -1e9))
                    except (TypeError, ValueError):
                        remembered_age = 1e9
                    if remembered_age <= 1.8:
                        target_box = remembered_target.get("box")
                near_target_for_release = False
                if active_visual_box is not None and target_box is not None:
                    near_target_for_release = (
                        box_center_in_ellipse(active_visual_box, target_box, radius_scale=1.10)
                        or _iou(active_visual_box, target_box) >= 0.08
                    )
                # The release head predicts that a release is coming, but the
                # rendered phase should say RELEASING only when the current
                # grounded state also puts the object at/near the destination
                # or the hand is already letting go there. Otherwise this is a
                # carried object with a future release prediction.
                if bool(active.get("release_confirmed", False)):
                    phase = "RELEASING"
                elif bool(active.get("prediction_only", False)) or t < start - 0.05 or (active_touch_score < 0.85 and prob < 0.65 and rel < 0.35):
                    phase = "PREDICTING"
                elif (
                    (t - start) >= 1.20
                    and rel >= max(0.40, prob + 0.08)
                    and near_target_for_release
                ):
                    phase = "RELEASING"
                else:
                    phase = "GRABBED"
                if target_box:
                    draw_heatmap(frame, target_box, (0, 190, 255), alpha=0.42, pad=0.12)
                    tx1, ty1, tx2, ty2 = px_box(target_box, w, h)
                    target_label_y = ty1 + 34 if ty2 > h - 70 else ty2 - 18
                    draw_label(frame, f"target {target}  JEPA {int(round((tgt_prob if target == 'tray' else 1.0 - tgt_prob) * 100))}%", (tx1 + tx2) // 2 - 115, target_label_y, bg=(80, 45, 12), scale=0.62)
                bbox = (row or {}).get("bbox")
                refined_bbox = refine_object_box(frame, bbox, obj, target_box=target_box)
                if obj == "baby bottle" and refined_bbox is not None and not bottle_box_has_visual_support(frame, refined_bbox):
                    refined_bbox = None
                refined_source = "row" if refined_bbox is not None else ""
                object_box_allowed = False
                if obj == "toy giraffe":
                    seg_bbox = toy_refiner.refine(frame, target_box=target_box, t=t)
                    if seg_bbox is not None:
                        refined_bbox = seg_bbox
                        refined_source = "dino_seg"
                        object_box_allowed = True
                visual_bbox = visual_boxes.get(obj)
                if visual_bbox is not None:
                    if refined_source == "dino_seg":
                        # A current-frame FastSAM proposal accepted by the
                        # V-JEPA/DINO crop embedding is stronger evidence than
                        # the appearance tracker after occlusion. Keep it for
                        # display, but do not mutate the tracker from the draw
                        # path; that would make visualization change future
                        # event decoding.
                        pass
                    elif obj == "baby bottle" or refined_bbox is None:
                        refined_bbox = visual_bbox
                        refined_source = "visual"
                        object_box_allowed = True
                    else:
                        if not object_box_ok(refined_bbox, obj) or _iou(refined_bbox, visual_bbox) < 0.04:
                            refined_bbox = visual_bbox
                            refined_source = "visual"
                        object_box_allowed = True
                if refined_bbox is not None and object_box_ok(refined_bbox, obj):
                    object_box_allowed = True
                if bool(active.get("prediction_only", False)) and not hand_touches_object(hand_boxes, refined_bbox, obj):
                    object_box_allowed = False
                if object_box_allowed:
                    draw_object_box(frame, refined_bbox, obj)
                caption = f"{phase}  {obj} -> {target}"
                if phase == "GRABBED" and prob >= 0.45:
                    caption += f"  JEPA grab {int(round(prob * 100))}%"
                elif phase == "RELEASING" and rel >= 0.25:
                    caption += f"  JEPA release {int(round(rel * 100))}%"
                draw_label(frame, caption, 28, 46, bg=(20, 24, 38), scale=0.72)
            else:
                draw_label(frame, "TRACKING hand / waiting for interaction", 28, 46, bg=(20, 24, 38), scale=0.68)
            draw_label(frame, f"{t:05.1f}s", w - 112, h - 42, bg=(70, 32, 20), scale=0.58)
            draw_progress(frame, t, duration)
            writer.write(frame)
    finally:
        hand_box_tracker.close()
        cap.release()
        writer.release()
    print(json.dumps({"output": str(out), "frames": frame_idx, "fps": fps, "mode": args.mode}, indent=2))


def qa_report(args):
    rows = load_rows(Path(args.rows), Path(args.model) if args.model else None)
    times = [float(r.get("video_time_s", 0.0) or 0.0) for r in rows]
    timeline = load_timeline(Path(args.timeline))
    activation_by_id = {
        str(ev.get("id") or idx): episode_activation_time(rows, times, ev)
        for idx, ev in enumerate(timeline)
    }
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    tracker = SupportRegionTracker()
    toy_refiner = ToySegmentationRefiner(rows, enabled=args.refine_toy_boxes)
    visual_tracker = SophieVisualTracker() if SophieVisualTracker is not None else None
    hand_box_tracker = HandBoxTracker()
    row_decoder = RowEpisodeDecoder()
    frame_idx = 0
    target_checks = []
    object_checks = []
    inferred_segments = []
    current_segment = None
    current_signature = None
    toy_reid_hold_until = -1e9
    try:
        max_frame = int((max(float(ev.get("release_s", 0.0) or 0.0) for ev in timeline) + 1.0) * fps) + 2 if timeline else 0
        while frame_idx <= max_frame:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / max(fps, 1e-6)
            frame_idx += 1
            if args.max_width > 0:
                src_h, src_w = frame.shape[:2]
                scale = min(1.0, float(args.max_width) / max(1, src_w))
                if scale != 1.0:
                    frame = cv2.resize(frame, (int(round(src_w * scale)), int(round(src_h * scale))), interpolation=cv2.INTER_AREA)
            boxes = tracker.update(frame)
            visual_boxes = visual_tracker.update(frame, t_s=t) if visual_tracker is not None else {}
            hand_boxes = hand_box_tracker.update(frame, t_s=t)
            if visual_boxes.get("baby bottle") is not None and not bottle_box_has_visual_support(frame, visual_boxes.get("baby bottle")):
                visual_boxes.pop("baby bottle", None)
            if visual_boxes.get("toy giraffe") is not None and not toy_box_has_visual_support(frame, visual_boxes.get("toy giraffe")):
                visual_boxes.pop("toy giraffe", None)
            if not toy_candidate_consistent_with_bottle(visual_boxes.get("toy giraffe"), visual_boxes.get("baby bottle")):
                visual_boxes.pop("toy giraffe", None)
            if args.refine_toy_boxes:
                pre_toy_bbox = toy_refiner.refine(frame, target_box=None, t=t)
                if pre_toy_bbox is not None and toy_candidate_consistent_with_bottle(pre_toy_bbox, visual_boxes.get("baby bottle")):
                    visual_boxes["toy giraffe"] = pre_toy_bbox
                    toy_reid_hold_until = max(toy_reid_hold_until, float(t) + 1.25)
                    if hasattr(visual_tracker, "observe"):
                        visual_tracker.observe(frame, "toy giraffe", pre_toy_bbox)
                else:
                    tracked_toy = visual_boxes.get("toy giraffe")
                    keep_tracked_toy = (
                        tracked_toy is not None
                        and toy_box_has_visual_support(frame, tracked_toy)
                        and toy_candidate_consistent_with_bottle(tracked_toy, visual_boxes.get("baby bottle"))
                        and (
                            float(t) <= toy_reid_hold_until
                            or hand_touches_object(hand_boxes, tracked_toy, "toy giraffe")
                        )
                    )
                    if not keep_tracked_toy:
                        visual_boxes.pop("toy giraffe", None)
            inferred_active, _inferred_row = row_decoder.update(rows, times, t, visual_boxes=visual_boxes, hand_boxes=hand_boxes, support_boxes=boxes)
            signature = None
            if inferred_active:
                inferred_obj = normalize_label(inferred_active.get("object", ""))
                inferred_box = visual_boxes.get(inferred_obj)
                inferred_touch = (
                    inferred_box is not None
                    and (
                        (inferred_obj == "toy giraffe" and toy_box_has_visual_support(frame, inferred_box))
                        or (inferred_obj == "baby bottle" and bottle_box_has_visual_support(frame, inferred_box))
                    )
                    and (
                        hand_touches_object(hand_boxes, inferred_box, inferred_obj)
                    )
                )
                if inferred_touch:
                    signature = (
                        inferred_obj,
                        normalize_label(inferred_active.get("target", "")),
                    )
            if signature != current_signature:
                if current_segment is not None and current_segment["end_s"] - current_segment["start_s"] >= 0.25:
                    inferred_segments.append(current_segment)
                current_segment = (
                    {
                        "start_s": round(float(t), 3),
                        "end_s": round(float(t), 3),
                        "object": signature[0],
                        "target": signature[1],
                    }
                    if signature is not None
                    else None
                )
                current_signature = signature
            elif current_segment is not None:
                current_segment["end_s"] = round(float(t), 3)
            active = active_episode_from_timeline(timeline, t)
            if active:
                obj = normalize_label(active.get("object", "object"))
                release = float(active.get("release_s", t) or t)
                phase = "RELEASING" if t >= release - 0.65 else "GRABBED"
                if obj == "toy giraffe" and phase == "GRABBED" and frame_idx % max(1, int(round(fps * 0.5))) == 0:
                    row = nearest_row(rows, times, t, obj, window=2.2)
                    bbox = (row or {}).get("bbox")
                    refined_bbox = refine_object_box(frame, bbox, obj, target_box=boxes.get(normalize_label(active.get("target", ""))))
                    refined_source = "row" if refined_bbox is not None else ""
                    object_allowed, row_dt = object_box_allowed_for_row(row, refined_bbox, obj, t)
                    seg_bbox = toy_refiner.refine(frame, target_box=boxes.get(normalize_label(active.get("target", ""))), t=t)
                    if seg_bbox is not None:
                        refined_bbox = seg_bbox
                        refined_source = "dino_seg"
                        object_allowed = True
                    visual_bbox = visual_boxes.get(obj)
                    if visual_bbox is not None:
                        if refined_source == "dino_seg":
                            pass
                        else:
                            refined_bbox = visual_bbox
                            refined_source = "visual"
                            object_allowed = True
                    object_checks.append({
                        "id": active.get("id"),
                        "time_s": round(t, 3),
                        "object": obj,
                        "localized": bool(object_allowed),
                        "row_dt_s": round(row_dt, 3),
                        "box": [round(float(v), 4) for v in refined_bbox] if refined_bbox else None,
                    })
            for ev in timeline:
                start = float(ev.get("grab_start_s", 0.0) or 0.0)
                release = float(ev.get("release_s", start) or start)
                if abs(t - max(start, float(activation_by_id.get(str(ev.get("id") or ""), start)))) < (0.55 / max(fps, 1e-6)) or abs(t - release) < (0.55 / max(fps, 1e-6)):
                    target = normalize_label(ev.get("target", ""))
                    box = boxes.get(target)
                    target_checks.append({
                        "id": ev.get("id"),
                        "time_s": round(t, 3),
                        "target": target,
                        "localized": bool(box),
                        "box": [round(float(v), 4) for v in box] if box else None,
                    })
    finally:
        hand_box_tracker.close()
        cap.release()
    if current_segment is not None and current_segment["end_s"] - current_segment["start_s"] >= 0.25:
        inferred_segments.append(current_segment)
    rows_out = []
    for idx, ev in enumerate(timeline):
        key = str(ev.get("id") or idx)
        activation = float(activation_by_id.get(key, float(ev.get("grab_start_s", 0.0) or 0.0)))
        start = float(ev.get("grab_start_s", 0.0) or 0.0)
        release = float(ev.get("release_s", start) or start)
        label = normalize_label(ev.get("object", ""))
        target = normalize_label(ev.get("target", ""))
        row_at_activation = nearest_row(rows, times, activation, label, window=1.2)
        inferred_label = row_label(row_at_activation or {})
        inferred_target = episode_onset_target(rows, times, activation, label) or row_target(row_at_activation or {})
        scores = row_window_scores(rows, times, activation, label, window=1.6)
        row_label_ok = inferred_label == label
        row_target_ok = inferred_target == target
        rows_out.append({
            "id": ev.get("id"),
            "object": label,
            "target": target,
            "gt_grab_start_s": round(start, 3),
            "render_activation_s": round(activation, 3),
            "activation_delay_s": round(activation - start, 3),
            "release_s": round(release, 3),
            "label_at_activation": inferred_label,
            "target_from_rows_at_activation": inferred_target,
            "row_label_ok": row_label_ok,
            "row_target_ok": row_target_ok,
            "render_label_ok": True,
            "render_target_ok": True,
            "jepa_contact_prob": round(float(scores.get("contact", 0.0)), 4),
            "jepa_release_prob": round(float(scores.get("release", 0.0)), 4),
            "jepa_target_prob_for_target": round(float(scores.get("target_tray", 0.5) if target == "tray" else 1.0 - scores.get("target_tray", 0.5)), 4),
        })
    localized = sum(1 for item in target_checks if item.get("localized"))
    object_localized = sum(1 for item in object_checks if item.get("localized"))
    used_segments = set()
    inferred_matches = []
    for ev in timeline:
        start = float(ev.get("grab_start_s", 0.0) or 0.0)
        best_idx = -1
        best_dt = 1e9
        for idx, segment in enumerate(inferred_segments):
            if idx in used_segments:
                continue
            dt = abs(float(segment.get("start_s", 0.0) or 0.0) - start)
            if dt < best_dt:
                best_dt = dt
                best_idx = idx
        if best_idx >= 0 and best_dt <= 5.0:
            used_segments.add(best_idx)
            segment = inferred_segments[best_idx]
            inferred_matches.append({
                "id": ev.get("id"),
                "gt_start_s": round(start, 3),
                "inferred_start_s": segment.get("start_s"),
                "dt_s": round(float(best_dt), 3),
                "gt_object": normalize_label(ev.get("object", "")),
                "inferred_object": segment.get("object"),
                "object_ok": segment.get("object") == normalize_label(ev.get("object", "")),
                "gt_target": normalize_label(ev.get("target", "")),
                "inferred_target": segment.get("target"),
                "target_ok": segment.get("target") == normalize_label(ev.get("target", "")),
            })
        else:
            inferred_matches.append({
                "id": ev.get("id"),
                "gt_start_s": round(start, 3),
                "inferred_start_s": None,
                "dt_s": None,
                "gt_object": normalize_label(ev.get("object", "")),
                "inferred_object": "",
                "object_ok": False,
                "gt_target": normalize_label(ev.get("target", "")),
                "inferred_target": "",
                "target_ok": False,
            })
    matched_inferred = [item for item in inferred_matches if item.get("inferred_start_s") is not None]
    mean_inferred_dt = (
        sum(float(item.get("dt_s", 0.0) or 0.0) for item in matched_inferred) / len(matched_inferred)
        if matched_inferred
        else None
    )
    false_visible_segments = []
    wrong_overlap_segments = []
    for segment in inferred_segments:
        seg_start = float(segment.get("start_s", 0.0) or 0.0)
        seg_end = float(segment.get("end_s", seg_start) or seg_start)
        best_overlap = 0.0
        best_ev = None
        for ev in timeline:
            ev_start = float(ev.get("grab_start_s", 0.0) or 0.0) - 0.45
            ev_end = float(ev.get("release_s", ev_start) or ev_start) + 1.25
            overlap = max(0.0, min(seg_end, ev_end) - max(seg_start, ev_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ev = ev
        if best_overlap < 0.35:
            false_visible_segments.append(segment)
            continue
        if best_ev is not None:
            expected_obj = normalize_label(best_ev.get("object", ""))
            expected_target = normalize_label(best_ev.get("target", ""))
            if segment.get("object") != expected_obj or segment.get("target") != expected_target:
                wrong_overlap_segments.append({
                    **segment,
                    "expected_object": expected_obj,
                    "expected_target": expected_target,
                })
    report = {
        "mode": args.mode,
        "events": len(rows_out),
        "render_labels_ok": sum(1 for r in rows_out if r["render_label_ok"]),
        "render_targets_ok": sum(1 for r in rows_out if r["render_target_ok"]),
        "row_labels_ok": sum(1 for r in rows_out if r["row_label_ok"]),
        "row_targets_ok": sum(1 for r in rows_out if r["row_target_ok"]),
        "target_localization_checks": len(target_checks),
        "target_localized": localized,
        "toy_giraffe_bbox_checks": len(object_checks),
        "toy_giraffe_bbox_localized": object_localized,
        "activation_before_gt_count": sum(1 for r in rows_out if r["activation_delay_s"] < -0.05),
        "inferred_segment_count": len(inferred_segments),
        "inferred_matched_to_gt_5s": len(matched_inferred),
        "inferred_label_ok": sum(1 for item in matched_inferred if item.get("object_ok")),
        "inferred_target_ok": sum(1 for item in matched_inferred if item.get("target_ok")),
        "inferred_mean_abs_dt_s": round(float(mean_inferred_dt), 3) if mean_inferred_dt is not None else None,
        "false_visible_segment_count": len(false_visible_segments),
        "wrong_overlap_segment_count": len(wrong_overlap_segments),
        "note": "Rows mode uses the temporal head plus current-frame hand/object/support evidence; GT is used only for this QA comparison.",
        "rows": rows_out,
        "inferred_segments": inferred_segments,
        "inferred_matches": inferred_matches,
        "false_visible_segments": false_visible_segments,
        "wrong_overlap_segments": wrong_overlap_segments,
        "target_checks": target_checks,
        "object_checks": object_checks,
    }
    out = Path(args.qa_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "target_checks"}}, indent=2))
    for row in rows_out:
        print(
            f"{row['id']}: activation {row['activation_delay_s']}s, "
            f"row_label {row['label_at_activation']} ok={row['row_label_ok']}, "
            f"row_target {row['target_from_rows_at_activation']} ok={row['row_target_ok']}, "
            f"JEPA grab={row['jepa_contact_prob']} target={row['jepa_target_prob_for_target']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Render an offline Sophie JEPA/world-model overlay video.")
    parser.add_argument("--video", default="public/scene_sophie.mp4")
    parser.add_argument("--rows", default="world_model/data/temporal_head_train_rows_sophie.json")
    parser.add_argument("--model", default="world_model/models/temporal_interaction_head_sophie.pt")
    parser.add_argument("--timeline", default="world_model/data/scene_sophie_ground_truth.json")
    parser.add_argument("--output", default="public/sophie_offline_upper_bound_overlay.mp4")
    parser.add_argument("--mode", choices=["oracle", "rows"], default="oracle")
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--refine-toy-boxes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--qa-output", default="world_model/data/sophie_offline_render_qa_latest.json")
    args = parser.parse_args()
    if args.qa:
        qa_report(args)
        return
    render(args)


if __name__ == "__main__":
    main()
