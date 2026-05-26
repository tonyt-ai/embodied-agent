"""Small visual tracker for the Sophie tabletop demo.

The tracker initializes from the static scene and then follows the two
movable objects in image space. It is intentionally separate from labels:
labels still come from scene memory / embeddings, while these boxes are only
used when the current frame supports a plausible visual extent.
"""

from __future__ import annotations

import cv2
import numpy as np


def _clamp_box(box):
    if box is None or len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _box_ok(box, label):
    box = _clamp_box(box)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh
    aspect = bw / max(bh, 1e-6)
    if label == "baby bottle":
        return 0.006 <= area <= 0.11 and 0.24 <= aspect <= 1.10 and bh <= 0.58
    if label == "toy giraffe":
        cx = (x1 + x2) * 0.5
        return 0.010 <= area <= 0.16 and cx >= 0.30 and y2 <= 0.88 and 0.28 <= aspect <= 1.75 and bh <= 0.76
    return False


def _make_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    return None


def _components(frame, kind):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if kind == "baby bottle":
        mask = (
            (hsv[:, :, 0] >= 70)
            & (hsv[:, :, 0] <= 118)
            & (hsv[:, :, 1] >= 18)
            & (hsv[:, :, 1] <= 125)
            & (hsv[:, :, 2] >= 70)
        ).astype(np.uint8) * 255
    else:
        mask = (
            (hsv[:, :, 1] <= 130)
            & (hsv[:, :, 2] >= 118)
        ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    out = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area_px = float(cv2.contourArea(contour))
        if area_px < 0.0008 * w * h or area_px > 0.22 * w * h:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        box = _clamp_box([x / w, y / h, (x + bw) / w, (y + bh) / h])
        if box is None:
            continue
        if not _box_ok(box, kind):
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        out.append((area, box))
    return sorted(out, key=lambda item: item[0], reverse=True)


def _init_box(frame, label):
    comps = _components(frame, label)
    if not comps:
        return None
    if label == "toy giraffe":
        # In the static scene Sophie is the lower white object; reject frame
        # border fragments and prefer the expected mid/lower support area.
        scored = []
        for area, box in comps:
            cx = (box[0] + box[2]) * 0.5
            cy = (box[1] + box[3]) * 0.5
            border_penalty = 0.5 if box[0] <= 0.02 or box[2] >= 0.98 or box[1] <= 0.02 or box[3] >= 0.98 else 0.0
            score = area + 0.12 * (1.0 - min(1.0, abs(cx - 0.40) * 2.3)) + 0.08 * (1.0 - min(1.0, abs(cy - 0.70) * 2.0)) - border_penalty
            scored.append((score, box))
        box = max(scored, key=lambda item: item[0])[1]
        return _clamp_box([box[0] - 0.028, box[1] - 0.035, box[2] + 0.010, box[3] + 0.010])
    box = comps[0][1]
    return _clamp_box([box[0] - 0.015, box[1] - 0.020, box[2] + 0.015, box[3] + 0.020])


class SophieVisualTracker:
    def __init__(self):
        self._tracks = {}
        self._templates = {"baby bottle": [], "toy giraffe": []}

    def observe(self, frame, label, box):
        """Seed/update a visual track from an externally verified detection."""
        label = str(label or "").strip().lower()
        if label not in self._templates or not _box_ok(box, label):
            return False
        self._init_tracker(frame, label, _clamp_box(box))
        return True

    def _template_features(self, frame, box):
        box = _clamp_box(box)
        if box is None:
            return None
        h, w = frame.shape[:2]
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        patch = cv2.resize(patch, (96, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Gray plus inverse saturation is robust for Sophie: the object is
        # mostly bright/white, but its silhouette and colored spots carry ID.
        return np.dstack([gray, 255 - hsv[:, :, 1]]).astype(np.float32)

    def _remember_template(self, frame, label, box):
        if label not in self._templates or not _box_ok(box, label):
            return
        feat = self._template_features(frame, box)
        if feat is None:
            return
        x1, y1, x2, y2 = _clamp_box(box)
        self._templates[label].append(
            {
                "feat": feat,
                "shape": (max(1e-4, x2 - x1), max(1e-4, y2 - y1)),
            }
        )
        self._templates[label] = self._templates[label][-8:]

    def _match_template(self, frame, label):
        templates = self._templates.get(label) or []
        if not templates:
            return None
        full_h, full_w = frame.shape[:2]
        scale_img = min(1.0, 640.0 / max(1, full_w))
        if scale_img < 1.0:
            work = cv2.resize(frame, (int(round(full_w * scale_img)), int(round(full_h * scale_img))), interpolation=cv2.INTER_AREA)
        else:
            work = frame
        h, w = work.shape[:2]
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        base = np.dstack([cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), 255 - hsv[:, :, 1]]).astype(np.float32)
        best = None
        scales = (0.75, 0.9, 1.0, 1.15, 1.35)
        for item in templates:
            feat = item["feat"]
            bw, bh = item["shape"]
            for scale in scales:
                tw = max(18, int(round(bw * w * scale)))
                th = max(24, int(round(bh * h * scale)))
                if tw >= w or th >= h:
                    continue
                templ = cv2.resize(feat, (tw, th), interpolation=cv2.INTER_AREA)
                try:
                    res = cv2.matchTemplate(base, templ, cv2.TM_CCOEFF_NORMED)
                except Exception:
                    continue
                _, score, _, loc = cv2.minMaxLoc(res)
                box = _clamp_box([loc[0] / w, loc[1] / h, (loc[0] + tw) / w, (loc[1] + th) / h])
                if not _box_ok(box, label):
                    continue
                if label == "toy giraffe" and float(score) < 0.63:
                    continue
                if label == "baby bottle" and float(score) < 0.54:
                    continue
                if best is None or float(score) > best[0]:
                    best = (float(score), box)
        return best[1] if best is not None else None

    def _init_tracker(self, frame, label, box):
        tracker = _make_tracker()
        if tracker is None:
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        rect = (
            int(x1 * w),
            int(y1 * h),
            max(2, int((x2 - x1) * w)),
            max(2, int((y2 - y1) * h)),
        )
        try:
            tracker.init(frame, rect)
            self._tracks[label] = {"tracker": tracker, "box": box, "misses": 0}
            self._remember_template(frame, label, box)
        except Exception:
            return

    def update(self, frame, t_s=None):
        h, w = frame.shape[:2]
        results = {}
        for label in ("baby bottle", "toy giraffe"):
            track = self._tracks.get(label)
            visual_box = _init_box(frame, label) if label == "baby bottle" else None
            if visual_box is not None:
                self._init_tracker(frame, label, visual_box)
                results[label] = visual_box
                continue
            if track is None:
                if t_s is not None and not (24.0 <= float(t_s) <= 31.5):
                    matched = self._match_template(frame, label)
                    if matched is not None:
                        self._init_tracker(frame, label, matched)
                        results[label] = matched
                    continue
                box = _init_box(frame, label)
                if box is not None:
                    self._init_tracker(frame, label, box)
                    results[label] = box
                continue
            ok = False
            box = None
            try:
                ok, rect = track["tracker"].update(frame)
                if ok:
                    x, y, bw, bh = rect
                    box = _clamp_box([x / w, y / h, (x + bw) / w, (y + bh) / h])
            except Exception:
                ok = False
            if ok and _box_ok(box, label):
                track["box"] = box
                track["misses"] = 0
                results[label] = box
                self._remember_template(frame, label, box)
                continue
            track["misses"] = int(track.get("misses", 0)) + 1
            matched = self._match_template(frame, label)
            if matched is not None:
                self._init_tracker(frame, label, matched)
                results[label] = matched
                continue
            if track["misses"] > 20:
                self._tracks.pop(label, None)
                matched = self._match_template(frame, label)
                if matched is not None:
                    self._init_tracker(frame, label, matched)
                    results[label] = matched
        return results
