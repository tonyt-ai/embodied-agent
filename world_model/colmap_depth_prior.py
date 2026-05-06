from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np


def _qvec_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    q /= n
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass
class _ImageAnchors:
    name: str
    anchors: list[tuple[float, float, float]]


class ColmapDepthPrior:
    """Align relative depth map to metric depth using COLMAP anchors.

    This is designed for replay/demo videos where a COLMAP sparse model was
    built from the same sequence.
    """

    def __init__(self, sparse_txt_dir: str, prior_fps: float = 3.0, runtime_fps: float = 5.0):
        self.enabled = False
        self.prior_fps = max(0.1, float(prior_fps))
        self.runtime_fps = max(0.1, float(runtime_fps))
        self.ema_alpha = float(os.environ.get("COLMAP_DEPTH_PRIOR_EMA_ALPHA", "0.25"))
        self.max_abs_a = float(os.environ.get("COLMAP_DEPTH_PRIOR_MAX_ABS_A", "30.0"))
        self.max_abs_b = float(os.environ.get("COLMAP_DEPTH_PRIOR_MAX_ABS_B", "30.0"))
        self.max_rmse = float(os.environ.get("COLMAP_DEPTH_PRIOR_MAX_RMSE_M", "0.6"))
        self.min_anchors_used = int(os.environ.get("COLMAP_DEPTH_PRIOR_MIN_ANCHORS", "18"))
        self.normalize_sfm_scale = os.environ.get("COLMAP_DEPTH_PRIOR_NORMALIZE_SFM_SCALE", "1").lower() in {"1", "true", "yes"}
        self.images: list[_ImageAnchors] = []
        self.last_fit: dict = {}
        self._a_ema: float | None = None
        self._b_ema: float | None = None
        self._fit_kind: str = "linear"
        self._sfm_scale_ema: float | None = None
        try:
            self.images = self._load_anchors(sparse_txt_dir)
            self.enabled = len(self.images) > 0
        except Exception:
            self.images = []
            self.enabled = False

    def _load_anchors(self, sparse_txt_dir: str) -> list[_ImageAnchors]:
        images_txt = os.path.join(sparse_txt_dir, "images.txt")
        points_txt = os.path.join(sparse_txt_dir, "points3D.txt")
        if not (os.path.isfile(images_txt) and os.path.isfile(points_txt)):
            return []

        world_pts: dict[int, np.ndarray] = {}
        with open(points_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals = line.split()
                if len(vals) < 4:
                    continue
                pid = int(vals[0])
                world_pts[pid] = np.asarray([float(vals[1]), float(vals[2]), float(vals[3])], dtype=np.float64)

        images: list[_ImageAnchors] = []
        with open(images_txt, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]

        i = 0
        while i + 1 < len(lines):
            if lines[i].startswith("#"):
                i += 1
                continue
            h = lines[i].split()
            if len(h) < 10:
                i += 1
                continue
            qw, qx, qy, qz = map(float, h[1:5])
            tx, ty, tz = map(float, h[5:8])
            name = h[9]
            obs = lines[i + 1].split()
            R = _qvec_to_rotmat(qw, qx, qy, qz)
            t = np.asarray([tx, ty, tz], dtype=np.float64)
            anchors: list[tuple[float, float, float]] = []
            for j in range(0, len(obs) - 2, 3):
                u = float(obs[j])
                v = float(obs[j + 1])
                pid = int(float(obs[j + 2]))
                if pid < 0 or pid not in world_pts:
                    continue
                Xw = world_pts[pid]
                Xc = R @ Xw + t
                z = float(Xc[2])
                if z <= 0.0 or not math.isfinite(z):
                    continue
                anchors.append((u, v, z))
            images.append(_ImageAnchors(name=name, anchors=anchors))
            i += 2
        return images

    def _image_index_for_frame(self, frame_idx: int, elapsed_s: float | None = None) -> int | None:
        if not self.images:
            return None
        if elapsed_s is not None and math.isfinite(float(elapsed_s)):
            t = max(0.0, float(elapsed_s))
        else:
            t = max(0.0, float(frame_idx) / self.runtime_fps)
        idx = int(round(t * self.prior_fps))
        if idx < 0 or idx >= len(self.images):
            return None
        return idx

    def _inside_dynamic_bbox(self, u: float, v: float, w: int, h: int, dynamic_bboxes: list[list[float]], margin: float) -> bool:
        if not dynamic_bboxes:
            return False
        nx = float(u) / max(1.0, float(w))
        ny = float(v) / max(1.0, float(h))
        for box in dynamic_bboxes:
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            x1 -= margin
            y1 -= margin
            x2 += margin
            y2 += margin
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return True
        return False

    def _robust_positive_fit(self, x: np.ndarray, y: np.ndarray, kind: str) -> dict | None:
        if x.size < 12 or y.size != x.size:
            return None
        if kind == "inverse":
            x_fit = 1.0 / np.clip(x, 1e-6, None)
        else:
            x_fit = x
        if not np.isfinite(x_fit).all() or float(np.std(x_fit)) <= 1e-8:
            return None

        A = np.vstack([x_fit, np.ones_like(x_fit)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        pred = a * x_fit + b
        err = np.abs(y - pred)
        med = float(np.median(err))
        mad = float(np.median(np.abs(err - med)))
        keep = err <= max(0.04, med + 3.0 * max(mad, 1e-6))
        if int(np.sum(keep)) >= 10:
            x2 = x_fit[keep]
            y2 = y[keep]
            A2 = np.vstack([x2, np.ones_like(x2)]).T
            sol2, *_ = np.linalg.lstsq(A2, y2, rcond=None)
            a, b = float(sol2[0]), float(sol2[1])
            pred2 = a * x2 + b
            rmse = float(np.sqrt(np.mean((y2 - pred2) ** 2)))
            anchors_used = int(np.sum(keep))
        else:
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            anchors_used = int(x.size)

        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(rmse)):
            return None
        if a <= 0.0:
            return None
        return {
            "kind": kind,
            "a": a,
            "b": b,
            "rmse": rmse,
            "anchors_used": anchors_used,
        }

    def _apply_fit(self, depth_map: np.ndarray, a: float, b: float, kind: str) -> np.ndarray:
        if kind == "inverse":
            aligned = a * (1.0 / np.clip(depth_map, 1e-6, None)) + b
        else:
            aligned = a * depth_map + b
        return np.clip(aligned, 0.01, 50.0).astype(np.float32)

    def align_depth(
        self,
        depth_map: np.ndarray,
        frame_idx: int,
        elapsed_s: float | None = None,
        dynamic_bboxes: list[list[float]] | None = None,
    ) -> tuple[np.ndarray, dict]:
        dbg = {"enabled": bool(self.enabled), "anchors_total": 0, "anchors_used": 0, "mode": "disabled"}
        if not self.enabled or depth_map is None or depth_map.size == 0:
            return depth_map, dbg

        h, w = depth_map.shape[:2]
        image_idx = self._image_index_for_frame(frame_idx, elapsed_s=elapsed_s)
        if image_idx is None:
            if self._a_ema is not None and self._b_ema is not None:
                aligned = self._apply_fit(depth_map, self._a_ema, self._b_ema, self._fit_kind)
                dbg["mode"] = "hold-last-fit"
                dbg["fit_kind"] = self._fit_kind
                dbg["a"] = round(float(self._a_ema), 5)
                dbg["b"] = round(float(self._b_ema), 5)
                return aligned, dbg
            dbg["mode"] = "out-of-prior-range"
            return depth_map, dbg
        image = self.images[image_idx]
        dbg["anchors_total"] = len(image.anchors)
        if len(image.anchors) < 12:
            dbg["mode"] = "insufficient-anchors"
            return depth_map, dbg

        d_rel = []
        z_met = []
        skipped_dynamic = 0
        bboxes = dynamic_bboxes or []
        margin = float(os.environ.get("COLMAP_DEPTH_PRIOR_DYNAMIC_MARGIN_NORM", "0.015"))
        for u, v, z in image.anchors:
            if self._inside_dynamic_bbox(u, v, w, h, bboxes, margin):
                skipped_dynamic += 1
                continue
            x = int(max(0, min(w - 1, round(u))))
            y = int(max(0, min(h - 1, round(v))))
            d = float(depth_map[y, x])
            if d > 0.0 and math.isfinite(d) and math.isfinite(z):
                d_rel.append(d)
                z_met.append(z)
        dbg["anchors_skipped_dynamic"] = int(skipped_dynamic)
        if len(d_rel) < 12:
            dbg["mode"] = "insufficient-samples"
            return depth_map, dbg

        x = np.asarray(d_rel, dtype=np.float64)
        y = np.asarray(z_met, dtype=np.float64)
        if self.normalize_sfm_scale:
            raw_med = float(np.median(y))
            depth_med = float(np.median(x))
            if raw_med > 1e-6 and depth_med > 1e-6 and np.isfinite(raw_med) and np.isfinite(depth_med):
                scale_now = depth_med / raw_med
                if self._sfm_scale_ema is None:
                    self._sfm_scale_ema = scale_now
                else:
                    alpha = min(1.0, max(0.01, self.ema_alpha))
                    self._sfm_scale_ema = (1.0 - alpha) * self._sfm_scale_ema + alpha * scale_now
                y = y * float(self._sfm_scale_ema)
                dbg["sfm_scale"] = round(float(self._sfm_scale_ema), 6)
                dbg["sfm_raw_median_z"] = round(raw_med, 5)
                dbg["depth_anchor_median"] = round(depth_med, 5)
        fits = [
            fit for fit in (
                self._robust_positive_fit(x, y, "linear"),
                self._robust_positive_fit(x, y, "inverse"),
            )
            if fit is not None
        ]
        if not fits:
            dbg["mode"] = "rejected-fit"
            dbg["reason"] = "no-positive-monotonic-fit"
            return depth_map, dbg
        fit = min(fits, key=lambda item: (float(item["rmse"]), -int(item["anchors_used"])))
        a = float(fit["a"])
        b = float(fit["b"])
        rmse = float(fit["rmse"])
        kind = str(fit["kind"])
        dbg["anchors_used"] = int(fit["anchors_used"])
        dbg["fit_kind"] = kind
        dbg["mode"] = "robust-fit"
        dbg["rmse"] = round(rmse, 5)
        if (
            dbg["anchors_used"] < max(8, self.min_anchors_used)
            or not np.isfinite(a)
            or not np.isfinite(b)
            or a <= 0.0
            or abs(a) > self.max_abs_a
            or abs(b) > self.max_abs_b
            or rmse > self.max_rmse
        ):
            dbg["mode"] = "rejected-fit"
            return depth_map, dbg

        if self._a_ema is None or self._b_ema is None:
            self._a_ema, self._b_ema = a, b
            self._fit_kind = kind
        else:
            if self._fit_kind != kind:
                self._a_ema, self._b_ema = a, b
                self._fit_kind = kind
                dbg["mode"] = "robust-fit-reset-kind"
            else:
                alpha = min(1.0, max(0.01, self.ema_alpha))
                self._a_ema = (1.0 - alpha) * self._a_ema + alpha * a
                self._b_ema = (1.0 - alpha) * self._b_ema + alpha * b
        a_use = float(self._a_ema)
        b_use = float(self._b_ema)

        aligned = self._apply_fit(depth_map, a_use, b_use, self._fit_kind)
        dbg["a"] = round(a_use, 5)
        dbg["b"] = round(b_use, 5)
        self.last_fit = dbg
        return aligned, dbg
