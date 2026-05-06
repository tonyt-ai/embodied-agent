"""Compare two point clouds with normalized Chamfer distance.

The comparison is shape-focused:
- centers each cloud
- normalizes by RMS radius
- aligns with PCA (including sign-flip search)
- reports nearest-neighbor Chamfer-style error
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _load_ascii_ply_xyz(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "end_header":
            start = i + 1
            break
    pts = []
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            pts.append((x, y, z))
    return np.asarray(pts, dtype=np.float64)


def _normalize(points: np.ndarray) -> tuple[np.ndarray, dict]:
    center = np.median(points, axis=0)
    centered = points - center
    radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if radius <= 1e-9:
        radius = 1.0
    normed = centered / radius
    return normed, {"center": center.tolist(), "radius_rms": radius}


def _pca_basis(points: np.ndarray) -> np.ndarray:
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1.0
    return basis


def _directed_nn(a: np.ndarray, b: np.ndarray) -> float:
    tree = cKDTree(b)
    dists, _ = tree.query(a, k=1)
    return float(np.mean(dists))


def _chamfer(a: np.ndarray, b: np.ndarray) -> dict:
    a2b = _directed_nn(a, b)
    b2a = _directed_nn(b, a)
    return {"a_to_b": a2b, "b_to_a": b2a, "symmetric": 0.5 * (a2b + b2a)}


def compare_clouds(ref_pts: np.ndarray, test_pts: np.ndarray) -> dict:
    ref_n, ref_meta = _normalize(ref_pts)
    test_n, test_meta = _normalize(test_pts)

    ref_basis = _pca_basis(ref_n)
    test_basis = _pca_basis(test_n)
    ref_aligned = ref_n @ ref_basis
    test_aligned_base = test_n @ test_basis

    best = None
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                flip = np.array([sx, sy, sz], dtype=np.float64)
                candidate = test_aligned_base * flip.reshape(1, 3)
                metrics = _chamfer(ref_aligned, candidate)
                if best is None or metrics["symmetric"] < best["metrics"]["symmetric"]:
                    best = {"flip": [sx, sy, sz], "metrics": metrics}

    raw = _chamfer(ref_n, test_n)
    return {
        "reference_meta": ref_meta,
        "test_meta": test_meta,
        "raw_normalized_chamfer": raw,
        "pca_aligned_best": best,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare point clouds (normalized Chamfer).")
    parser.add_argument("--reference", required=True, help="Reference PLY (e.g., COLMAP sparse).")
    parser.add_argument("--test", required=True, help="Test PLY (e.g., SLAM export).")
    parser.add_argument("--out-json", default="", help="Optional path to save comparison JSON.")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    test_path = Path(args.test)
    ref_pts = _load_ascii_ply_xyz(ref_path)
    test_pts = _load_ascii_ply_xyz(test_path)
    if len(ref_pts) == 0 or len(test_pts) == 0:
        raise RuntimeError(f"Empty cloud(s): ref={len(ref_pts)}, test={len(test_pts)}")

    result = {
        "reference_path": str(ref_path.resolve()),
        "test_path": str(test_path.resolve()),
        "reference_points": int(len(ref_pts)),
        "test_points": int(len(test_pts)),
        "comparison": compare_clouds(ref_pts, test_pts),
    }
    blob = json.dumps(result, indent=2)
    print(blob)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(blob, encoding="utf-8")


if __name__ == "__main__":
    main()
