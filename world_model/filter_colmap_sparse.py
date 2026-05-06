"""Filter COLMAP sparse points into a cleaner target point cloud.

Uses robust quality gates:
- minimum track length
- maximum reprojection error
- optional distance-to-center outlier clipping
- optional voxel-density pruning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_points3d_txt(path: Path):
    points = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw or raw.startswith("#"):
                continue
            parts = raw.strip().split()
            if len(parts) < 9:
                continue
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            error = float(parts[7])
            track_len = max(0, (len(parts) - 8) // 2)
            points.append((x, y, z, r, g, b, error, track_len))
    return points


def _write_ascii_ply(path: Path, xyz_rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(xyz_rgb)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for row in xyz_rgb:
            handle.write(
                f"{float(row[0]):.6f} {float(row[1]):.6f} {float(row[2]):.6f} "
                f"{int(row[3])} {int(row[4])} {int(row[5])}\n"
            )


def _voxel_density_mask(points_xyz: np.ndarray, voxel_size: float, min_per_voxel: int) -> np.ndarray:
    if voxel_size <= 0.0 or min_per_voxel <= 1 or len(points_xyz) == 0:
        return np.ones((len(points_xyz),), dtype=bool)
    vox = np.floor(points_xyz / voxel_size).astype(np.int64)
    _, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    return counts[inv] >= min_per_voxel


def main():
    parser = argparse.ArgumentParser(description="Filter COLMAP sparse points into a cleaner PLY.")
    parser.add_argument(
        "--points3d-txt",
        default="world_model/data/colmap_scene/sparse_txt/points3D.txt",
        help="Input COLMAP points3D.txt",
    )
    parser.add_argument(
        "--out-ply",
        default="world_model/data/colmap_scene/sparse_points_filtered.ply",
        help="Filtered output PLY",
    )
    parser.add_argument(
        "--out-report",
        default="world_model/data/colmap_scene/sparse_points_filtered_report.json",
        help="Filter report JSON",
    )
    parser.add_argument("--min-track-len", type=int, default=4)
    parser.add_argument("--max-reproj-error", type=float, default=2.0)
    parser.add_argument(
        "--max-center-distance",
        type=float,
        default=8.0,
        help="Max Euclidean distance from median center in scene units (<=0 disables).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional voxel size for density pruning (<=0 disables).",
    )
    parser.add_argument(
        "--min-points-per-voxel",
        type=int,
        default=1,
        help="Min points in voxel when voxel filtering is enabled.",
    )
    args = parser.parse_args()

    in_path = Path(args.points3d_txt)
    points = _read_points3d_txt(in_path)
    if not points:
        raise RuntimeError(f"No points loaded from: {in_path}")

    arr = np.asarray(points, dtype=np.float64)
    xyz = arr[:, 0:3]
    rgb = np.clip(arr[:, 3:6], 0, 255).astype(np.uint8)
    errors = arr[:, 6]
    tracks = arr[:, 7]

    mask = (tracks >= float(args.min_track_len)) & (errors <= float(args.max_reproj_error))

    center = np.median(xyz, axis=0)
    dist = np.linalg.norm(xyz - center, axis=1)
    if args.max_center_distance > 0:
        mask &= dist <= float(args.max_center_distance)

    kept_xyz = xyz[mask]
    kept_rgb = rgb[mask]
    voxel_mask = _voxel_density_mask(
        kept_xyz,
        voxel_size=float(args.voxel_size),
        min_per_voxel=int(args.min_points_per_voxel),
    )
    kept_xyz = kept_xyz[voxel_mask]
    kept_rgb = kept_rgb[voxel_mask]

    out = np.concatenate([kept_xyz, kept_rgb.astype(np.float64)], axis=1)
    _write_ascii_ply(Path(args.out_ply), out)

    report = {
        "input_points3d_txt": str(in_path.resolve()),
        "output_ply": str(Path(args.out_ply).resolve()),
        "total_points_in": int(len(points)),
        "points_after_quality_filters": int(mask.sum()),
        "points_after_all_filters": int(len(out)),
        "params": {
            "min_track_len": int(args.min_track_len),
            "max_reproj_error": float(args.max_reproj_error),
            "max_center_distance": float(args.max_center_distance),
            "voxel_size": float(args.voxel_size),
            "min_points_per_voxel": int(args.min_points_per_voxel),
        },
        "stats": {
            "error_q50_q75_q90_q95": np.quantile(errors, [0.5, 0.75, 0.9, 0.95]).tolist(),
            "track_q50_q75_q90": np.quantile(tracks, [0.5, 0.75, 0.9]).tolist(),
            "dist_q50_q75_q90_q95_q99": np.quantile(dist, [0.5, 0.75, 0.9, 0.95, 0.99]).tolist(),
        },
    }
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
