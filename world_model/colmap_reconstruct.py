"""COLMAP-based offline reconstruction for a demo video.

This script extracts frames from a video, runs COLMAP sparse SfM and optional
dense MVS, then exports PLY point clouds and a JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2


def _find_colmap_binary() -> str:
    env_bin = os.environ.get("COLMAP_BIN")
    candidates = [env_bin, "colmap", "COLMAP.bat", "COLMAP"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            proc = subprocess.run(
                [candidate, "-h"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode in (0, 1):
                return candidate
        except FileNotFoundError:
            continue
    raise RuntimeError(
        "COLMAP executable not found. Install COLMAP and/or set COLMAP_BIN to its executable path."
    )


def _load_calibration(calibration_file: Path | None) -> dict | None:
    if calibration_file is None:
        return None
    if not calibration_file.exists():
        return None
    try:
        return json.loads(calibration_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _colmap_camera_params_from_calibration(calib: dict | None) -> str | None:
    if not calib:
        return None
    matrix = calib.get("camera_matrix")
    dist = calib.get("distortion_coefficients")
    if not matrix or len(matrix) != 3 or not dist or len(dist) < 4:
        return None
    fx = float(matrix[0][0])
    fy = float(matrix[1][1])
    cx = float(matrix[0][2])
    cy = float(matrix[1][2])
    k1 = float(dist[0])
    k2 = float(dist[1])
    p1 = float(dist[2])
    p2 = float(dist[3])
    return f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2}"


def _run(cmd: list[str], cwd: Path) -> tuple[int, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode, elapsed


def _extract_frames(video_path: Path, out_dir: Path, target_fps: float, max_frames: int) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0.0:
        src_fps = 30.0
    target_fps = max(0.1, float(target_fps))
    stride = max(1, int(round(src_fps / target_fps)))

    decoded = 0
    saved = 0
    width = 0
    height = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            height, width = frame.shape[:2]
            if (decoded - 1) % stride != 0:
                continue
            if max_frames > 0 and saved >= max_frames:
                break
            frame_name = f"frame_{saved:06d}.jpg"
            frame_path = out_dir / frame_name
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved += 1
    finally:
        cap.release()

    return {
        "decoded_frames": decoded,
        "saved_frames": saved,
        "width": width,
        "height": height,
        "source_fps": src_fps,
        "target_fps": target_fps,
        "stride": stride,
    }


def _pick_sparse_model_dir(sparse_root: Path) -> Path:
    candidates = [item for item in sparse_root.iterdir() if item.is_dir()]
    if not candidates:
        raise RuntimeError(f"No sparse model was generated in: {sparse_root}")
    candidates.sort(key=lambda p: p.name)
    return candidates[0]


def _model_analyzer(colmap_bin: str, model_dir: Path, cwd: Path) -> dict:
    cmd = [colmap_bin, "model_analyzer", "--path", str(model_dir)]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    result = {"registered_images": None, "points3d": None, "raw_output": proc.stdout}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.lower().startswith("registered images"):
            try:
                result["registered_images"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        if line.lower().startswith("points"):
            try:
                result["points3d"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
    return result


def run_colmap_pipeline(
    *,
    video_path: Path,
    workspace: Path,
    calibration_file: Path | None,
    target_fps: float,
    max_frames: int,
    matcher: str,
    dense: bool,
    clean: bool,
    use_gpu: bool,
) -> dict:
    colmap_bin = _find_colmap_binary()
    if clean and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    images_dir = workspace / "images"
    sparse_dir = workspace / "sparse"
    dense_dir = workspace / "dense"
    db_path = workspace / "database.db"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    extraction = _extract_frames(video_path, images_dir, target_fps=target_fps, max_frames=max_frames)
    if extraction["saved_frames"] < 12:
        raise RuntimeError("Too few frames extracted for COLMAP (need at least 12).")

    calib = _load_calibration(calibration_file)
    camera_params = _colmap_camera_params_from_calibration(calib)
    camera_model = "OPENCV" if camera_params else "SIMPLE_RADIAL"

    timings = {}
    gpu_flag = "1" if use_gpu else "0"

    feature_cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        camera_model,
        "--FeatureExtraction.use_gpu",
        gpu_flag,
    ]
    if camera_params:
        feature_cmd.extend(["--ImageReader.camera_params", camera_params])
    _, timings["feature_extractor_s"] = _run(feature_cmd, workspace)

    if matcher == "sequential":
        match_cmd = [
            colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(db_path),
            "--FeatureMatching.use_gpu",
            gpu_flag,
            "--SequentialMatching.overlap",
            "12",
        ]
    else:
        match_cmd = [
            colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            str(db_path),
            "--FeatureMatching.use_gpu",
            gpu_flag,
        ]
    _, timings["matcher_s"] = _run(match_cmd, workspace)

    mapper_cmd = [
        colmap_bin,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
        "--Mapper.ba_refine_focal_length",
        "0" if camera_params else "1",
        "--Mapper.ba_refine_principal_point",
        "0" if camera_params else "1",
        "--Mapper.ba_refine_extra_params",
        "0" if camera_params else "1",
    ]
    _, timings["mapper_s"] = _run(mapper_cmd, workspace)

    model_dir = _pick_sparse_model_dir(sparse_dir)
    sparse_ply = workspace / "sparse_points.ply"
    convert_cmd = [
        colmap_bin,
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(sparse_ply),
        "--output_type",
        "PLY",
    ]
    _, timings["model_converter_s"] = _run(convert_cmd, workspace)

    dense_ply = None
    if dense:
        undistort_cmd = [
            colmap_bin,
            "image_undistorter",
            "--image_path",
            str(images_dir),
            "--input_path",
            str(model_dir),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
        ]
        _, timings["image_undistorter_s"] = _run(undistort_cmd, workspace)

        patch_cmd = [
            colmap_bin,
            "patch_match_stereo",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
        ]
        _, timings["patch_match_stereo_s"] = _run(patch_cmd, workspace)

        dense_ply = dense_dir / "fused.ply"
        fusion_cmd = [
            colmap_bin,
            "stereo_fusion",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(dense_ply),
        ]
        _, timings["stereo_fusion_s"] = _run(fusion_cmd, workspace)

    model_stats = _model_analyzer(colmap_bin, model_dir, workspace)
    sparse_ply_points = None
    sparse_ply_size = sparse_ply.stat().st_size if sparse_ply.exists() else 0
    dense_ply_size = dense_ply.stat().st_size if dense_ply and dense_ply.exists() else 0

    report = {
        "video_path": str(video_path.resolve()),
        "workspace": str(workspace.resolve()),
        "colmap_bin": colmap_bin,
        "calibration_file": str(calibration_file.resolve()) if calibration_file else None,
        "camera_model": camera_model,
        "camera_params_used": camera_params,
        "frames": extraction,
        "matcher": matcher,
        "dense_enabled": dense,
        "sparse_model_dir": str(model_dir.resolve()),
        "sparse_ply": str(sparse_ply.resolve()) if sparse_ply.exists() else None,
        "dense_ply": str(dense_ply.resolve()) if dense_ply and dense_ply.exists() else None,
        "sparse_ply_size_bytes": sparse_ply_size,
        "dense_ply_size_bytes": dense_ply_size,
        "timings_seconds": {key: round(val, 3) for key, val in timings.items()},
        "model_stats": model_stats,
        "sparse_ply_points": sparse_ply_points,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="COLMAP reconstruction runner for a demo video.")
    parser.add_argument("--video", default="public/scene_sophie.mp4", help="Input video path.")
    parser.add_argument(
        "--workspace",
        default="world_model/data/colmap_scene",
        help="Workspace directory for extracted frames and COLMAP outputs.",
    )
    parser.add_argument(
        "--calibration-file",
        default="world_model/data/camera_calibration.json",
        help="Calibration JSON used to initialize COLMAP camera parameters.",
    )
    parser.add_argument("--target-fps", type=float, default=4.0, help="Frame sampling FPS from input video.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max sampled frames (0 = unlimited).")
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument("--dense", action="store_true", help="Run dense MVS and export fused dense PLY.")
    parser.add_argument("--clean", action="store_true", help="Delete previous workspace before running.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU in SIFT extraction/matching if available.")
    parser.add_argument(
        "--report-json",
        default="world_model/data/colmap_scene_report.json",
        help="Output report JSON path.",
    )
    args = parser.parse_args()

    calibration_file = Path(args.calibration_file) if args.calibration_file else None
    report = run_colmap_pipeline(
        video_path=Path(args.video),
        workspace=Path(args.workspace),
        calibration_file=calibration_file,
        target_fps=float(args.target_fps),
        max_frames=int(args.max_frames),
        matcher=str(args.matcher),
        dense=bool(args.dense),
        clean=bool(args.clean),
        use_gpu=bool(args.use_gpu),
    )

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
