from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class _VoxelStat:
    sum_xyz: np.ndarray
    count: int


class BackgroundDepthFusion:
    """Lightweight dynamic-masked background fusion (TSDF-lite point accumulation)."""

    def __init__(self):
        self.enabled = os.environ.get("DEPTH_FUSION_ENABLED", "1").lower() not in {"0", "false", "no"}
        self.voxel_size = float(os.environ.get("DEPTH_FUSION_VOXEL_SIZE_M", "0.025"))
        self.sample_stride = int(os.environ.get("DEPTH_FUSION_SAMPLE_STRIDE", "6"))
        self.max_points = int(os.environ.get("DEPTH_FUSION_MAX_POINTS", "12000"))
        self.max_depth = float(os.environ.get("DEPTH_FUSION_MAX_DEPTH_M", "8.0"))
        self.min_depth = float(os.environ.get("DEPTH_FUSION_MIN_DEPTH_M", "0.08"))
        self._voxels: dict[tuple[int, int, int], _VoxelStat] = {}

    def reset(self):
        self._voxels = {}

    def update(self, depth_map, intrinsics: dict, camera_pose: dict, dynamic_mask=None) -> dict:
        dbg = {"enabled": bool(self.enabled), "added_samples": 0, "voxels": len(self._voxels)}
        if not self.enabled or depth_map is None or depth_map.size == 0 or not camera_pose:
            return dbg
        h, w = depth_map.shape[:2]
        fx = float(intrinsics.get("fx", 1.0))
        fy = float(intrinsics.get("fy", 1.0))
        cx = float(intrinsics.get("cx", 0.0))
        cy = float(intrinsics.get("cy", 0.0))
        cam_w = np.asarray(camera_pose.get("camera_position_world", [0.0, 0.0, 0.0]), dtype=np.float32)
        Rwc = np.asarray(camera_pose.get("rotation_wc", np.eye(3, dtype=np.float32)), dtype=np.float32)
        if Rwc.shape != (3, 3):
            Rwc = np.eye(3, dtype=np.float32)

        step = max(2, int(self.sample_stride))
        for y in range(0, h, step):
            for x in range(0, w, step):
                if dynamic_mask is not None and dynamic_mask.size > 0:
                    if int(dynamic_mask[y, x]) > 0:
                        continue
                z = float(depth_map[y, x])
                if not math.isfinite(z) or z < self.min_depth or z > self.max_depth:
                    continue
                Xc = np.asarray([((x - cx) / max(fx, 1e-6)) * z, ((y - cy) / max(fy, 1e-6)) * z, z], dtype=np.float32)
                Xw = Rwc @ Xc + cam_w
                if not np.all(np.isfinite(Xw)):
                    continue
                k = tuple(int(math.floor(float(v) / self.voxel_size)) for v in Xw)
                cur = self._voxels.get(k)
                if cur is None:
                    self._voxels[k] = _VoxelStat(sum_xyz=Xw.astype(np.float64), count=1)
                else:
                    cur.sum_xyz += Xw.astype(np.float64)
                    cur.count += 1
                dbg["added_samples"] += 1

        if len(self._voxels) > self.max_points:
            # Keep denser/older-supported voxels.
            items = sorted(self._voxels.items(), key=lambda kv: kv[1].count, reverse=True)[: self.max_points]
            self._voxels = dict(items)
        dbg["voxels"] = len(self._voxels)
        return dbg

    def export_points(self):
        out = []
        for i, (_, st) in enumerate(self._voxels.items()):
            if st.count <= 0:
                continue
            p = st.sum_xyz / float(st.count)
            out.append(
                {
                    "id": f"fused_{i}",
                    "position_world": [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)],
                    "quality": min(1.0, 0.05 * math.log1p(st.count)),
                    "hits": int(st.count),
                    "status": "fused",
                    "is_local_map": False,
                }
            )
        return out

