"""Latent 3D world-state scaffold for embodied reasoning.

This module provides:
1) tokenization helpers for sparse anchors, objects, hands, camera/depth stats
2) a compact Transformer encoder that maps tokens -> latent world state
3) explicit interaction heads for downstream tasks

It is designed as a drop-in prototype model and does not replace SLAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TokenBatch:
    """Fixed-size token tensors used by `WorldLatentModel`."""

    anchor_tokens: torch.Tensor  # [B, Na, Da]
    object_tokens: torch.Tensor  # [B, No, Do]
    hand_tokens: torch.Tensor  # [B, Nh, Dh]
    camera_tokens: torch.Tensor  # [B, Dc]
    depth_tokens: torch.Tensor  # [B, Dd]
    anchor_mask: torch.Tensor  # [B, Na] bool
    object_mask: torch.Tensor  # [B, No] bool
    hand_mask: torch.Tensor  # [B, Nh] bool


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(scalar):
        return float(default)
    return float(scalar)


def _first_finite_xyz(point: dict[str, Any]) -> np.ndarray | None:
    for key in ("position_world", "triangulated_position_world", "position_world_depth_prior"):
        value = point.get(key)
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            continue
        xyz = np.asarray(value[:3], dtype=np.float32)
        if np.isfinite(xyz).all():
            return xyz
    return None


def build_token_batch_from_world_state(
    *,
    camera_pose: dict[str, Any] | None,
    objects_3d: list[dict[str, Any]] | None,
    hands: list[dict[str, Any]] | None,
    max_anchors: int = 256,
    max_objects: int = 24,
    max_hands: int = 4,
    device: torch.device | str = "cpu",
) -> TokenBatch:
    """Create a single-sample token batch from world-model payload data."""

    camera_pose = camera_pose or {}
    objects_3d = objects_3d or []
    hands = hands or []

    anchor_dim = 12
    object_dim = 10
    hand_dim = 10
    camera_dim = 12
    depth_dim = 6

    anchor_tokens = np.zeros((1, max_anchors, anchor_dim), dtype=np.float32)
    object_tokens = np.zeros((1, max_objects, object_dim), dtype=np.float32)
    hand_tokens = np.zeros((1, max_hands, hand_dim), dtype=np.float32)
    anchor_mask = np.zeros((1, max_anchors), dtype=bool)
    object_mask = np.zeros((1, max_objects), dtype=bool)
    hand_mask = np.zeros((1, max_hands), dtype=bool)

    persistent = camera_pose.get("persistent_map", []) or []
    for idx, point in enumerate(persistent[:max_anchors]):
        xyz = _first_finite_xyz(point)
        if xyz is None:
            continue
        anchor_tokens[0, idx, 0:3] = xyz
        anchor_tokens[0, idx, 3] = _safe_float(point.get("quality", 0.0))
        anchor_tokens[0, idx, 4] = _safe_float(point.get("hits", 0.0))
        anchor_tokens[0, idx, 5] = 1.0 if str(point.get("status", "visible")) == "visible" else 0.0
        anchor_tokens[0, idx, 6] = 1.0 if bool(point.get("is_local_map", False)) else 0.0
        anchor_tokens[0, idx, 7] = 1.0 if bool(point.get("is_triangulated", False)) else 0.0
        anchor_tokens[0, idx, 8] = 1.0 if bool(point.get("is_geometry_verified", False)) else 0.0
        anchor_tokens[0, idx, 9] = _safe_float(point.get("mean_reprojection_error", 0.0))
        anchor_tokens[0, idx, 10] = _safe_float(point.get("dynamic_score", 0.0))
        anchor_tokens[0, idx, 11] = _safe_float(point.get("age", 0.0))
        anchor_mask[0, idx] = True

    for idx, obj in enumerate(objects_3d[:max_objects]):
        pos = obj.get("position_3d", [0.0, 0.0, 0.0])
        vel = obj.get("velocity_3d", [0.0, 0.0, 0.0])
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            pos = [0.0, 0.0, 0.0]
        if not isinstance(vel, (list, tuple)) or len(vel) < 3:
            vel = [0.0, 0.0, 0.0]
        object_tokens[0, idx, 0:3] = np.asarray(pos[:3], dtype=np.float32)
        object_tokens[0, idx, 3:6] = np.asarray(vel[:3], dtype=np.float32)
        object_tokens[0, idx, 6] = _safe_float(obj.get("depth", 0.0))
        object_tokens[0, idx, 7] = _safe_float(obj.get("depth_confidence", 0.0))
        object_tokens[0, idx, 8] = _safe_float(obj.get("landmark_support", 0.0))
        object_tokens[0, idx, 9] = _safe_float(obj.get("landmark_blend_weight", 0.0))
        object_mask[0, idx] = True

    for idx, hand in enumerate(hands[:max_hands]):
        center = hand.get("center_3d", hand.get("position_3d", [0.0, 0.0, 0.0]))
        if not isinstance(center, (list, tuple)) or len(center) < 3:
            center = [0.0, 0.0, 0.0]
        hand_tokens[0, idx, 0:3] = np.asarray(center[:3], dtype=np.float32)
        hand_tokens[0, idx, 3] = _safe_float(hand.get("confidence", 0.0))
        hand_tokens[0, idx, 4] = _safe_float(hand.get("openness", 0.0))
        hand_tokens[0, idx, 5] = _safe_float(hand.get("velocity_3d", [0.0, 0.0, 0.0])[0] if hand.get("velocity_3d") else 0.0)
        hand_tokens[0, idx, 6] = _safe_float(hand.get("velocity_3d", [0.0, 0.0, 0.0])[1] if hand.get("velocity_3d") else 0.0)
        hand_tokens[0, idx, 7] = _safe_float(hand.get("velocity_3d", [0.0, 0.0, 0.0])[2] if hand.get("velocity_3d") else 0.0)
        hand_tokens[0, idx, 8] = 1.0 if str(hand.get("side", "unknown")).lower() == "left" else 0.0
        hand_tokens[0, idx, 9] = 1.0 if str(hand.get("side", "unknown")).lower() == "right" else 0.0
        hand_mask[0, idx] = True

    cam_pos = camera_pose.get("camera_position_world", [0.0, 0.0, 0.0])
    if not isinstance(cam_pos, (list, tuple)) or len(cam_pos) < 3:
        cam_pos = [0.0, 0.0, 0.0]
    camera_tokens = np.array(
        [
            [
                _safe_float(cam_pos[0]),
                _safe_float(cam_pos[1]),
                _safe_float(cam_pos[2]),
                _safe_float(camera_pose.get("tracking_quality", 0.0)),
                _safe_float(camera_pose.get("pnp_inliers", 0.0)),
                _safe_float(camera_pose.get("local_keyframe_baseline", 0.0)),
                _safe_float(camera_pose.get("keyframes", 0.0)),
                _safe_float(camera_pose.get("geometry_verified_landmark_count", 0.0)),
                _safe_float(camera_pose.get("triangulated_landmark_count", 0.0)),
                _safe_float(camera_pose.get("persistent_landmark_count", 0.0)),
                _safe_float(camera_pose.get("frames_since_pnp_lock", 0.0)),
                1.0 if str(camera_pose.get("pose_source", "unknown")) == "pnp" else 0.0,
            ]
        ],
        dtype=np.float32,
    )

    tri = camera_pose.get("triangulation", {}) or {}
    depth_tokens = np.array(
        [
            [
                _safe_float(tri.get("candidates", 0.0)),
                _safe_float(tri.get("accepted", 0.0)),
                _safe_float(tri.get("rejected_reprojection", 0.0)),
                _safe_float(tri.get("rejected_depth", 0.0)),
                _safe_float(tri.get("depth_disagreement", 0.0)),
                _safe_float(camera_pose.get("mean_stable_reprojection_error", 0.0)),
            ]
        ],
        dtype=np.float32,
    )

    return TokenBatch(
        anchor_tokens=torch.tensor(anchor_tokens, device=device),
        object_tokens=torch.tensor(object_tokens, device=device),
        hand_tokens=torch.tensor(hand_tokens, device=device),
        camera_tokens=torch.tensor(camera_tokens, device=device),
        depth_tokens=torch.tensor(depth_tokens, device=device),
        anchor_mask=torch.tensor(anchor_mask, device=device),
        object_mask=torch.tensor(object_mask, device=device),
        hand_mask=torch.tensor(hand_mask, device=device),
    )


class WorldLatentModel(nn.Module):
    """Small token-based latent world model with explicit interaction heads."""

    def __init__(
        self,
        *,
        anchor_dim: int = 12,
        object_dim: int = 10,
        hand_dim: int = 10,
        camera_dim: int = 12,
        depth_dim: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        latent_dim: int = 96,
        action_classes: int = 8,
    ):
        super().__init__()
        self.anchor_proj = nn.Linear(anchor_dim, d_model)
        self.object_proj = nn.Linear(object_dim, d_model)
        self.hand_proj = nn.Linear(hand_dim, d_model)
        self.camera_proj = nn.Linear(camera_dim, d_model)
        self.depth_proj = nn.Linear(depth_dim, d_model)

        self.token_type_embed = nn.Parameter(torch.zeros(5, d_model))
        nn.init.normal_(self.token_type_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.world_proj = nn.Linear(d_model, latent_dim)

        self.object_refine_head = nn.Sequential(
            nn.LayerNorm(d_model + latent_dim),
            nn.Linear(d_model + latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.occlusion_head = nn.Sequential(
            nn.LayerNorm(d_model + latent_dim),
            nn.Linear(d_model + latent_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_classes),
        )
        self.distance_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _concat_tokens(self, batch: TokenBatch):
        anchor = self.anchor_proj(batch.anchor_tokens) + self.token_type_embed[0].view(1, 1, -1)
        obj = self.object_proj(batch.object_tokens) + self.token_type_embed[1].view(1, 1, -1)
        hand = self.hand_proj(batch.hand_tokens) + self.token_type_embed[2].view(1, 1, -1)
        camera = self.camera_proj(batch.camera_tokens).unsqueeze(1) + self.token_type_embed[3].view(1, 1, -1)
        depth = self.depth_proj(batch.depth_tokens).unsqueeze(1) + self.token_type_embed[4].view(1, 1, -1)

        tokens = torch.cat([anchor, obj, hand, camera, depth], dim=1)
        masks = torch.cat(
            [
                batch.anchor_mask,
                batch.object_mask,
                batch.hand_mask,
                torch.ones_like(batch.camera_tokens[:, :1], dtype=torch.bool),
                torch.ones_like(batch.depth_tokens[:, :1], dtype=torch.bool),
            ],
            dim=1,
        )
        return tokens, masks

    def forward(self, batch: TokenBatch) -> dict[str, torch.Tensor]:
        tokens, valid_mask = self._concat_tokens(batch)
        encoded = self.encoder(tokens, src_key_padding_mask=~valid_mask)
        encoded = self.norm(encoded)

        # Masked mean pool into compact latent state.
        denom = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (encoded * valid_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        world_z = self.world_proj(pooled)

        # Split encoded back into token groups.
        n_anchor = batch.anchor_tokens.shape[1]
        n_object = batch.object_tokens.shape[1]
        n_hand = batch.hand_tokens.shape[1]
        object_encoded = encoded[:, n_anchor:n_anchor + n_object]
        hand_encoded = encoded[:, n_anchor + n_object:n_anchor + n_object + n_hand]

        # Object refinement and occlusion logits.
        world_expand_obj = world_z.unsqueeze(1).expand(-1, n_object, -1)
        object_with_world = torch.cat([object_encoded, world_expand_obj], dim=-1)
        object_delta_xyz = self.object_refine_head(object_with_world)
        occlusion_logits = self.occlusion_head(object_with_world).squeeze(-1)

        # Single hand-object distance matrix using first hand token as anchor.
        if n_hand > 0:
            first_hand = hand_encoded[:, :1, :].expand(-1, n_object, -1)
            hand_obj_pair = torch.cat([object_encoded, first_hand], dim=-1)
            hand_object_distance = self.distance_head(hand_obj_pair).squeeze(-1)
        else:
            hand_object_distance = torch.zeros(
                object_encoded.shape[0],
                object_encoded.shape[1],
                device=object_encoded.device,
            )

        action_hint_logits = self.action_head(world_z)
        return {
            "world_z": world_z,
            "object_delta_xyz": object_delta_xyz,
            "occlusion_logits": occlusion_logits,
            "hand_object_distance": hand_object_distance,
            "action_hint_logits": action_hint_logits,
        }
