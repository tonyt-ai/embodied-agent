"""Interaction-conditioned temporal head over grounded JEPA/DINO tokens.

This is the JEPA-like predictive layer for the Sophie demo: it predicts future
contact, placement, motion, and latent state from observed hand-object geometry
and appearance memory. It is not a full action-conditioned JEPA-AC model yet
because no explicit commanded action token is provided.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


EMB_DIM = 64


def _pad(v: List[float], n: int) -> np.ndarray:
    out = np.zeros((n,), dtype=np.float32)
    if not isinstance(v, list):
        return out
    m = min(len(v), n)
    if m > 0:
        out[:m] = np.asarray(v[:m], dtype=np.float32)
    return out


def build_feature_vector(
    hand_emb: List[float],
    obj_emb: List[float],
    distance_m: float,
    effective_distance_m: float,
    jepa_similarity: float,
    hand_speed_m: float,
    obj_speed_m: float,
    contact_streak: int,
    obj_pos_3d: List[float] | None = None,
    bbox: List[float] | None = None,
    support_label: str = "",
) -> np.ndarray:
    h = _pad(hand_emb, EMB_DIM)
    o = _pad(obj_emb, EMB_DIM)
    pos = _pad(obj_pos_3d or [], 3)
    box = _pad(bbox or [], 4)
    if box.size >= 4:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        box_geom = np.asarray([
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            max(0.0, x2 - x1),
            max(0.0, y2 - y1),
        ], dtype=np.float32)
    else:
        box_geom = np.zeros((4,), dtype=np.float32)
    support = str(support_label or "").strip().lower().replace("_", " ")
    support_geom = np.asarray([
        1.0 if support in {"mat", "black mat", "table mat", "placemat", "dish", "plate"} else 0.0,
        1.0 if support in {"tray", "plastic tray", "white tray"} else 0.0,
    ], dtype=np.float32)
    g = np.asarray(
        [
            float(distance_m),
            float(effective_distance_m),
            float(jepa_similarity),
            float(hand_speed_m),
            float(obj_speed_m),
            float(contact_streak),
        ],
        dtype=np.float32,
    )
    return np.concatenate([h, o, g, pos, box_geom, support_geom], axis=0)


class TemporalInteractionHead(nn.Module):
    def __init__(self, in_dim: int = EMB_DIM * 2 + 15, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.contact = nn.Linear(hidden, 1)
        self.motion = nn.Linear(hidden, 3)
        self.placement = nn.Linear(hidden, 1)
        self.release = nn.Linear(hidden, 1)
        self.target_tray = nn.Linear(hidden, 1)
        self.future_latent = nn.Linear(hidden, EMB_DIM)

    def forward(self, x):
        z = self.trunk(x)
        return {
            "contact_logit": self.contact(z),
            "motion_delta": self.motion(z),
            "placement_logit": self.placement(z),
            "release_logit": self.release(z),
            "target_tray_logit": self.target_tray(z),
            "future_latent": self.future_latent(z),
        }


class TemporalInteractionPredictor:
    def __init__(self, model_path: str | None = None):
        self.enabled = os.environ.get("TEMPORAL_HEAD_ENABLED", "1").lower() in {"1", "true", "yes"}
        self.ready = False
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.model_path = model_path or os.environ.get(
            "TEMPORAL_HEAD_MODEL_PATH",
            os.path.join(os.path.dirname(__file__), "models", "temporal_interaction_head_sophie.pt"),
        )
        self.contact_threshold = float(os.environ.get("TEMPORAL_HEAD_CONTACT_THRESHOLD", "0.20"))
        self.placement_threshold = float(os.environ.get("TEMPORAL_HEAD_PLACEMENT_THRESHOLD", "0.45"))
        self.target_tray_threshold = float(os.environ.get("TEMPORAL_HEAD_TARGET_TRAY_THRESHOLD", "0.45"))
        self.model = None
        if not self.enabled or torch is None or nn is None:
            return
        if os.path.isfile(self.model_path):
            state = torch.load(self.model_path, map_location=self.device)
            in_dim = EMB_DIM * 2 + 15
            first_weight = state.get("trunk.0.weight") if isinstance(state, dict) else None
            if first_weight is not None and hasattr(first_weight, "shape") and len(first_weight.shape) == 2:
                in_dim = int(first_weight.shape[1])
            self.model = TemporalInteractionHead(in_dim=in_dim).to(self.device)
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            self.ready = True

    def predict(self, feat: np.ndarray) -> Dict[str, float]:
        if not self.ready or self.model is None or torch is None:
            return {
                "contact_prob": 0.0,
                "placement_prob": 0.0,
                "release_prob": 0.0,
                "target_tray_prob": 0.5,
                "target_label": "target",
                "contact_signal": False,
                "placement_signal": False,
                "release_signal": False,
                "contact_threshold": float(self.contact_threshold),
                "placement_threshold": float(self.placement_threshold),
                "target_tray_threshold": float(self.target_tray_threshold),
                "motion_dx": 0.0,
                "motion_dy": 0.0,
                "motion_dz": 0.0,
                "source": "disabled",
            }
        with torch.no_grad():
            feat_arr = feat.astype(np.float32)
            expected_dim = int(self.model.trunk[0].in_features)
            if feat_arr.shape[0] < expected_dim:
                feat_arr = np.pad(feat_arr, (0, expected_dim - feat_arr.shape[0]))
            elif feat_arr.shape[0] > expected_dim:
                feat_arr = feat_arr[:expected_dim]
            x = torch.from_numpy(feat_arr).unsqueeze(0).to(self.device)
            out = self.model(x)
            contact_prob = torch.sigmoid(out["contact_logit"])[0, 0].item()
            place_prob = torch.sigmoid(out["placement_logit"])[0, 0].item()
            release_prob = torch.sigmoid(out.get("release_logit", out["placement_logit"]))[0, 0].item()
            target_tray_prob = torch.sigmoid(out.get("target_tray_logit", out["placement_logit"]))[0, 0].item()
            md = out["motion_delta"][0].detach().float().cpu().numpy().tolist()
            future_latent = out["future_latent"][0].detach().float().cpu().numpy().tolist()
        return {
            "contact_prob": float(contact_prob),
            "placement_prob": float(place_prob),
            "release_prob": float(release_prob),
            "target_tray_prob": float(target_tray_prob),
            "target_label": "tray" if target_tray_prob >= self.target_tray_threshold else "mat",
            "contact_signal": bool(contact_prob >= self.contact_threshold),
            "placement_signal": bool(place_prob >= self.placement_threshold),
            "release_signal": bool(release_prob >= self.placement_threshold),
            "contact_threshold": float(self.contact_threshold),
            "placement_threshold": float(self.placement_threshold),
            "target_tray_threshold": float(self.target_tray_threshold),
            "motion_dx": float(md[0]),
            "motion_dy": float(md[1]),
            "motion_dz": float(md[2]),
            "future_latent": [float(v) for v in future_latent],
            "source": "model",
        }
