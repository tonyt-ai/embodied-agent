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
) -> np.ndarray:
    h = _pad(hand_emb, EMB_DIM)
    o = _pad(obj_emb, EMB_DIM)
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
    return np.concatenate([h, o, g], axis=0)


class TemporalInteractionHead(nn.Module):
    def __init__(self, in_dim: int = EMB_DIM * 2 + 6, hidden: int = 128):
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
        self.model = None
        if not self.enabled or torch is None or nn is None:
            return
        self.model = TemporalInteractionHead().to(self.device)
        if os.path.isfile(self.model_path):
            state = torch.load(self.model_path, map_location=self.device)
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
                "motion_dx": 0.0,
                "motion_dy": 0.0,
                "motion_dz": 0.0,
                "source": "disabled",
            }
        with torch.no_grad():
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(self.device)
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
            "target_label": "tray" if target_tray_prob >= 0.5 else "mat",
            "contact_signal": bool(contact_prob >= self.contact_threshold),
            "placement_signal": bool(place_prob >= self.placement_threshold),
            "release_signal": bool(release_prob >= self.placement_threshold),
            "contact_threshold": float(self.contact_threshold),
            "placement_threshold": float(self.placement_threshold),
            "motion_dx": float(md[0]),
            "motion_dy": float(md[1]),
            "motion_dz": float(md[2]),
            "future_latent": [float(v) for v in future_latent],
            "source": "model",
        }
