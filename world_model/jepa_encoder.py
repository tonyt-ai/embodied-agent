"""Frozen visual feature encoder for V-JEPA2.1 or DINOv2 embeddings.

This is intentionally lightweight and robust:
- Primary path when available: local V-JEPA2.1 repo/checkpoint.
- Optional path: HF ViT model when explicitly configured.
- Fallback path: DINOv2 crop embedding already available in this repo.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np
import torchvision.transforms as T

from dino_encoder import encode_bbox

try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
except Exception:
    torch = None
    Image = None
    AutoImageProcessor = None
    AutoModel = None


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if not np.isfinite(n) or n < 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / n).astype(np.float32)


class JepaFeatureEncoder:
    """Frozen visual encoder used for appearance memory and temporal prediction."""

    def __init__(self):
        self.enabled = os.environ.get("JEPA_ENABLED", "0").lower() in {"1", "true", "yes"}
        self.out_dim = int(os.environ.get("JEPA_OUT_DIM", "64"))
        self.backend = "disabled"
        self.ready = False
        self._processor = None
        self._model = None
        self._vjepa2_adapter = None
        self._vjepa2_model = None
        self._device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.model_id = os.environ.get("JEPA_MODEL_ID", "").strip()
        self.vjepa2_repo = os.environ.get("VJEPA2_REPO", "").strip() or self._auto_detect_vjepa2_repo()
        self.vjepa2_model = os.environ.get("VJEPA2_MODEL", "").strip()
        self.vjepa2_checkpoint = os.environ.get("VJEPA2_CHECKPOINT", "").strip()
        self.vjepa2_error = None
        self.local_only = os.environ.get("JEPA_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
        if not self.enabled:
            return
        if self._try_load_vjepa2():
            self.backend = "vjepa2"
            self.ready = True
            return
        # Optional HF backend; otherwise fallback to DINO crop helper.
        if self.model_id and AutoImageProcessor is not None and AutoModel is not None and torch is not None:
            try:
                self._processor = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=self.local_only)
                self._model = AutoModel.from_pretrained(self.model_id, local_files_only=self.local_only).to(self._device)
                self._model.eval()
                self.backend = "hf-vit"
                self.ready = True
                return
            except Exception:
                self._processor = None
                self._model = None
        self.backend = "dino-fallback"
        self.ready = True

    def _auto_detect_vjepa2_repo(self) -> str:
        here = os.path.abspath(os.path.dirname(__file__))
        repo_root = os.path.abspath(os.path.join(here, os.pardir))
        parent = os.path.abspath(os.path.join(repo_root, os.pardir))
        candidates = [
            os.path.join(repo_root, "third_party", "vjepa2-main"),
            os.path.join(repo_root, "third_party", "v-jepa2"),
            os.path.join(parent, "vjepa2-main"),
            os.path.join(parent, "v-jepa2"),
            r"C:\code\vjepa2-main",
            r"C:\code\v-jepa2",
        ]
        for candidate in candidates:
            if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "src")):
                return os.path.abspath(candidate)
        return ""

    def _try_load_vjepa2(self) -> bool:
        if not self.vjepa2_repo:
            return False
        repo = os.path.abspath(self.vjepa2_repo)
        if not os.path.isdir(repo):
            return False
        if repo not in sys.path:
            sys.path.insert(0, repo)
        # Best-effort adapter discovery (project-specific).
        adapter_candidates = (
            "vjepa2_adapter",
            "tools.vjepa2_adapter",
            "inference.vjepa2_adapter",
        )
        for mod_name in adapter_candidates:
            try:
                mod = __import__(mod_name, fromlist=["*"])
            except Exception:
                continue
            init_fn = getattr(mod, "load_encoder", None) or getattr(mod, "build_encoder", None)
            encode_fn_name = "encode_bbox"
            if init_fn is None or not callable(init_fn):
                continue
            try:
                adapter = init_fn(model_name=self.vjepa2_model or None, device=self._device)
                if adapter is None:
                    continue
                if callable(adapter):
                    self._vjepa2_adapter = {"callable": adapter, "mode": "callable"}
                    return True
                if hasattr(adapter, encode_fn_name) and callable(getattr(adapter, encode_fn_name)):
                    self._vjepa2_adapter = {"obj": adapter, "mode": "object"}
                    return True
            except Exception:
                continue
        # Native VJEPA2 repo path support.
        try:
            from src.hub.backbones import vjepa2_1_vit_base_384
            import torch
            model, _predictor = vjepa2_1_vit_base_384(pretrained=False)
            ckpt_path = self.vjepa2_checkpoint
            if not ckpt_path and self.vjepa2_model:
                if os.path.isfile(self.vjepa2_model):
                    ckpt_path = self.vjepa2_model
                else:
                    candidate = os.path.join(repo, "checkpoints", self.vjepa2_model)
                    if os.path.isfile(candidate):
                        ckpt_path = candidate
            if not ckpt_path:
                candidate = os.path.join(repo, "checkpoints", "vjepa2_1_vitb_dist_vitG_384.pt")
                if os.path.isfile(candidate):
                    ckpt_path = candidate
            if not ckpt_path or not os.path.isfile(ckpt_path):
                self.vjepa2_error = "checkpoint-not-found"
                return False
            state = torch.load(ckpt_path, map_location="cpu")
            enc = state.get("ema_encoder", state.get("target_encoder", state.get("encoder", state)))
            cleaned = {}
            for k, v in enc.items():
                nk = k.replace("module.", "").replace("backbone.", "")
                cleaned[nk] = v
            model.load_state_dict(cleaned, strict=False)
            model = model.to(self._device)
            model.eval()
            self._vjepa2_model = model
            self._vjepa2_adapter = {"mode": "native-model"}
            return True
        except Exception as exc:
            self.vjepa2_error = str(exc)
        return False

    def _encode_with_hf(self, frame_bgr: np.ndarray, bbox_norm) -> Optional[np.ndarray]:
        if self._processor is None or self._model is None or Image is None or torch is None:
            return None
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, int(round(float(bbox_norm[0]) * w))))
        y1 = max(0, min(h - 1, int(round(float(bbox_norm[1]) * h))))
        x2 = max(0, min(w, int(round(float(bbox_norm[2]) * w))))
        y2 = max(0, min(h, int(round(float(bbox_norm[3]) * h))))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        with torch.no_grad():
            batch = self._processor(images=image, return_tensors="pt")
            batch = {k: v.to(self._device) for k, v in batch.items()}
            out = self._model(**batch)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feat = out.pooler_output[0].detach().float().cpu().numpy()
            else:
                feat = out.last_hidden_state[:, 0, :][0].detach().float().cpu().numpy()
        if feat.ndim != 1 or feat.size == 0:
            return None
        feat = _l2_normalize(feat)
        if feat.size >= self.out_dim:
            return feat[: self.out_dim].astype(np.float32)
        padded = np.zeros((self.out_dim,), dtype=np.float32)
        padded[: feat.size] = feat
        return padded

    def encode_bbox(self, frame_bgr: np.ndarray, bbox_norm) -> list[float]:
        if not self.enabled or not self.ready:
            return []
        feat = None
        if self.backend == "vjepa2" and self._vjepa2_adapter is not None:
            try:
                mode = self._vjepa2_adapter.get("mode")
                if mode == "callable":
                    raw = self._vjepa2_adapter["callable"](frame_bgr, bbox_norm)
                elif mode == "object":
                    raw = self._vjepa2_adapter["obj"].encode_bbox(frame_bgr, bbox_norm)
                elif mode == "native-model":
                    h, w = frame_bgr.shape[:2]
                    x1 = max(0, min(w - 1, int(round(float(bbox_norm[0]) * w))))
                    y1 = max(0, min(h - 1, int(round(float(bbox_norm[1]) * h))))
                    x2 = max(0, min(w, int(round(float(bbox_norm[2]) * w))))
                    y2 = max(0, min(h, int(round(float(bbox_norm[3]) * h))))
                    if x2 > x1 and y2 > y1 and self._vjepa2_model is not None:
                        crop = frame_bgr[y1:y2, x1:x2]
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(rgb)
                        tfm = T.Compose([
                            T.Resize((384, 384)),
                            T.ToTensor(),
                            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ])
                        x = tfm(img).unsqueeze(0).to(self._device)
                        with torch.no_grad():
                            out = self._vjepa2_model(x)
                        arr = out[0] if isinstance(out, (list, tuple)) else out
                        if hasattr(arr, "ndim") and arr.ndim == 3:
                            arr = arr.mean(dim=1)
                        raw = arr[0].detach().float().cpu().numpy()
                    else:
                        raw = None
                else:
                    raw = None
                if raw is not None:
                    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
                    if arr.size > 0:
                        arr = _l2_normalize(arr)
                        if arr.size >= self.out_dim:
                            feat = arr[: self.out_dim]
                        else:
                            padded = np.zeros((self.out_dim,), dtype=np.float32)
                            padded[: arr.size] = arr
                            feat = padded
            except Exception:
                feat = None
        if feat is None and self.backend == "hf-vit":
            feat = self._encode_with_hf(frame_bgr, bbox_norm)
        if feat is None:
            try:
                emb = encode_bbox(frame_bgr, bbox_norm, out_dim=self.out_dim)
                feat = np.asarray(emb, dtype=np.float32)
                feat = _l2_normalize(feat)
            except Exception:
                feat = np.zeros((self.out_dim,), dtype=np.float32)
        return feat.astype(np.float32).tolist()

    def debug_info(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "ready": bool(self.ready),
            "backend": self.backend,
            "out_dim": int(self.out_dim),
            "model_id": self.model_id if self.model_id else None,
            "vjepa2_repo": self.vjepa2_repo if self.vjepa2_repo else None,
            "vjepa2_model": self.vjepa2_model if self.vjepa2_model else None,
            "vjepa2_error": self.vjepa2_error,
        }
