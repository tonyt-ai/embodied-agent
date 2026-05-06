"""Temporal semantic stabilization for SLAM foreground masking.

This module smooths frame-level detector outputs using track hysteresis and
optional DINO embedding similarity to reduce label flicker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter + 1e-6))


def _norm_embedding(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    if n <= 1e-6:
        return None
    return arr / n


@dataclass
class SemanticTrack:
    tid: int
    label: str
    bbox: list[float]
    confidence_ema: float
    embedding: np.ndarray | None
    hits: int = 1
    misses: int = 0
    age: int = 1
    label_votes: dict = field(default_factory=dict)

    def add_vote(self, label: str):
        self.label_votes[label] = int(self.label_votes.get(label, 0)) + 1
        self.label = max(self.label_votes.items(), key=lambda item: item[1])[0]

    def stable(self, min_hits: int, stable_conf: float, max_misses: int) -> bool:
        return self.hits >= min_hits and self.confidence_ema >= stable_conf and self.misses <= max_misses


class SemanticStabilizer:
    def __init__(
        self,
        *,
        min_confidence: float = 0.15,
        match_threshold: float = 0.28,
        bbox_ema: float = 0.4,
        conf_ema: float = 0.35,
        embedding_ema: float = 0.35,
        min_hits: int = 3,
        stable_confidence: float = 0.25,
        max_misses: int = 7,
        dynamic_labels: Iterable[str] | None = None,
    ):
        self.min_confidence = float(min_confidence)
        self.match_threshold = float(match_threshold)
        self.bbox_ema = float(bbox_ema)
        self.conf_ema = float(conf_ema)
        self.embedding_ema = float(embedding_ema)
        self.min_hits = int(min_hits)
        self.stable_confidence = float(stable_confidence)
        self.max_misses = int(max_misses)
        self.dynamic_labels = {str(label).lower() for label in (dynamic_labels or [])}
        self.tracks: dict[int, SemanticTrack] = {}
        self.next_tid = 1

    def reset(self):
        self.tracks.clear()
        self.next_tid = 1

    def _match_score(self, det: dict, track: SemanticTrack) -> float:
        iou_score = _iou(det["bbox"], track.bbox)
        emb_score = 0.0
        emb = det.get("embedding_norm")
        if emb is not None and track.embedding is not None:
            emb_score = float(np.dot(emb, track.embedding))
            emb_score = max(0.0, min(1.0, emb_score))
        return 0.68 * iou_score + 0.32 * emb_score

    def update(self, detections: list[dict]) -> dict:
        valid = []
        for det in detections:
            conf = float(det.get("confidence", 0.0))
            if conf < self.min_confidence:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            if x2 <= x1 or y2 <= y1:
                continue
            item = {
                "label": str(det.get("label", "unknown")),
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "embedding_norm": _norm_embedding(det.get("embedding")),
            }
            valid.append(item)

        track_ids = list(self.tracks.keys())
        used_dets = set()
        matched_tracks = set()

        candidates = []
        for d_idx, det in enumerate(valid):
            for tid in track_ids:
                score = self._match_score(det, self.tracks[tid])
                if score >= self.match_threshold:
                    candidates.append((score, d_idx, tid))
        candidates.sort(reverse=True, key=lambda item: item[0])

        for score, d_idx, tid in candidates:
            if d_idx in used_dets or tid in matched_tracks:
                continue
            used_dets.add(d_idx)
            matched_tracks.add(tid)
            det = valid[d_idx]
            track = self.tracks[tid]
            track.age += 1
            track.hits += 1
            track.misses = 0
            track.confidence_ema = (1.0 - self.conf_ema) * track.confidence_ema + self.conf_ema * det["confidence"]
            track.bbox = (
                (1.0 - self.bbox_ema) * np.asarray(track.bbox, dtype=np.float32)
                + self.bbox_ema * np.asarray(det["bbox"], dtype=np.float32)
            ).tolist()
            track.add_vote(det["label"])
            if det["embedding_norm"] is not None:
                if track.embedding is None:
                    track.embedding = det["embedding_norm"]
                else:
                    mix = (1.0 - self.embedding_ema) * track.embedding + self.embedding_ema * det["embedding_norm"]
                    n = float(np.linalg.norm(mix))
                    track.embedding = mix / max(n, 1e-6)

        created_ids = set()
        for d_idx, det in enumerate(valid):
            if d_idx in used_dets:
                continue
            tid = self.next_tid
            self.next_tid += 1
            track = SemanticTrack(
                tid=tid,
                label=det["label"],
                bbox=det["bbox"],
                confidence_ema=float(det["confidence"]),
                embedding=det["embedding_norm"],
                hits=1,
                misses=0,
                age=1,
                label_votes={det["label"]: 1},
            )
            self.tracks[tid] = track
            created_ids.add(tid)

        stale_ids = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks and tid not in created_ids:
                track.age += 1
                track.misses += 1
                track.confidence_ema *= (1.0 - 0.15)
            if track.misses > self.max_misses:
                stale_ids.append(tid)
        for tid in stale_ids:
            self.tracks.pop(tid, None)

        stable_tracks = [
            track for track in self.tracks.values()
            if track.stable(self.min_hits, self.stable_confidence, self.max_misses)
        ]

        dynamic_tracks = [
            track for track in stable_tracks
            if (not self.dynamic_labels) or (track.label.lower() in self.dynamic_labels)
        ]
        return {
            "stable_tracks": [
                {
                    "id": track.tid,
                    "label": track.label,
                    "bbox": [round(float(v), 4) for v in track.bbox],
                    "confidence": round(float(track.confidence_ema), 3),
                    "hits": int(track.hits),
                    "misses": int(track.misses),
                }
                for track in stable_tracks
            ],
            "dynamic_bboxes": [track.bbox for track in dynamic_tracks],
            "num_tracks": len(self.tracks),
            "num_stable_tracks": len(stable_tracks),
            "num_dynamic_tracks": len(dynamic_tracks),
        }


def build_foreground_mask(shape, bboxes: list[list[float]]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if not bboxes:
        return mask
    for bbox in bboxes:
        x1 = max(0, min(width - 1, int(round(float(bbox[0]) * width))))
        y1 = max(0, min(height - 1, int(round(float(bbox[1]) * height))))
        x2 = max(0, min(width, int(round(float(bbox[2]) * width))))
        y2 = max(0, min(height, int(round(float(bbox[3]) * height))))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 1
    return mask
