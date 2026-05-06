"""Tiny JEPA interaction head.

Deterministic lightweight scoring used for online gating:
- higher score when hand/object embeddings are similar
- higher score when 3D distance is small
"""

from __future__ import annotations

import math
from typing import Sequence


def _cos(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        aa += x * x
        bb += y * y
    den = math.sqrt(max(aa, 1e-8) * max(bb, 1e-8))
    return max(-1.0, min(1.0, dot / den))


def interaction_score(
    hand_embedding: Sequence[float] | None,
    object_embedding: Sequence[float] | None,
    distance_m: float,
    *,
    near_distance_m: float = 0.22,
) -> dict:
    sim = _cos(hand_embedding, object_embedding)
    sim01 = 0.5 * (sim + 1.0)
    d = max(0.0, float(distance_m))
    near = max(0.0, 1.0 - d / max(near_distance_m, 1e-6))
    score = 0.62 * sim01 + 0.38 * near
    return {
        "jepa_similarity": round(sim, 4),
        "jepa_similarity01": round(sim01, 4),
        "jepa_interaction_score": round(max(0.0, min(1.0, score)), 4),
    }
