"""Small semantic label normalization helpers shared by runtime modules."""

from __future__ import annotations

import os


LABEL_ALIASES = {
    "cake stand": "dish",
    "cake plate": "dish",
    "serving stand": "dish",
    "serving plate": "dish",
    "fruit stand": "dish",
    "fruit plate": "dish",
    "fruit bowl": "dish",
    "teddy bear": "toy",
    "stuffed animal": "toy",
    "plush": "toy",
    "giraffe toy": "toy giraffe",
    "sophie the giraffe": "toy giraffe",
    "sophie giraffe": "toy giraffe",
}


def normalize_label(label: str | None) -> str:
    text = str(label or "").strip().lower().replace("_", " ")
    text = " ".join(text.split())
    if not text:
        return text
    aliases = dict(LABEL_ALIASES)
    for part in os.environ.get("SEMANTIC_LABEL_ALIASES", "").split(","):
        if ":" not in part:
            continue
        src, dst = part.split(":", 1)
        src = " ".join(src.strip().lower().replace("_", " ").split())
        dst = " ".join(dst.strip().lower().replace("_", " ").split())
        if src and dst:
            aliases[src] = dst
    return aliases.get(text, text)
