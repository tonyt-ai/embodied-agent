"""Clean raw transition records into a filtered dataset.

This script reads transitions from `data/transitions.jsonl` and writes a
cleaned subset to `data/transitions_clean.jsonl`. The filtering removes
small/noisy motions, overly large jumps, axis-aligned outliers, and
actions that do not align with the observed displacement.

Filtering parameters (tuned heuristics):
- `MIN_MOVE`: ignore transitions with tiny movement magnitude
- `MAX_MOVE`: ignore transitions with implausibly large movement
- `MAX_AXIS`: ignore transitions with a single-axis spike
- `MIN_COS`: require action vector to be reasonably aligned with motion

This module is meant to be run as a script.
"""

import json
import math
import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

# Input / output files (JSON Lines format)
RAW_FILE = os.path.join(DATA_DIR, "transitions.jsonl")
CLEAN_FILE = os.path.join(DATA_DIR, "transitions_clean.jsonl")

# Heuristic thresholds used for filtering transitions
MIN_MOVE = 0.010   # minimum Euclidean displacement to consider
MAX_MOVE = 0.090   # maximum reasonable displacement to keep
MAX_AXIS = 0.075   # max allowed movement along a single axis
MIN_COS = 0.75     # minimum cosine similarity between action and displacement


def norm(x, y):
    """Return the Euclidean norm for a 2D vector (x, y)."""
    return math.sqrt(x * x + y * y)


# Counters for diagnostics printed at the end
kept = 0
total = 0

with open(RAW_FILE, "r", encoding="utf-8") as f, \
     open(CLEAN_FILE, "w", encoding="utf-8") as out:
    for line in f:
        total += 1
        t = json.loads(line)

        # Expect each transition to contain: state, action, next_state
        s = t["state"]
        a = t["action"]
        ns = t["next_state"]

        # Basic sanity checks: require at least 2 dims for position
        if len(s) < 2 or len(ns) < 2:
            continue

        # Ignore transitions where start or end position is effectively zero
        # (likely invalid or uninitialized readings)
        if abs(s[0]) < 1e-6 and abs(s[1]) < 1e-6:
            continue
        if abs(ns[0]) < 1e-6 and abs(ns[1]) < 1e-6:
            continue

        # Displacement vector and its magnitude
        dx = ns[0] - s[0]
        dy = ns[1] - s[1]
        move = norm(dx, dy)

        # Discard too small or too large motions
        if move < MIN_MOVE:
            continue
        if move > MAX_MOVE:
            continue

        # Discard transitions dominated by movement along a single axis
        if abs(dx) > MAX_AXIS or abs(dy) > MAX_AXIS:
            continue

        # Compute cosine similarity between the action vector and observed
        # displacement. Add small epsilons to denominators to avoid div-by-zero.
        an = norm(a[0], a[1]) + 1e-8
        cos = (dx * a[0] + dy * a[1]) / (move * an + 1e-8)

        # Keep only transitions where the action is reasonably aligned with motion
        if cos < MIN_COS:
            continue

        # Passed all filters: write to clean output
        kept += 1
        out.write(json.dumps(t) + "\n")

print(f"kept {kept}/{total} transitions")