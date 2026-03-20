"""Train a small dynamics model from collected transitions.

This script loads transitions (state, action, next_state) from
`data/transitions_clean.jsonl` if present, falling back to the raw
`data/transitions.jsonl`. It filters out invalid or noisy records,
constructs tensors, and trains a compact MLP `DynamicsModel`.

Hyperparameters are set inline and are suitable for quick experiments.
"""

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random

from dynamics_model import DynamicsModel

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths for raw, cleaned transitions and the saved model
RAW_FILE = os.path.join(DATA_DIR, "transitions.jsonl")
CLEAN_FILE = os.path.join(DATA_DIR, "transitions_clean.jsonl")
MODEL_FILE = os.path.join(MODEL_DIR, "dynamics_model.pt")


def load_data():
    """Load and filter transitions, returning input/output tensors.

    Filtering performed here mirrors the original heuristics:
    - prefer the cleaned transitions file if present
    - require matching state dimensions and 2D actions
    - ignore near-zero or invalid start/end positions
    - only keep transitions whose 2D displacement magnitude falls
      within a reasonable range
    """
    xs = []
    ys = []

    transitions_file = CLEAN_FILE if os.path.exists(CLEAN_FILE) else RAW_FILE
    print(f"Using transitions file: {transitions_file}")

    with open(transitions_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            s = row["state"]
            a = row["action"]
            ns = row["next_state"]

            # Basic consistency checks
            if len(s) != len(ns):
                continue
            if len(a) != 2:
                continue

            # Ignore transitions where start or next positions are essentially zero
            if sum(abs(v) for v in s[:2]) < 1e-6:
                continue
            if sum(abs(v) for v in ns[:2]) < 1e-6:
                continue

            # Compute 2D displacement and filter by magnitude
            dx = ns[0] - s[0]
            dy = ns[1] - s[1]
            move = (dx * dx + dy * dy) ** 0.5
            if move < 0.008 or move > 0.12:
                continue

            # Input is concatenated state+action, target is next state
            xs.append(s + a)
            ys.append(ns)

    X = torch.tensor(xs, dtype=torch.float32)
    Y = torch.tensor(ys, dtype=torch.float32)
    print(f"Loaded {len(X)} clean transitions")
    return X, Y


def train():
    """Load data, create model, and run training loop.

    Training details:
    - Optimizer: Adam, lr=1e-3
    - Loss: MSE
    - Batch size: up to 64 (clamped to dataset size)
    - Epochs: 400
    - Prints loss every 50 epochs
    """
    X, Y = load_data()
    if len(X) == 0:
        raise RuntimeError("No valid transitions found.")

    state_dim = Y.shape[1]
    action_dim = 2

    model = DynamicsModel(state_dim=state_dim, action_dim=action_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = min(64, len(X))
    epochs = 400

    for epoch in range(epochs):
        # Shuffle indices for each epoch to form random minibatches
        idx = list(range(len(X)))
        random.shuffle(idx)
        total_loss = 0.0

        for i in range(0, len(X), batch_size):
            batch_idx = idx[i:i + batch_size]
            xb = X[batch_idx]
            yb = Y[batch_idx]

            # Split xb into state and action parts according to state_dim
            pred = model(xb[:, :state_dim], xb[:, state_dim:])
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"epoch={epoch} loss={total_loss:.6f}")

    # Save trained model weights
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    train()
