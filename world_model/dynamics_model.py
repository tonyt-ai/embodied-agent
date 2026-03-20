"""Simple dynamics predictor network.

This module implements a small fully-connected residual-free network
that predicts the next state from the current `state` and `action` vectors.
It is intentionally lightweight and used for prototyping and training in the `world_model` experiments.
"""

import torch
import torch.nn as nn


class DynamicsModel(nn.Module):
    """A compact MLP that maps (state, action) -> predicted next state.

    Args:
        state_dim: dimensionality of the state vector
        action_dim: dimensionality of the action vector
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # A small 2-layer MLP: input -> 128 hidden -> output
        # Input size is concatenated state + action
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )

    def forward(self, state, action):
        """Forward pass: concatenate inputs and run through MLP.

        `state` and `action` are expected to be tensors with the same
        leading batch dimensions; concatenation happens along the last
        feature axis (`dim=-1`). The network returns a tensor shaped
        like `state` (the predicted next state).
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)