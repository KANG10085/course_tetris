from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class AfterstateValueNet(nn.Module):
    """
    A simple MLP that scores one afterstate feature vector with a single value.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()

        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
