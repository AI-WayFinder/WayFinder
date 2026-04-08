from __future__ import annotations

import torch
import torch.nn as nn

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
}

class TorchMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        dropout: float = 0.0,
        activation: str = "relu",
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        if activation not in ACTIVATION_MAP:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(ACTIVATION_MAP[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)