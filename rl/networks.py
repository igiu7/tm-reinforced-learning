"""Sieci neuronowe używane przez agentów."""

from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, hidden: tuple, out_dim: int) -> nn.Sequential:
    """Wielowarstwowy perceptron: Linear -> ReLU -> ... -> Linear (wyjście bez aktywacji)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Stan -> wartości Q wszystkich akcji naraz.

    dueling=True: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a). Sieć osobno uczy się "jak dobry jest stan"
    i "o ile lepsza jest dana akcja" - pomaga, gdy w wielu stanach wybór akcji niewiele zmienia
    (np. na prostej)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple, dueling: bool):
        super().__init__()
        self.dueling = dueling
        if dueling:
            self.body = mlp(obs_dim, hidden[:-1], hidden[-1])
            self.value = nn.Sequential(nn.ReLU(), nn.Linear(hidden[-1], 1))
            self.advantage = nn.Sequential(nn.ReLU(), nn.Linear(hidden[-1], n_actions))
        else:
            self.body = mlp(obs_dim, hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.body(obs)
        h = self.body(obs)
        adv = self.advantage(h)
        return self.value(h) + adv - adv.mean(dim=1, keepdim=True)
