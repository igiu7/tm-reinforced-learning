"""
Rejestr algorytmów. Dodanie nowego:
  1. rl/agents/<nazwa>.py z klasą dziedziczącą po rl.agents.base.Agent,
  2. wpis w REGISTRY poniżej,
  3. sekcja z parametrami w rl_config.py (AGENT_CONFIGS["<nazwa>"]).
"""

from __future__ import annotations

from .base import Agent
from .dqn import DQNAgent

REGISTRY: dict[str, type[Agent]] = {
    "dqn": DQNAgent,
}


def make_agent(algorithm: str, obs_dim: int, n_actions: int, cfg, seed: int = 0) -> Agent:
    if algorithm not in REGISTRY:
        raise ValueError(f"Nieznany algorytm '{algorithm}'. Dostępne: {', '.join(REGISTRY)}")
    return REGISTRY[algorithm](obs_dim, n_actions, cfg, seed)


__all__ = ["Agent", "REGISTRY", "make_agent"]
