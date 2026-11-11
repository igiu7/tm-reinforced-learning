"""
Wspólny interfejs agentów. Trener (rl/trainer.py) zna tylko te metody, więc nowy algorytm
(np. SARSA) to nowa klasa dziedzicząca po Agent + wpis w rl/agents/__init__.py.

Cykl życia w jednym epizodzie:
    agent.start_episode()
    a = agent.act(obs, explore=True)                 # wybór akcji (indeks w CONFIG.actions)
    agent.observe(obs, a, r, next_obs, term, trunc)  # nauka na przejściu (tylko w treningu)
    ...
    agent.end_episode()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Agent(ABC):
    name: str = "agent"

    def __init__(self, obs_dim: int, n_actions: int, cfg, seed: int = 0):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    # --- decyzje i nauka ----------------------------------------------------
    @abstractmethod
    def act(self, obs: np.ndarray, explore: bool) -> int:
        """Indeks akcji. explore=False -> polityka zachłanna (przejazdy testowe, jazda)."""

    @abstractmethod
    def observe(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray,
                terminated: bool, truncated: bool) -> None:
        """Jedno przejście środowiska w trakcie treningu. terminated = stan terminalny (bez bootstrapu),
        truncated = przerwanie z zewnątrz (limit czasu) - wartość dalszej jazdy nadal się liczy."""

    def start_episode(self) -> None:
        pass

    def end_episode(self) -> None:
        pass

    # --- informacje do logów i wykresów -------------------------------------
    def stats(self) -> dict:
        """Liczby opisujące stan nauki (trafiają do episodes.csv). Klucze: epsilon, loss, q_mean, total_steps."""
        return {}

    def describe(self) -> str:
        return self.name

    # --- zapis / odczyt -----------------------------------------------------
    @abstractmethod
    def save(self, path: Path, meta: dict, with_buffer: bool = False) -> None:
        """Zapisuje stan agenta. meta: dane trenera (epizod, najlepszy czas, ...)."""

    @abstractmethod
    def load(self, path: Path, with_buffer: bool = False) -> dict:
        """Wczytuje stan agenta i zwraca meta zapisane przez save()."""
