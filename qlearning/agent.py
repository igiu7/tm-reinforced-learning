"""
Tabelaryczny Q(lambda) Watkinsa.

Zwykły Q-learning (lam=0):
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

Ze śladami eligibility (lam>0) ten sam błąd TD jest rozprowadzany na ostatnio odwiedzone pary
(s,a), z wagą malejącą o gamma*lam na krok. Kara za wypadnięcie od razu "dociera" do decyzji
sprzed sekundy, a nie tylko do ostatniego kroku - nauka jest kilkukrotnie szybsza.
Wariant Watkinsa czyści ślady po akcji eksploracyjnej, bo dalsze kroki nie pochodzą już
z polityki zachłannej, której wartość szacujemy (to zachowuje własność off-policy Q-learningu).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, cfg, rng: np.random.Generator):
        self.cfg = cfg
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = rng
        self.q = np.full((n_states, n_actions), cfg.q_init, dtype=np.float64)
        self.visits = np.zeros((n_states, n_actions), dtype=np.int64)
        self.traces: dict[tuple[int, int], float] = {}

    # --- wybór akcji --------------------------------------------------------
    def greedy_action(self, s: int) -> int:
        row = self.q[s]
        best = np.flatnonzero(row == row.max())
        return int(best[0] if len(best) == 1 else self.rng.choice(best))  # losowo przy remisie

    def act(self, s: int, epsilon: float) -> tuple[int, bool]:
        """Zwraca (akcja, czy_zachłanna)."""
        if self.rng.random() < epsilon:
            a = int(self.rng.integers(self.n_actions))
            return a, bool(self.q[s, a] == self.q[s].max())
        return self.greedy_action(s), True

    # --- nauka --------------------------------------------------------------
    def start_episode(self) -> None:
        self.traces.clear()

    def update(self, s: int, a: int, r: float, s_next: int, terminal: bool,
               next_is_greedy: bool, alpha: float) -> float:
        """Jeden krok Q(lambda). `terminal` = koniec bez bootstrapu (meta, crash);
        przy przerwaniu limitem czasu przekaż terminal=False, next_is_greedy=False."""
        target = r if terminal else r + self.cfg.gamma * self.q[s_next].max()
        delta = target - self.q[s, a]
        self.visits[s, a] += 1

        if self.cfg.lam <= 0.0:
            self.q[s, a] += alpha * delta
            return delta

        # ślady zastępujące: dla stanu s zostaje tylko wykonana akcja, z wagą 1
        for b in range(self.n_actions):
            self.traces.pop((s, b), None)
        self.traces[(s, a)] = 1.0

        step = alpha * delta
        for (si, ai), e in self.traces.items():
            self.q[si, ai] += step * e

        if terminal or not next_is_greedy:
            self.traces.clear()
        else:
            decay = self.cfg.gamma * self.cfg.lam
            self.traces = {k: e * decay for k, e in self.traces.items() if e * decay >= self.cfg.trace_min}
        return delta

    # --- statystyki ---------------------------------------------------------
    def visited_fraction(self) -> float:
        return float((self.visits.sum(axis=1) > 0).mean())

    # --- zapis / odczyt -----------------------------------------------------
    def save(self, path: Path, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, q=self.q, visits=self.visits, meta=json.dumps(meta))
        tmp.replace(path)  # atomowo: przerwanie w trakcie zapisu nie psuje poprzedniego pliku

    def load(self, path: Path, expected_meta: dict) -> dict:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        for key in ("signature", "actions"):
            if meta.get(key) != expected_meta.get(key):
                raise ValueError(
                    f"Zapis {path} ma inną konfigurację ({key}). Zmieniłeś dyskretyzację lub akcje - "
                    "ustaw inny run_name albo resume=False."
                )
        self.q = data["q"]
        self.visits = data["visits"]
        return meta
