"""
Bufor powtórek (experience replay) z n-krokowymi zwrotami.

Dlaczego bufor: kolejne kroki jazdy są do siebie bardzo podobne (silnie skorelowane). Ucząc sieć
na losowych próbkach z dużej historii, a nie na ostatnich krokach, gradienty są stabilniejsze
i każde doświadczenie jest wykorzystane wiele razy.

n-krokowy zwrot: zamiast r_t + g*maxQ(s_{t+1}) bierzemy r_t + g*r_{t+1} + ... + g^{n-1}*r_{t+n-1}
+ g^n*maxQ(s_{t+n}). Kara za wypadek szybciej "dociera" do decyzji, które do niego doprowadziły
(podobnie jak ślady eligibility w Q(lambda)).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, gamma: float, n_step: int, rng: np.random.Generator):
        self.capacity, self.gamma, self.n_step, self.rng = capacity, gamma, max(1, n_step), rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)    # suma zdyskontowanych nagród z n kroków
        self.discounts = np.zeros(capacity, dtype=np.float32)  # g^k dla bootstrapu (0, gdy stan terminalny)
        self.pos = 0
        self.size = 0
        self._pending: deque = deque()   # ostatnie kroki czekające na złożenie n-krokowego zwrotu

    def __len__(self) -> int:
        return self.size

    # --- dodawanie ------------------------------------------------------------
    def add(self, obs, action, reward, next_obs, terminated: bool, truncated: bool) -> None:
        self._pending.append((obs, action, reward))
        if terminated or truncated:
            # koniec epizodu: zapisz wszystkie oczekujące kroki z krótszymi zwrotami
            while self._pending:
                self._store_front(next_obs, terminated)
        elif len(self._pending) >= self.n_step:
            self._store_front(next_obs, False)

    def reset_episode(self) -> None:
        self._pending.clear()

    def _store_front(self, next_obs, terminated: bool) -> None:
        ret, g = 0.0, 1.0
        for _, _, r in self._pending:
            ret += g * r
            g *= self.gamma
        obs, action, _ = self._pending.popleft()
        i = self.pos
        self.obs[i], self.actions[i], self.returns[i] = obs, action, ret
        self.next_obs[i] = next_obs
        self.discounts[i] = 0.0 if terminated else g
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # --- próbkowanie ------------------------------------------------------------
    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.returns[idx], self.next_obs[idx], self.discounts[idx])

    # --- zapis / odczyt (żeby wznowiony trening nie zaczynał z pustą pamięcią) ----------------
    def save(self, path: Path) -> None:
        n = self.size
        tmp = path.with_suffix(".tmp.npz")
        # zapisujemy w kolejności od najstarszego, żeby po wczytaniu nadpisywanie szło jak wcześniej
        order = np.arange(self.pos - n, self.pos) % self.capacity if n == self.capacity else np.arange(n)
        np.savez(tmp, obs=self.obs[order], next_obs=self.next_obs[order], actions=self.actions[order],
                 returns=self.returns[order], discounts=self.discounts[order])
        tmp.replace(path)

    def load(self, path: Path) -> bool:
        data = np.load(path)
        if data["obs"].shape[1] != self.obs.shape[1]:
            return False
        n = min(len(data["obs"]), self.capacity)
        sl = slice(len(data["obs"]) - n, None)   # przy mniejszym buforze bierzemy najnowsze
        self.obs[:n], self.next_obs[:n] = data["obs"][sl], data["next_obs"][sl]
        self.actions[:n], self.returns[:n], self.discounts[:n] = data["actions"][sl], data["returns"][sl], data["discounts"][sl]
        self.size, self.pos = n, n % self.capacity
        return True
