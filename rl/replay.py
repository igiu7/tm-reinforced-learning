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
        # pojedyncze nagrody z n kroków, ich liczba i czy next_obs jest końcem epizodu - potrzebne
        # mini-wyścigom (rl/agents/dqn.py), które składają zwrot dopiero przy losowaniu
        self.step_rewards = np.zeros((capacity, self.n_step), dtype=np.float32)
        self.n_steps = np.zeros(capacity, dtype=np.int16)
        self.terminal = np.zeros(capacity, dtype=bool)
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
        rewards = [r for _, _, r in self._pending]
        for r in rewards:
            ret += g * r
            g *= self.gamma
        obs, action, _ = self._pending.popleft()
        i = self.pos
        self.obs[i], self.actions[i], self.returns[i] = obs, action, ret
        self.next_obs[i] = next_obs
        self.discounts[i] = 0.0 if terminated else g
        self.step_rewards[i] = 0.0
        self.step_rewards[i, :len(rewards)] = rewards
        self.n_steps[i] = len(rewards)
        self.terminal[i] = terminated
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # --- próbkowanie ------------------------------------------------------------
    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.returns[idx], self.next_obs[idx], self.discounts[idx], idx)

    # --- zapis / odczyt (żeby wznowiony trening nie zaczynał z pustą pamięcią) ----------------
    def save(self, path: Path) -> None:
        n = self.size
        tmp = path.with_suffix(".tmp.npz")
        # zapisujemy w kolejności od najstarszego, żeby po wczytaniu nadpisywanie szło jak wcześniej
        order = np.arange(self.pos - n, self.pos) % self.capacity if n == self.capacity else np.arange(n)
        np.savez(tmp, obs=self.obs[order], next_obs=self.next_obs[order], actions=self.actions[order],
                 returns=self.returns[order], discounts=self.discounts[order],
                 step_rewards=self.step_rewards[order], n_steps=self.n_steps[order], terminal=self.terminal[order])
        tmp.replace(path)

    def load(self, path: Path) -> bool:
        data = np.load(path)
        if data["obs"].shape[1] != self.obs.shape[1] or "step_rewards" not in data.files                 or data["step_rewards"].shape[1] != self.n_step:
            return False   # bufor z innej wersji programu / innego n_step - zaczynamy z pustym
        n = min(len(data["obs"]), self.capacity)
        sl = slice(len(data["obs"]) - n, None)   # przy mniejszym buforze bierzemy najnowsze
        self.obs[:n], self.next_obs[:n] = data["obs"][sl], data["next_obs"][sl]
        self.actions[:n], self.returns[:n], self.discounts[:n] = data["actions"][sl], data["returns"][sl], data["discounts"][sl]
        self.step_rewards[:n], self.n_steps[:n], self.terminal[:n] = (data["step_rewards"][sl], data["n_steps"][sl],
                                                                    data["terminal"][sl])
        self.size, self.pos = n, n % self.capacity
        return True


class SumTree:
    """Drzewo sum: liście = priorytety przejść, każdy węzeł = suma dzieci, korzeń = suma wszystkich.
    Losowanie proporcjonalne do priorytetu i aktualizacja kosztują O(log N) zamiast O(N).
    Operacje są zwektoryzowane w numpy dla całej paczki (batcha) naraz."""

    def __init__(self, capacity: int):
        self.size = 1
        while self.size < capacity:          # pełne drzewo binarne: liczba liści = potęga dwójki
            self.size *= 2
        self.tree = np.zeros(2 * self.size, dtype=np.float64)   # węzeł 1 = korzeń, liście od self.size
        self.depth = int(np.log2(self.size))

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def get(self, idx: np.ndarray) -> np.ndarray:
        return self.tree[idx + self.size]

    def set(self, idx: np.ndarray, values: np.ndarray) -> None:
        node = np.asarray(idx) + self.size
        self.tree[node] = values
        for _ in range(self.depth):           # przelicz sumy w górę drzewa, poziom po poziomie
            node = np.unique(node // 2)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def find(self, values: np.ndarray) -> np.ndarray:
        """Dla każdej wartości z [0, total) zwraca indeks liścia, w którego przedział wpada."""
        node = np.ones(len(values), dtype=np.int64)
        values = values.astype(np.float64).copy()
        for _ in range(self.depth):
            left = 2 * node
            go_right = values >= self.tree[left]
            values -= self.tree[left] * go_right
            node = left + go_right
        return node - self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Priorytetowy bufor powtórek (Schaul i in., 2016, wariant proporcjonalny).

    Przejście i jest losowane z prawdopodobieństwem P(i) = p_i^alpha / sum_k p_k^alpha, gdzie
    p_i = |błąd TD| + eps. Sieć częściej uczy się na tym, czego jeszcze nie umie (wypadki, trudne
    miejsca trasy), zamiast w kółko na łatwej prostej.

    Losowanie nierównomierne zmienia rozkład danych, więc gradient każdej próbki jest mnożony przez
    wagę korygującą w_i = (N * P(i))^(-beta) / max w. beta rośnie do 1 w trakcie treningu (pełna korekta
    na końcu, gdy stabilność jest najważniejsza). alpha=0 to zwykły bufor jednorodny."""

    def __init__(self, capacity: int, obs_dim: int, gamma: float, n_step: int, rng: np.random.Generator,
                 alpha: float, eps: float):
        super().__init__(capacity, obs_dim, gamma, n_step, rng)
        self.alpha, self.eps = alpha, eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0           # nowe przejścia dostają najwyższy priorytet -> trafią do nauki

    def _store_front(self, next_obs, terminated: bool) -> None:
        i = self.pos
        super()._store_front(next_obs, terminated)
        self.tree.set(np.array([i]), np.array([self.max_priority ** self.alpha]))

    def sample(self, batch_size: int, beta: float = 1.0):
        total = self.tree.total
        # losowanie warstwowe: [0, total) dzielimy na batch_size równych odcinków, z każdego jedna próbka
        segment = total / batch_size
        values = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
        idx = np.minimum(self.tree.find(np.minimum(values, total * (1 - 1e-12))), self.size - 1)
        probs = self.tree.get(idx) / total
        weights = (self.size * np.maximum(probs, 1e-12)) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return (self.obs[idx], self.actions[idx], self.returns[idx], self.next_obs[idx], self.discounts[idx],
                idx, weights)

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = np.abs(td_errors) + self.eps
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.tree.set(idx, priorities ** self.alpha)

    def load(self, path: Path) -> bool:
        if not super().load(path):
            return False
        # priorytety nie są zapisywane - po wczytaniu wszystkie przejścia startują z równym priorytetem
        self.tree.set(np.arange(self.size), np.full(self.size, self.max_priority ** self.alpha))
        return True
