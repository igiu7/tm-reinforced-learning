"""
Sekcje toru: agent poprawia jazdę po kawałku i zatrzymuje tylko udane poprawki.

Trasa jest dzielona na `count` równych (wg długości linii referencyjnej) sekcji. Moduł:

1. Mierzy czas każdej sekcji w każdym przejeździe i pamięta najlepszy czas sekcji.
2. Wybiera sekcję do ćwiczenia (eksploracja tylko w jednej sekcji naraz). Agent jedzie swoją
   najlepszą (zachłanną) jazdą wszędzie poza wybraną sekcją, a tylko w niej próbuje innych akcji.
   Dzięki temu wiadomo, że zmiana wyniku sekcji wynika z prób w tej sekcji, a losowe ruchy nie psują
   sekcji, które agent już jedzie dobrze. Sekcja jest losowana proporcjonalnie do tego, ile można
   w niej zyskać: jak często kończą się w niej przejazdy (wypadki) i ile średnio traci do swojego
   najlepszego czasu.
3. Wskazuje przejazdy sekcji, które pobiły jej najlepszy czas (i po których auto jechało dalej, a nie
   rozbiło się tuż za sekcją). Trener zapisuje je do chronionej pamięci agenta (DQN: bufor "elitarny",
   którego zwykły bufor nie nadpisuje), więc sieć ciągle uczy się na najlepszej znanej jeździe
   w każdej sekcji i jej nie zapomina - poprawki gorsze od dotychczasowej jazdy nie wypierają lepszych.

Moduł nie zależy od algorytmu: trener pyta go o epsilon w danym miejscu trasy (jak rl/exploration.py).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class SectionTracker:
    def __init__(self, cfg, track_len: float):
        self.cfg = cfg
        self.track_len = track_len
        self.n = int(cfg.count)
        self.bounds = np.linspace(0.0, track_len, self.n + 1)          # granice sekcji [m]
        self.best_ms: list[float] = [math.inf] * self.n                  # najlepszy czas sekcji
        # ostatnie przejazdy sekcji: (czas [ms] albo None = przejazd skończył się w tej sekcji)
        self.recent: list[deque] = [deque(maxlen=cfg.window_episodes) for _ in range(self.n)]
        self.focus: int | None = None       # sekcja ćwiczona w bieżącym epizodzie
        self.rng = np.random.default_rng(cfg.seed)

    # --- pomocnicze --------------------------------------------------------------
    def section_of(self, progress_m: float) -> int:
        return int(min(self.n - 1, max(0, np.searchsorted(self.bounds, progress_m, side="right") - 1)))

    def label(self, k: int) -> str:
        return f"{k + 1} ({100 * self.bounds[k] / self.track_len:.0f}-{100 * self.bounds[k + 1] / self.track_len:.0f}%)"

    # --- eksploracja ----------------------------------------------------------------
    def active(self, base_epsilon: float) -> bool:
        """Ćwiczenie sekcji zaczyna się, gdy zwykła eksploracja (epsilon z harmonogramu) już opadła."""
        return base_epsilon <= self.cfg.start_below_epsilon

    def start_episode(self, base_epsilon: float) -> None:
        self.focus = self._choose_focus() if self.active(base_epsilon) else None

    def epsilon(self, base_epsilon: float, progress_m: float) -> float:
        if self.focus is None:
            return base_epsilon
        k = self.section_of(progress_m)
        # w ćwiczonej sekcji (i kawałek przed nią, żeby wjechać w nią inaczej) - eksploracja,
        # poza nią - najlepsza znana jazda
        lead = self.cfg.lead_in_m
        if self.bounds[self.focus] - lead <= progress_m < self.bounds[self.focus + 1] or k == self.focus:
            return self.cfg.focus_epsilon
        return min(base_epsilon, self.cfg.outside_epsilon)

    def potential(self) -> np.ndarray:
        """Ile można zyskać w każdej sekcji: udział przejazdów kończących się w niej + strata do najlepszego czasu."""
        c = self.cfg
        pot = np.zeros(self.n)
        for k in range(self.n):
            r = self.recent[k]
            if not r:
                continue
            fail = sum(t is None for t in r) / len(r)
            times = [t for t in r if t is not None]
            loss = (np.median(times) - self.best_ms[k]) / self.best_ms[k] if times and self.best_ms[k] < math.inf else 0.0
            pot[k] = c.failure_weight * fail + c.time_loss_weight * max(0.0, loss)
        return pot

    def _choose_focus(self) -> int | None:
        visited = [k for k in range(self.n) if self.recent[k]]
        if not visited:
            return None
        pot = self.potential()[visited] + self.cfg.min_potential   # każda odwiedzana sekcja ma szansę
        return int(self.rng.choice(visited, p=pot / pot.sum()))

    # --- po epizodzie ---------------------------------------------------------------
    def end_episode(self, trace: list[tuple[float, int]], finished: bool) -> list[tuple[int, int, int, float]]:
        """trace: (postęp [m], czas wyścigu [ms]) w każdym kroku. Aktualizuje statystyki sekcji.
        Zwraca poprawione sekcje: (sekcja, krok wjazdu, krok wyjazdu, nowy czas [ms])."""
        if not trace:
            return []
        prog = np.array([p for p, _ in trace])
        times = np.array([t for _, t in trace], dtype=float)
        final = self.track_len if finished else float(prog.max())
        improved = []
        for k in range(self.n):
            lo, hi = self.bounds[k], self.bounds[k + 1]
            if prog[0] > lo + 1.0:          # sekcja przed miejscem startu epizodu
                continue
            enter = int(np.argmax(prog >= lo)) if lo > 0 else 0
            if prog.max() < lo:
                break                        # do tej sekcji przejazd nie dojechał
            reached_end = finished if k == self.n - 1 else bool((prog >= hi).any())
            if not reached_end:
                self.recent[k].append(None)  # przejazd skończył się w tej sekcji
                break
            leave = len(trace) - 1 if k == self.n - 1 and not (prog >= hi).any() else int(np.argmax(prog >= hi))
            ms = float(times[leave] - times[enter])
            self.recent[k].append(ms)
            survived = finished or final >= hi + self.cfg.survive_after_m
            if ms < self.best_ms[k] and survived and leave > enter:
                self.best_ms[k] = ms
                improved.append((k, enter, leave, ms))
        return improved

    def section_times(self, trace: list[tuple[float, int]], finished: bool) -> list[float]:
        """Czasy sekcji jednego przejazdu (nan = sekcji nie przejechał) - do sections.csv."""
        out = [math.nan] * self.n
        if not trace:
            return out
        prog = np.array([p for p, _ in trace])
        times = np.array([t for _, t in trace], dtype=float)
        for k in range(self.n):
            lo, hi = self.bounds[k], self.bounds[k + 1]
            if prog.max() < lo:
                break
            reached = (prog >= hi).any() or (k == self.n - 1 and finished)
            if not reached:
                break
            enter = int(np.argmax(prog >= lo)) if lo > 0 else 0
            leave = int(np.argmax(prog >= hi)) if (prog >= hi).any() else len(trace) - 1
            out[k] = float(times[leave] - times[enter])
        return out

    # --- zapis ----------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"best_ms": [None if math.isinf(b) else b for b in self.best_ms],
                "recent": [list(r) for r in self.recent]}

    def load_state_dict(self, s: dict) -> None:
        if len(s.get("best_ms", [])) != self.n:
            return   # inna liczba sekcji - zaczynamy statystyki od nowa
        self.best_ms = [math.inf if b is None else float(b) for b in s["best_ms"]]
        for r, saved in zip(self.recent, s.get("recent", [])):
            r.extend(saved)
