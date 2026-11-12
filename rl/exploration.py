"""
Adaptacyjna eksploracja w strefie porażki.

Problem: gdy losowość (epsilon) spadnie, agent powtarza tę samą jazdę i potrafi długo "zacinać się"
w jednym miejscu trasy (np. wzgórze, na które nie umie wjechać). Podniesienie epsilonu na całej trasie
szkodzi - losowe ruchy w miejscach, które agent już umie, powodują wypadki, więc trudne miejsce jest
ćwiczone jeszcze rzadziej.

Rozwiązanie: wykrywamy zastój (agent przestaje dojeżdżać dalej) i podnosimy epsilon TYLKO w strefie
przed miejscem, w którym najczęściej kończą się nieudane przejazdy. Na reszcie trasy agent jedzie
jak dotąd. Gdy w strefie zacznie się kończyć wyraźnie mniej przejazdów (albo strefa długo nic nie daje),
podbicie wygasa.

Moduł nie zależy od algorytmu: trener pyta go o epsilon dla bieżącego miejsca na trasie
i przekazuje wynik do agent.act(..., epsilon=...).
"""

from __future__ import annotations

from collections import deque

import numpy as np


class FailureZoneExploration:
    def __init__(self, cfg, track_len: float):
        self.cfg = cfg
        self.track_len = track_len
        self.recent: deque = deque(maxlen=cfg.window_episodes)   # (postęp [m], czy porażka)
        self.best_frontier = 0.0        # najlepszy dotąd "zasięg" agenta (percentyl postępu w oknie)
        self.since_improvement = 0      # epizody bez poprawy zasięgu
        self.zone: tuple[float, float] | None = None   # (początek, koniec) strefy w metrach
        self.zone_failure_m = 0.0       # miejsce porażki, wokół którego powstała strefa
        self.active_for = 0             # ile epizodów strefa jest aktywna
        self.share_at_start = 0.0       # udział epizodów kończących się w strefie, gdy ją włączano
        self.zone_results: deque = deque(maxlen=cfg.min_active_episodes)  # czy epizod skończył się w strefie

    # --- używane w każdym kroku -------------------------------------------------
    def epsilon(self, base_epsilon: float, progress_m: float) -> float:
        if self.zone and self.zone[0] <= progress_m <= self.zone[1]:
            return max(base_epsilon, self.cfg.boost_epsilon)
        return base_epsilon

    # --- po każdym epizodzie treningowym ----------------------------------------------
    def end_episode(self, progress_m: float, finished: bool, base_epsilon: float) -> str | None:
        """Aktualizuje statystyki; zwraca komunikat, gdy strefa się włącza lub wyłącza."""
        c = self.cfg
        self.recent.append((self.track_len if finished else progress_m, not finished))
        if len(self.recent) < c.window_episodes:
            return None

        frontier = float(np.percentile([p for p, _ in self.recent], 90))
        if frontier >= self.best_frontier + c.min_improvement_m:
            self.best_frontier = frontier
            self.since_improvement = 0
        else:
            self.since_improvement += 1

        if self.zone:
            self.active_for += 1
            self.zone_results.append(self._fails_in_zone(progress_m, not finished))
            # "Przebite" = w ostatnich min_active_episodes epizodach w strefie kończy się wyraźnie mniej przejazdów
            # niż przy włączaniu. (Nie patrzymy na najdalsze przejazdy: wystarczy, że co 10. przejedzie,
            # a agent nadal zwykle rozbija się w tym samym miejscu.)
            if len(self.zone_results) >= c.min_active_episodes:
                share = sum(self.zone_results) / len(self.zone_results)
                if share <= self.share_at_start * c.solved_share_ratio:
                    return self._deactivate(f"miejsce przebite - kończy się tam {share:.0%} przejazdów "
                                            f"(było {self.share_at_start:.0%})")
            if self.active_for >= c.max_active_episodes:
                return self._deactivate(f"brak efektu przez {self.active_for} epizodów")
            return None

        # włączamy dopiero po zakończeniu zwykłej eksploracji - wcześniej losowości i tak jest dużo
        if self.since_improvement >= c.patience_episodes and base_epsilon <= c.boost_epsilon:
            failures = [p for p, failed in self.recent if failed]
            if not failures:
                return None
            self.zone_failure_m = self._most_common_failure(failures)
            self.zone = (max(0.0, self.zone_failure_m - c.zone_before_m),
                         min(self.track_len, self.zone_failure_m + c.zone_after_m))
            self.active_for = 0
            self.since_improvement = 0
            self.zone_results.clear()
            self.share_at_start = sum(self._fails_in_zone(p, f) for p, f in self.recent) / len(self.recent)
            return (f"zastój - zwiększam eksplorację do {c.boost_epsilon:.2f} w strefie "
                    f"{self._pct(self.zone[0])}-{self._pct(self.zone[1])} trasy "
                    f"(najczęstsza porażka przy {self._pct(self.zone_failure_m)}, "
                    f"kończy się tam {self.share_at_start:.0%} przejazdów)")
        return None

    def _fails_in_zone(self, progress_m: float, failed: bool) -> bool:
        return bool(failed and self.zone and self.zone[0] <= progress_m <= self.zone[1])

    def _deactivate(self, why: str) -> str:
        self.zone = None
        self.active_for = 0
        self.since_improvement = 0
        return f"wyłączam dodatkową eksplorację: {why}"

    def _most_common_failure(self, failures: list[float]) -> float:
        """Środek najliczniejszego przedziału miejsc porażki (histogram co bin_m metrów)."""
        bin_m = self.cfg.failure_bin_m
        edges = np.arange(0.0, self.track_len + bin_m, bin_m)
        counts, edges = np.histogram(failures, bins=edges)
        k = int(np.argmax(counts))
        return float((edges[k] + edges[k + 1]) / 2)

    def _pct(self, m: float) -> str:
        return f"{100.0 * m / self.track_len:.0f}%"

    # --- do logów i zapisu --------------------------------------------------------
    def zone_pct(self) -> tuple[float, float]:
        if not self.zone:
            return float("nan"), float("nan")
        return 100.0 * self.zone[0] / self.track_len, 100.0 * self.zone[1] / self.track_len

    def state_dict(self) -> dict:
        return {"recent": list(self.recent), "best_frontier": self.best_frontier,
                "since_improvement": self.since_improvement, "zone": self.zone,
                "zone_failure_m": self.zone_failure_m, "active_for": self.active_for,
                "share_at_start": self.share_at_start, "zone_results": list(self.zone_results)}

    def load_state_dict(self, s: dict) -> None:
        self.recent.extend(tuple(x) for x in s.get("recent", []))
        self.best_frontier = s.get("best_frontier", 0.0)
        self.since_improvement = s.get("since_improvement", 0)
        self.zone = tuple(s["zone"]) if s.get("zone") else None
        self.zone_failure_m = s.get("zone_failure_m", 0.0)
        self.active_for = s.get("active_for", 0)
        self.share_at_start = s.get("share_at_start", 0.0)
        self.zone_results.extend(bool(x) for x in s.get("zone_results", []))
