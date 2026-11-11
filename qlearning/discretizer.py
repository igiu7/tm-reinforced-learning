"""Zamiana ciągłej obserwacji z TrackmaniaEnv na numer stanu tablicy Q."""

from __future__ import annotations

import math

import numpy as np

# Układ wektora obserwacji z tm_env.RLClient._evaluate (gdy istnieje linia referencyjna):
#   [0:3]   prędkość w układzie auta (bok, góra, przód) / 100   [m/s]
#   [3:5]   sin(pitch), sin(roll)
#   [5:9]   kontakt kół z podłożem (0/1)
#   [9:9+2k] dla każdej odległości z lookahead_m: (bok, przód) / 100 do punktu linii [m]
#   [-1]    odległość od linii / max_deviation_m
_VEL_SIDE, _VEL_FWD = 0, 2
_WHEELS = slice(5, 9)
_LOOKAHEAD_START = 9
_MS_TO_KMH = 100.0 * 3.6   # obs prędkości jest w (m/s)/100


class Discretizer:
    def __init__(self, cfg, lookahead_m: tuple, max_deviation_m: float, obs_dim: int):
        expected = _LOOKAHEAD_START + 2 * len(lookahead_m) + 1
        if obs_dim != expected:
            raise ValueError(
                f"Obserwacja ma {obs_dim} wymiarów, oczekiwano {expected}. Q-learning wymaga linii "
                "referencyjnej - nagraj ją: python record_line.py"
            )
        self.max_dev = max_deviation_m

        # (nazwa, funkcja obs -> wartość, granice przedziałów)
        self.features: list[tuple[str, callable, np.ndarray]] = []
        for dist, edges in cfg.angle_features:
            if dist not in lookahead_m:
                raise ValueError(f"Odległość {dist} m nie występuje w EnvConfig.lookahead_m={lookahead_m}")
            i = _LOOKAHEAD_START + 2 * lookahead_m.index(dist)
            self.features.append((f"kąt_{dist}m", self._angle_fn(i), np.asarray(edges)))

        self.features.append(("prędkość", lambda o: o[_VEL_FWD] * _MS_TO_KMH, np.asarray(cfg.speed_edges_kmh)))
        self.features.append(("odchylenie", lambda o: o[-1] * self.max_dev, np.asarray(cfg.deviation_edges_m)))
        if cfg.slip_edges_kmh:
            self.features.append(("poślizg", lambda o: o[_VEL_SIDE] * _MS_TO_KMH, np.asarray(cfg.slip_edges_kmh)))
        if cfg.use_airborne:
            self.features.append(("w_powietrzu", lambda o: float(o[_WHEELS].sum() == 0), np.asarray([0.5])))

        self.dims = tuple(len(edges) + 1 for _, _, edges in self.features)
        self.n_states = int(np.prod(self.dims))

    @staticmethod
    def _angle_fn(i: int):
        def angle(o):
            side, fwd = o[i], o[i + 1]
            return math.degrees(math.atan2(side, fwd))
        return angle

    def __call__(self, obs: np.ndarray) -> int:
        idx = tuple(int(np.digitize(fn(obs), edges)) for _, fn, edges in self.features)
        return int(np.ravel_multi_index(idx, self.dims))

    def signature(self) -> str:
        """Opis podziału stanów; zapisywany z tablicą Q, żeby nie wczytać niezgodnej tablicy."""
        return ";".join(f"{name}:{','.join(f'{e:g}' for e in edges)}" for name, _, edges in self.features)

    def describe(self) -> str:
        parts = " x ".join(f"{name}({d})" for (name, _, _), d in zip(self.features, self.dims))
        return f"{parts} = {self.n_states} stanów"
