"""Statystyki epizodów: zapis do CSV i wypisywanie postępu."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class EpisodeStats:
    episode: int
    mode: str              # "train" albo "eval"
    steps: int
    reward: float
    race_time_ms: int
    reason: str            # finish / fell / off_track / stuck / timeout
    finished: int
    progress_m: float
    progress_pct: float
    checkpoints: int
    avg_speed_kmh: float
    max_speed_kmh: float
    epsilon: float
    alpha: float
    visited_states_pct: float
    wall_time_s: float


FIELDS = [f.name for f in fields(EpisodeStats)]


class EpisodeLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if new:
            self._writer.writeheader()

    def log(self, st: EpisodeStats) -> None:
        row = asdict(st)
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 4)
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("episode", "steps", "race_time_ms", "finished", "checkpoints"):
            r[k] = int(float(r[k]))
        for k in ("reward", "progress_m", "progress_pct", "avg_speed_kmh", "max_speed_kmh",
                  "epsilon", "alpha", "visited_states_pct", "wall_time_s"):
            r[k] = float(r[k])
    return rows


def format_line(st: EpisodeStats, best_ms: int | None) -> str:
    tag = "EVAL " if st.mode == "eval" else "     "
    best = f"{best_ms / 1000:.2f}s" if best_ms else "-"
    return (f"{tag}ep {st.episode:5d} | {st.reason:9s} | czas {st.race_time_ms / 1000:6.2f}s | "
            f"postęp {st.progress_pct:5.1f}% | nagroda {st.reward:8.1f} | "
            f"v_śr {st.avg_speed_kmh:5.0f} km/h | eps {st.epsilon:.3f} | "
            f"stany {st.visited_states_pct:5.1f}% | najlepszy {best}")
