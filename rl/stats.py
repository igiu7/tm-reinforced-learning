"""Statystyki epizodów: zapis do CSV i wypisywanie postępu."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class EpisodeStats:
    episode: int
    mode: str              # "train" albo "eval"
    steps: int
    reward: float
    race_time_ms: int
    reason: str            # finish / fell / off_track / stuck / no_progress / timeout
    finished: int
    progress_m: float
    progress_pct: float
    checkpoints: int
    avg_speed_kmh: float
    max_speed_kmh: float
    epsilon: float         # udział losowych akcji w tym epizodzie (0 w przejazdach testowych)
    loss: float            # średnia strata uczenia w epizodzie (nan, gdy nie było aktualizacji)
    q_mean: float          # średnie przewidywane Q w aktualizacjach (czy sieć nie "odlatuje")
    total_steps: int       # łączna liczba kroków treningowych agenta
    zone_start_pct: float  # strefa dodatkowej eksploracji (rl/exploration.py); nan = wyłączona
    zone_end_pct: float
    hits: int              # uderzenia (gwałtowne spadki prędkości) w epizodzie
    focus_section: int     # sekcja ćwiczona w tym epizodzie (rl/sections.py), 0 = żadna
    wall_time_s: float


FIELDS = [f.name for f in fields(EpisodeStats)]
_INT = ("episode", "steps", "race_time_ms", "finished", "checkpoints", "total_steps", "hits", "focus_section")
_FLOAT = ("reward", "progress_m", "progress_pct", "avg_speed_kmh", "max_speed_kmh",
          "epsilon", "loss", "q_mean", "zone_start_pct", "zone_end_pct", "wall_time_s")


class EpisodeLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        if not new:
            self._migrate_header()
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if new:
            self._writer.writeheader()

    def _migrate_header(self) -> None:
        """Log z wcześniejszej wersji programu (inne kolumny) -> przepisz go pod aktualne kolumny,
        żeby dopisywane wiersze pasowały do nagłówka. Brakujące wartości zostają puste."""
        with self.path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == FIELDS:
                return
            rows = list(reader)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in FIELDS})

    def log(self, st: EpisodeStats) -> None:
        row = asdict(st)
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = "" if math.isnan(v) else round(v, 5)
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
        for k in _INT:
            r[k] = int(float(r.get(k) or 0))
        for k in _FLOAT:
            v = r.get(k)
            r[k] = float(v) if v not in (None, "") else float("nan")
    return rows


def format_line(st: EpisodeStats, best_ms: int | None) -> str:
    tag = "EVAL " if st.mode == "eval" else "     "
    best = f"{best_ms / 1000:.2f}s" if best_ms else "-"
    loss = "-" if math.isnan(st.loss) else f"{st.loss:.4f}"
    zone = "" if math.isnan(st.zone_start_pct) else f" | strefa {st.zone_start_pct:.0f}-{st.zone_end_pct:.0f}%"
    if st.focus_section:
        zone += f" | ćwiczy sekcję {st.focus_section}"
    return (f"{tag}ep {st.episode:5d} | {st.reason:11s} | czas {st.race_time_ms / 1000:6.2f}s | "
            f"postęp {st.progress_pct:5.1f}% | nagroda {st.reward:7.1f} | v_śr {st.avg_speed_kmh:4.0f} km/h | "
            f"uderzenia {st.hits:2d} | eps {st.epsilon:.3f} | loss {loss} | kroki {st.total_steps:,} | najlepszy {best}{zone}")
