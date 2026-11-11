"""
Katalog danych per mapa:

    maps/
      <nazwa>_<uid>/
        map.json               uid, nazwa, autor
        reference_line.npy     linia referencyjna tej mapy
        runs/<run_name>/       trening: agent.pt, best.pt, episodes.csv, training_plots.png, config.txt
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tmi2_bridge import MapInfo

LINE_FILE = "reference_line.npy"


def _safe(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return text[:40] or "mapa"


@dataclass
class RunSummary:
    exists: bool
    train_episodes: int = 0
    finishes: int = 0
    best_ms: int | None = None
    last_progress_pct: float | None = None


class MapStore:
    def __init__(self, root: str | Path = "maps"):
        self.root = Path(root)

    def map_dir(self, info: MapInfo, create: bool = False) -> Path:
        # najpierw szukamy po uid (nazwa mogła się zmienić / inaczej oczyścić)
        for d in self._map_dirs():
            if self._read_info(d).uid == info.uid:
                return d
        d = self.root / f"{_safe(info.name)}_{_safe(info.uid)[:12]}"
        if create:
            d.mkdir(parents=True, exist_ok=True)
            (d / "map.json").write_text(json.dumps(asdict(info), ensure_ascii=False, indent=2), encoding="utf-8")
        return d

    def line_path(self, info: MapInfo) -> Path:
        return self.map_dir(info, create=True) / LINE_FILE

    def run_dir(self, info: MapInfo, run_name: str) -> Path:
        return self.map_dir(info, create=True) / "runs" / run_name

    def has_line(self, info: MapInfo) -> bool:
        return (self.map_dir(info) / LINE_FILE).exists()

    def known_maps(self) -> list[tuple[MapInfo, Path]]:
        return [(self._read_info(d), d) for d in self._map_dirs()]

    def _map_dirs(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(d for d in self.root.iterdir() if (d / "map.json").exists())

    @staticmethod
    def _read_info(d: Path) -> MapInfo:
        data = json.loads((d / "map.json").read_text(encoding="utf-8"))
        return MapInfo(data.get("uid", ""), data.get("name", d.name), data.get("author", ""))


def summarize_run(run_dir: Path) -> RunSummary:
    """Krótki opis treningu z episodes.csv (bez wczytywania agenta)."""
    from rl.stats import read_log

    if not (run_dir / "agent.pt").exists():
        return RunSummary(exists=False)
    rows = read_log(run_dir / "episodes.csv")
    train = [r for r in rows if r["mode"] == "train"]
    fin = [r for r in rows if r["finished"]]
    return RunSummary(
        exists=True,
        train_episodes=max((r["episode"] for r in train), default=0),
        finishes=len(fin),
        best_ms=min((r["race_time_ms"] for r in fin), default=None),
        last_progress_pct=train[-1]["progress_pct"] if train else None,
    )
