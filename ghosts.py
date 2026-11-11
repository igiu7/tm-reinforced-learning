"""
Duchy: ukończone przejazdy agenta zapisane jako skrypty TMInterface (runs/<run>/ghosts/*.txt).

Odtworzenie w grze: skrypt trafia do folderu skryptów TMInterface, potem `load <plik>` i restart
wyścigu - gra odtwarza dokładnie te same wejścia, a po mecie zapisujesz powtórkę ("Save replay").
Zapisane powtórki dodajesz jako duchy w edytorze powtórek (MediaTracker).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from plugin_install import documents_dir
from tmi2_bridge import TMI2Bridge

_NAME = re.compile(r"ep(\d+)_(train|eval)_(\d+)ms\.txt$")

MEDIATRACKER_HELP = """\
Jak zrobić jedną powtórkę z wieloma duchami (w grze):
  1. Menu główne gry -> Powtórki (Replays) -> otwórz jedną z zapisanych powtórek -> Edytuj (Edit).
  2. W MediaTrackerze dodaj nowy ślad typu Duch (Ghost) i wskaż kolejną zapisaną powtórkę.
     Powtórz dla każdej powtórki, którą chcesz widzieć na torze.
  3. Zapisz powtórkę - przy odtwarzaniu wszystkie auta jadą naraz.
Wskazówka: nazywaj powtórki numerem epizodu (np. ep00250), wtedy łatwo je rozpoznać."""


@dataclass(frozen=True)
class Ghost:
    path: Path
    episode: int
    mode: str        # train / eval
    time_ms: int

    @property
    def label(self) -> str:
        return f"ep{self.episode:05d} ({self.mode}, {self.time_ms / 1000:.2f} s)"


def list_ghosts(run_dir: Path) -> list[Ghost]:
    d = Path(run_dir) / "ghosts"
    if not d.exists():
        return []
    out = []
    for p in d.glob("*.txt"):
        m = _NAME.match(p.name)
        if m:
            out.append(Ghost(p, int(m.group(1)), m.group(2), int(m.group(3))))
    return sorted(out, key=lambda g: (g.episode, g.mode))


# --- wybór -----------------------------------------------------------------
def select_records(ghosts: list[Ghost]) -> list[Ghost]:
    """Kolejne rekordy w czasie nauki - najlepiej pokazują, jak agent się poprawiał."""
    out, best = [], None
    for g in ghosts:
        if best is None or g.time_ms < best:
            best = g.time_ms
            out.append(g)
    return out


def select_best(ghosts: list[Ghost], n: int) -> list[Ghost]:
    return sorted(sorted(ghosts, key=lambda g: g.time_ms)[:n], key=lambda g: g.episode)


def select_spread(ghosts: list[Ghost], n: int) -> list[Ghost]:
    """n przejazdów równo rozłożonych w czasie treningu (od pierwszego do ostatniego)."""
    if n >= len(ghosts):
        return list(ghosts)
    if n == 1:
        return [ghosts[-1]]
    idx = sorted({round(i * (len(ghosts) - 1) / (n - 1)) for i in range(n)})
    return [ghosts[i] for i in idx]


# --- odtwarzanie -----------------------------------------------------------
def scripts_dir(override: str | None = None) -> Path:
    return Path(override) if override else documents_dir() / "TMInterface" / "Scripts"


def playback(bridge: TMI2Bridge, ghosts: list[Ghost], map_tag: str, scripts_override: str | None,
             ask=input) -> None:
    """Odtwarza kolejne przejazdy w grze; po każdym czeka, aż zapiszesz powtórkę."""
    target = scripts_dir(scripts_override)
    target.mkdir(parents=True, exist_ok=True)
    print(f"\nSkrypty kopiuję do: {target}")
    print("Dla każdego przejazdu: gra sama zrestartuje wyścig i odtworzy jazdę agenta.\n"
          "Po mecie zapisz powtórkę w grze (Save replay), potem wróć tutaj i naciśnij Enter.\n")
    try:
        for i, g in enumerate(ghosts, 1):
            name = f"rl_{map_tag}_{g.path.name}"
            shutil.copyfile(g.path, target / name)
            bridge.execute(f"load {name}")
            bridge.give_up()      # restart wyścigu -> TMInterface odtwarza wczytany skrypt
            print(f"[{i}/{len(ghosts)}] Odtwarzam {g.label} - zapisz powtórkę jako np. ep{g.episode:05d}")
            if ask("  Enter = następny, q = zakończ: ").strip().lower() == "q":
                break
    finally:
        bridge.execute("unload")  # żeby skrypt nie sterował autem podczas treningu
    print("\n" + MEDIATRACKER_HELP)
