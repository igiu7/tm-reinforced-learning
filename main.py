"""
Menu uczenia ze wzmocnieniem dla Trackmanii (TMInterface 2 + wtyczka RL Bridge).

    python main.py

Wykrywa mapę załadowaną w grze i pracuje tylko na danych tej mapy (maps/<mapa>/):
nagranie linii referencyjnej, nowy trening, kontynuacja, jazda wyuczonym agentem, wykresy, duchy.
Algorytm i parametry: rl_config.py
"""

from __future__ import annotations

import dataclasses
import importlib

import ghosts
import rl_config
from maps import MapStore, summarize_run
from plugin_install import install_plugin
from rl_config import CONFIG, TMI_PLUGINS_DIR, TMI_SCRIPTS_DIR
from rl import Trainer
from record_line import record_line
from tm_env import TrackmaniaEnv
from tmi2_bridge import BridgeError, MapInfo, TMI2Bridge

LINE = "=" * 64


# ---------------------------------------------------------------------------
# pomocnicze wejście z klawiatury
# ---------------------------------------------------------------------------
def ask(prompt: str) -> str:
    return input(prompt).strip()


def ask_int(prompt: str, default: int) -> int:
    while True:
        text = ask(f"{prompt} [{default}]: ")
        if not text:
            return default
        if text.isdigit() and int(text) > 0:
            return int(text)
        print("  Podaj dodatnią liczbę całkowitą.")


def confirm(prompt: str) -> bool:
    return ask(f"{prompt} [t/N]: ").lower() in ("t", "tak", "y", "yes")


def fmt_time(ms: int | None) -> str:
    return f"{ms / 1000:.2f} s" if ms else "-"


# ---------------------------------------------------------------------------
# aplikacja
# ---------------------------------------------------------------------------
class App:
    def __init__(self, cfg):
        self.cfg = cfg
        self.store = MapStore(cfg.train.maps_dir)
        self.bridge = TMI2Bridge(cfg.env.host, cfg.env.port)
        self.map: MapInfo | None = None

    # --- połączenie i wykrywanie mapy ---------------------------------------
    def refresh(self) -> None:
        """Łączy się z grą (jedna próba) i odczytuje aktualnie załadowaną mapę."""
        self.map = None
        if not self.bridge.connected and not self.bridge.connect(wait=False):
            return
        try:
            info = self.bridge.get_map_info()
            self.map = info if info.loaded else None
        except (BridgeError, OSError):
            self.drop_connection()

    def drop_connection(self) -> None:
        try:
            self.bridge.close()
        except OSError:
            pass
        self.bridge.sock = None

    def trainer(self, info: MapInfo) -> Trainer:
        return Trainer(self.cfg, self.store.run_dir(info, self.cfg.run_name))

    # --- ekran główny ---------------------------------------------------------
    def status(self) -> dict:
        s = {"connected": self.bridge.connected, "map": self.map, "line": False, "run": None, "ghosts": 0}
        if self.map:
            s["line"] = self.store.has_line(self.map)
            run_dir = self.store.map_dir(self.map) / "runs" / self.cfg.run_name
            run = summarize_run(run_dir)
            s["run"] = run if run.exists else None
            s["ghosts"] = len(ghosts.list_ghosts(run_dir))
        return s

    def print_header(self, s: dict) -> None:
        print(f"\n{LINE}\n  TrackMania RL - algorytm: {self.cfg.algorithm.upper()}   "
              f"(konfiguracja: rl_config.py, trening '{self.cfg.run_name}')\n{LINE}")
        if not s["connected"]:
            print(f"Gra:    NIEPOŁĄCZONA - uruchom TMNF z TMInterface 2 i wtyczką RL Bridge (port {self.bridge.port})")
        else:
            print(f"Gra:    połączona (port {self.bridge.port})")
            if not s["map"]:
                print("Mapa:   brak - załaduj mapę w grze i wybierz 'Odśwież'")
        if s["map"]:
            m = s["map"]
            print(f"Mapa:   {m.name}" + (f"  (autor: {m.author})" if m.author else ""))
            print(f"Linia referencyjna: {'jest' if s['line'] else 'BRAK - nagraj ją (opcja 1)'}")
            r = s["run"]
            if r:
                prog = f", ostatni postęp {r.last_progress_pct:.0f}%" if r.last_progress_pct is not None else ""
                print(f"Trening '{self.cfg.run_name}': {r.train_episodes} epizodów, "
                      f"{r.finishes} ukończonych, najlepszy czas {fmt_time(r.best_ms)}{prog}")
                print(f"Zapisane przejazdy do powtórek (duchy): {s['ghosts']}")
            else:
                print(f"Trening '{self.cfg.run_name}': brak dla tej mapy")
        print(LINE)

    def menu_items(self, s: dict):
        game = s["connected"] and s["map"] is not None
        line = game and s["line"]
        run = s["run"] is not None
        return [
            ("1", "Nagraj linię referencyjną", game, "wymaga gry z załadowaną mapą", self.do_record),
            ("2", "Nowy trening (od zera, zaczyna od nagrania linii)", game, "wymaga gry z załadowaną mapą",
             self.do_new_training),
            ("3", "Kontynuuj trening", line and run, "brak treningu dla tej mapy", self.do_continue),
            ("4", "Jazda wyuczonym agentem", line and run, "brak treningu dla tej mapy", self.do_play),
            ("5", "Wykresy tej mapy", run, "brak treningu dla tej mapy", self.do_plot_current),
            ("6", "Wszystkie mapy z treningiem", True, "", self.do_list_maps),
            ("7", "Odśwież (połącz / wykryj mapę)", True, "", lambda: None),
            ("8", "Duchy: odtwórz przejazdy agenta, żeby zapisać powtórki", game and s["ghosts"] > 0,
             "brak gry albo brak ukończonych przejazdów", self.do_ghosts),
            ("0", "Wyjście", True, "", None),
        ]

    def reload_config(self) -> None:
        """Wczytuje rl_config.py na nowo, żeby zmiany działały bez restartu menu."""
        global TMI_SCRIPTS_DIR
        try:
            module = importlib.reload(rl_config)
        except Exception as e:  # noqa: BLE001 - błąd w pliku konfiguracji nie może zamknąć menu
            print(f"[CONFIG] Błąd w rl_config.py - zostaję przy poprzednich ustawieniach: {e}")
            return
        self.cfg = module.CONFIG
        TMI_SCRIPTS_DIR = module.TMI_SCRIPTS_DIR
        self.store = MapStore(self.cfg.train.maps_dir)

    def run(self) -> None:
        while True:
            self.reload_config()
            self.refresh()
            s = self.status()
            self.print_header(s)
            items = self.menu_items(s)
            for key, label, enabled, why, _ in items:
                print(f"  {key}. {label}" + ("" if enabled else f"   ({why})"))
            choice = ask("\nWybierz: ")
            item = next((it for it in items if it[0] == choice), None)
            if item is None:
                print("Nie ma takiej opcji.")
                continue
            key, label, enabled, why, action = item
            if action is None:
                return
            if not enabled:
                print(f"Niedostępne: {why}.")
                continue
            action()

    # --- akcje wymagające gry -------------------------------------------------
    def _current_map_unchanged(self) -> MapInfo | None:
        """Sprawdza mapę tuż przed startem - mogła się zmienić, gdy menu czekało na wybór."""
        before = self.map
        self.refresh()
        if self.map is None or before is None or self.map.uid != before.uid:
            print("Mapa w grze zmieniła się (albo gra się rozłączyła) - sprawdź menu jeszcze raz.")
            return None
        return self.map

    def _with_env(self, info: MapInfo, game_speed: float, fn) -> None:
        env_cfg = dataclasses.replace(self.cfg.env, game_speed=game_speed,
                                      reference_line=str(self.store.line_path(info)))
        print("Jeśli nic się nie dzieje: w grze musi trwać wyścig na tej mapie (nie menu ani pauza).")
        env = None
        try:
            env = TrackmaniaEnv(env_cfg, continuous=False, bridge=self.bridge)
            fn(env)
        except KeyboardInterrupt:
            print("\nPrzerwano natychmiast - łączę się z grą od nowa.")
            self.drop_connection()
        except (BridgeError, OSError) as e:
            print(f"\nUtracono połączenie z grą: {e}")
            self.drop_connection()
        except (RuntimeError, ValueError) as e:
            print(f"\nBłąd: {e}")
            if env is not None:
                env.close()
        else:
            env.close()

    def _record(self, info: MapInfo) -> bool:
        """Nagrywa linię referencyjną mapy. Zwraca True, jeśli została zapisana."""
        print(f"\nNagrywanie linii referencyjnej dla mapy: {info.name}")
        print("Przejedź trasę sam, czysto i środkiem drogi - agent będzie się uczył wzdłuż tej linii.")
        try:
            return record_line(self.bridge, self.store.line_path(info))
        except KeyboardInterrupt:
            self.drop_connection()
        except (BridgeError, OSError) as e:
            print(f"Utracono połączenie z grą: {e}")
            self.drop_connection()
        return False

    def do_record(self) -> None:
        info = self._current_map_unchanged()
        if not info:
            return
        if self.store.has_line(info) and not confirm("Ta mapa ma już linię referencyjną. Nagrać nową?"):
            return
        self._record(info)

    def do_new_training(self) -> None:
        info = self._current_map_unchanged()
        if not info:
            return
        tr = self.trainer(info)
        if tr.has_saved and not confirm("Ta mapa ma już trening. Przenieść go do kopii zapasowej i zacząć od zera?"):
            return

        # krok 1: linia referencyjna - nowy trening zawsze zaczyna się od niej
        if not self.store.has_line(info):
            print("\nKrok 1/2: ta mapa nie ma linii referencyjnej - trzeba ją teraz nagrać.")
            if not self._record(info):
                print("Bez linii referencyjnej trening nie wystartuje. Wracam do menu.")
                return
        elif confirm("\nKrok 1/2: mapa ma już linię referencyjną. Nagrać nową przed treningiem?"):
            if not self._record(info):
                print("Nagrywanie anulowane - nowy trening nie wystartował. Wracam do menu.")
                return

        # krok 2: trening
        if tr.has_saved:
            print(f"Stary trening przeniesiony do: {tr.archive()}")
        print("\nKrok 2/2: trening.")
        episodes = ask_int("Ile epizodów?", self.cfg.train.episodes)
        self._with_env(info, self.cfg.env.game_speed, lambda env: tr.train(env, episodes, resume=False))

    def do_continue(self) -> None:
        info = self._current_map_unchanged()
        if not info:
            return
        tr = self.trainer(info)
        episodes = ask_int("Ile kolejnych epizodów?", self.cfg.train.episodes)
        self._with_env(info, self.cfg.env.game_speed, lambda env: tr.train(env, episodes, resume=True))

    def do_play(self) -> None:
        info = self._current_map_unchanged()
        if not info:
            return
        tr = self.trainer(info)
        self._with_env(info, self.cfg.play_game_speed, tr.play)

    def do_ghosts(self) -> None:
        info = self._current_map_unchanged()
        if not info:
            return
        run_dir = self.store.run_dir(info, self.cfg.run_name)
        all_ghosts = ghosts.list_ghosts(run_dir)
        records = ghosts.select_records(all_ghosts)
        print(f"\nUkończone przejazdy: {len(all_ghosts)}  (najlepszy {fmt_time(min(g.time_ms for g in all_ghosts))})")
        print(f"  1. Kolejne rekordy - jak agent się poprawiał ({len(records)} przejazdów)")
        print("  2. Najlepsze N przejazdów")
        print("  3. N przejazdów równo rozłożonych w czasie nauki")
        print(f"  4. Wszystkie ({len(all_ghosts)})")
        choice = ask("Wybierz (Enter = powrót): ")
        if choice == "1":
            chosen = records
        elif choice == "2":
            chosen = ghosts.select_best(all_ghosts, ask_int("Ile?", min(10, len(all_ghosts))))
        elif choice == "3":
            chosen = ghosts.select_spread(all_ghosts, ask_int("Ile?", min(10, len(all_ghosts))))
        elif choice == "4":
            chosen = all_ghosts
        else:
            return
        print("Wybrane: " + ", ".join(g.label for g in chosen))
        if not confirm(f"Odtworzyć {len(chosen)} przejazdów w grze?"):
            return
        map_tag = self.store.map_dir(info).name[:24]
        try:
            ghosts.playback(self.bridge, chosen, map_tag, TMI_SCRIPTS_DIR, ask=ask)
        except KeyboardInterrupt:
            self.drop_connection()
        except (BridgeError, OSError) as e:
            print(f"Utracono połączenie z grą: {e}")
            self.drop_connection()

    # --- akcje bez gry --------------------------------------------------------
    def do_plot_current(self) -> None:
        self.trainer(self.map).plot(open_file=True)

    def do_list_maps(self) -> None:
        rows = []
        for info, d in self.store.known_maps():
            run = summarize_run(d / "runs" / self.cfg.run_name)
            rows.append((info, d, run))
        if not rows:
            print("Brak zapisanych map. Nagraj linię referencyjną na jakiejś mapie (opcja 1).")
            return
        print(f"\n{'nr':>3}  {'mapa':30} {'linia':6} {'epizody':>8} {'ukończone':>10} {'najlepszy':>10}")
        for i, (info, d, run) in enumerate(rows, 1):
            line = "jest" if (d / "reference_line.npy").exists() else "brak"
            cur = " <- w grze" if self.map and info.uid == self.map.uid else ""
            print(f"{i:>3}  {info.name[:30]:30} {line:6} {run.train_episodes:>8} {run.finishes:>10} "
                  f"{fmt_time(run.best_ms):>10}{cur}")
        choice = ask("\nNumer mapy, aby otworzyć wykresy (Enter = powrót): ")
        if choice.isdigit() and 1 <= int(choice) <= len(rows):
            info, d, run = rows[int(choice) - 1]
            if not run.exists:
                print("Ta mapa nie ma jeszcze treningu.")
                return
            Trainer(self.cfg, d / "runs" / self.cfg.run_name).plot(open_file=True)


def main():
    install_plugin(TMI_PLUGINS_DIR)   # przed połączeniem: podmiana pliku przeładowuje wtyczkę w grze
    app = App(CONFIG)
    try:
        app.run()
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        app.drop_connection()


if __name__ == "__main__":
    main()
