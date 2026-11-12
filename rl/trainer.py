"""
Pętla treningu i jazdy - wspólna dla wszystkich algorytmów (zna tylko interfejs rl.agents.base.Agent).
Środowisko (połączenie z grą) dostarcza wywołujący (menu).

Pliki w katalogu treningu (maps/<mapa>/runs/<run_name>/):
    agent.pt            ostatni stan agenta (do kontynuacji)
    best.pt             agent z najlepszym czasem przejazdu (do jazdy pokazowej)
    replay_buffer.npz   pamięć doświadczeń (DQN), żeby wznowiony trening nie zaczynał od zera
    episodes.csv        wynik każdego epizodu;  training_plots.png  wykresy;  config.txt  użyte parametry
    ghosts/             wejścia ukończonych przejazdów (menu -> Duchy)
"""

from __future__ import annotations

import math
import os
import shutil
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from tm_env import TrackmaniaEnv

from .agents import Agent, make_agent
from .exploration import FailureZoneExploration
from .sections import SectionTracker
from .plots import plot_training
from .stats import EpisodeLogger, EpisodeStats, format_line, read_log
from .stop import GracefulStop

AGENT_FILE = "agent.pt"
BEST_FILE = "best.pt"


class Trainer:
    def __init__(self, cfg, run_dir: Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.agent_path = self.run_dir / AGENT_FILE
        self.best_path = self.run_dir / BEST_FILE
        self.csv_path = self.run_dir / "episodes.csv"
        self.plot_path = self.run_dir / "training_plots.png"
        self.actions = tuple(cfg.actions)

        self.env: TrackmaniaEnv | None = None
        self.agent: Agent | None = None
        self.explorer: FailureZoneExploration | None = None
        self.sections: SectionTracker | None = None
        self.sections_csv = self.run_dir / "sections.csv"
        self._training = False
        self.track_len = 1.0
        self.episode = 0       # licznik epizodów treningowych (ciągły między uruchomieniami)
        self.best_ms: int | None = None

    @property
    def has_saved(self) -> bool:
        return self.agent_path.exists()

    def archive(self) -> Path | None:
        """Przenosi istniejący trening do kopii zapasowej (przed treningiem od zera)."""
        if not self.run_dir.exists():
            return None
        backup = self.run_dir.with_name(f"{self.run_dir.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(self.run_dir), str(backup))
        return backup

    # --- przygotowanie ------------------------------------------------------
    def _prepare(self, env: TrackmaniaEnv, load_from: Path | None, with_buffer: bool = False):
        self.env = env
        if env.line is None:
            raise RuntimeError("Brak linii referencyjnej dla tej mapy - nagraj ją najpierw.")
        self.track_len = env.line.length

        algo = self.cfg.algorithm
        self.agent = make_agent(algo, env.observation_space.shape[0], len(self.actions),
                                self.cfg.agent_config, seed=self.cfg.train.seed)
        weights = [self.cfg.random_action_weights.get(a, 1.0) for a in self.actions]
        self.agent.set_random_action_probs(weights)
        probs = ", ".join(f"{a}:{p:.0%}" for a, p in zip(self.actions, self.agent.random_action_probs))
        print(f"[RL] {self.agent.describe()}  |  akcje: {self.actions}  |  losowanie akcji: {probs}")

        sc = self.cfg.sections
        self.sections = SectionTracker(sc, self.track_len) if sc.enabled else None
        ex = self.cfg.exploration
        # strefa porażki i ćwiczenie sekcji robią podobną rzecz - przy włączonych sekcjach strefa jest pomijana
        self.explorer = FailureZoneExploration(ex, self.track_len) if ex.enabled and not self.sections else None

        self.episode, self.best_ms = 0, None
        if load_from is not None:
            meta = self.agent.load(load_from, with_buffer=with_buffer)
            self.episode = meta.get("episode", 0)
            self.best_ms = meta.get("best_ms")
            if self.explorer and meta.get("exploration"):
                self.explorer.load_state_dict(meta["exploration"])
            if self.sections and meta.get("sections"):
                self.sections.load_state_dict(meta["sections"])
            best = f"{self.best_ms / 1000:.2f} s" if self.best_ms else "-"
            print(f"[RL] Wczytano {load_from.name}: epizod {self.episode}, najlepszy czas {best}")

    def _meta(self) -> dict:
        return {"algorithm": self.cfg.algorithm, "actions": list(self.actions),
                "episode": self.episode, "best_ms": self.best_ms,
                "exploration": self.explorer.state_dict() if self.explorer else None,
                "sections": self.sections.state_dict() if self.sections else None}

    def _save(self, with_buffer: bool = False):
        self.agent.save(self.agent_path, self._meta(), with_buffer=with_buffer)

    # --- jeden epizod -------------------------------------------------------
    def run_episode(self, learn: bool) -> EpisodeStats:
        env, agent = self.env, self.agent
        t0 = time.perf_counter()
        obs, info = env.reset()
        agent.start_episode()
        explorer = self.explorer if learn else None
        zone = explorer.zone_pct() if explorer else (math.nan, math.nan)
        sections = self.sections
        if sections:
            if learn:
                sections.start_episode(agent.epsilon)
            else:
                sections.focus = None
        focus = sections.focus if sections else None

        total, steps, speeds = 0.0, 0, []
        trace: list[tuple[float, int]] = []    # (postęp, czas) przed każdą akcją - do czasów sekcji
        transitions: list = []                 # przejścia - najlepsze kawałki trafiają do pamięci agenta
        done = False
        while not done:
            p = info.get("progress_m", 0.0)
            trace.append((p, int(info.get("race_time", 0))))
            if focus is not None:
                eps = sections.epsilon(agent.epsilon, p)
            else:
                eps = explorer.epsilon(agent.epsilon, p) if explorer else None
            a = agent.act(obs, explore=learn, epsilon=eps)
            next_obs, r, terminated, truncated, info = env.step(self.actions[a])
            if learn:
                agent.observe(obs, a, r, next_obs, terminated, truncated)
            if sections:
                transitions.append((obs, a, r, next_obs, terminated, truncated))
            obs = next_obs
            done = terminated or truncated
            total += r
            steps += 1
            speeds.append(info["speed_kmh"])
        agent.end_episode()

        finished = info["reason"] == "finish"
        if finished and self.cfg.train.save_ghosts:
            self._save_ghost(learn, int(info["race_time"]))
        if sections:
            trace.append((info.get("progress_m", 0.0), int(info["race_time"])))
            if self._training:
                self._log_sections(learn, sections.section_times(trace, finished))
            improved = sections.end_episode(trace, finished)
            if self._training:
                for k, enter, leave, _ in improved:
                    agent.add_elite(transitions[enter:leave])
            if improved and self._training and focus is not None:
                names = ", ".join(f"{k + 1} ({ms / 1000:.2f} s)" for k, _, _, ms in improved)
                print(f"[RL] Nowy najlepszy czas sekcji: {names}")

        s = agent.stats()
        return EpisodeStats(
            episode=self.episode, mode="train" if learn else "eval", steps=steps, reward=total,
            race_time_ms=int(info["race_time"]), reason=info["reason"] or "?", finished=int(finished),
            progress_m=info["progress_m"],
            progress_pct=100.0 if finished else min(100.0, 100.0 * info["progress_m"] / self.track_len),
            checkpoints=info["checkpoints"], avg_speed_kmh=float(np.mean(speeds)) if speeds else 0.0,
            max_speed_kmh=float(np.max(speeds)) if speeds else 0.0,
            epsilon=s.get("epsilon", 0.0) if learn else 0.0,
            loss=s.get("loss", math.nan) if learn else math.nan,
            q_mean=s.get("q_mean", math.nan) if learn else math.nan,
            total_steps=s.get("total_steps", 0), zone_start_pct=zone[0], zone_end_pct=zone[1],
            hits=int(info.get("hits", 0)), focus_section=0 if focus is None else focus + 1,
            wall_time_s=time.perf_counter() - t0,
        )

    def _log_sections(self, learn: bool, times: list[float]):
        """sections.csv: czas każdej sekcji w każdym przejeździe [ms] (puste = nie przejechał)."""
        new = not self.sections_csv.exists()
        self.sections_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.sections_csv.open("a", encoding="utf-8") as f:
            if new:
                f.write("episode,mode," + ",".join(f"s{k + 1}" for k in range(len(times))) + "\n")
            f.write(f"{self.episode},{'train' if learn else 'eval'},"
                    + ",".join("" if math.isnan(t) else str(int(t)) for t in times) + "\n")

    def _save_ghost(self, learn: bool, race_time_ms: int):
        """Zapisuje wejścia ukończonego przejazdu jako skrypt TMInterface (menu -> Duchy)."""
        ghost_dir = self.run_dir / "ghosts"
        ghost_dir.mkdir(parents=True, exist_ok=True)
        mode = "train" if learn else "eval"
        path = ghost_dir / f"ep{self.episode:05d}_{mode}_{race_time_ms}ms.txt"
        path.write_text(self.env.episode_inputs_script() + "\n", encoding="utf-8")

    def _record(self, st: EpisodeStats, logger: EpisodeLogger):
        if st.finished and (self.best_ms is None or st.race_time_ms < self.best_ms):
            self.best_ms = st.race_time_ms
            self.agent.save(self.best_path, self._meta())
            print(f"[RL] *** Nowy najlepszy czas: {st.race_time_ms / 1000:.2f} s (zapisano {BEST_FILE}) ***")
        logger.log(st)
        print(format_line(st, self.best_ms))

    # --- tryby --------------------------------------------------------------
    def train(self, env: TrackmaniaEnv, episodes: int, resume: bool):
        """Trenuje `episodes` epizodów. resume=True kontynuuje zapisany trening."""
        tc = self.cfg.train
        self._prepare(env, self.agent_path if resume and self.has_saved else None,
                      with_buffer=tc.save_replay_buffer)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.txt").write_text(_dump_config(self.cfg, env.cfg), encoding="utf-8")
        logger = EpisodeLogger(self.csv_path)
        start = self.episode
        end = start + episodes
        print(f"[RL] Trening {episodes} epizodów: {start + 1}..{end}. Ctrl+C = zakończ po bieżącym epizodzie.")

        stop = None
        error: BaseException | None = None
        self._training = True
        try:
            with GracefulStop() as stop:
                while self.episode < end and not stop.requested:
                    self.episode += 1
                    st = self.run_episode(learn=True)
                    self._record(st, logger)
                    if self.explorer:
                        msg = self.explorer.end_episode(st.progress_m, bool(st.finished), self.agent.epsilon)
                        if msg:
                            print(f"[RL] Eksploracja: {msg}")

                    if tc.eval_every and self.episode % tc.eval_every == 0 and not stop.requested:
                        self._record(self.run_episode(learn=False), logger)
                    if tc.checkpoint_every and self.episode % tc.checkpoint_every == 0:
                        self._save()
                    if tc.plot_every and self.episode % tc.plot_every == 0:
                        self._try_plot()
        except BaseException as e:
            error = e
            raise
        finally:
            self._training = False
            self._save(with_buffer=tc.save_replay_buffer)
            logger.close()
            if self.episode >= end:
                why = "zadana liczba epizodów"
            elif error is not None:
                why = f"błąd {type(error).__name__}: {error}"
            elif stop is not None and stop.requested:
                why = "przerwano Ctrl+C"
            else:
                why = "nieznany powód"
            msg = f"Koniec treningu: {self.episode - start} z {episodes} epizodów ({why})"
            print(f"[RL] {msg}. Zapisano (epizod {self.episode}) w {self.run_dir}")
            self._log_session(f"{msg}; epizody {start + 1}..{end}", error)
            self._try_plot(open_file=tc.open_plot_at_end)

    def _try_plot(self, open_file: bool = False):
        """Wykres nie może przerwać treningu - błąd (np. zablokowany plik PNG) tylko wypisujemy."""
        try:
            self.plot(open_file=open_file) if open_file else plot_training(
                self.csv_path, self.plot_path, self.cfg.train.moving_avg)
        except Exception as e:  # noqa: BLE001
            print(f"[RL] Uwaga: nie udało się zapisać wykresu ({type(e).__name__}: {e}) - trening trwa dalej.")
            self._log_session("nie udało się zapisać wykresu", e)

    def _log_session(self, text: str, error: BaseException | None = None):
        """Dziennik treningu (training_log.txt): kiedy i dlaczego trening się skończył, z pełnym błędem."""
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with (self.run_dir / "training_log.txt").open("a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
                if error is not None and not isinstance(error, KeyboardInterrupt):
                    f.write("".join(traceback.format_exception(error)) + "\n")
        except OSError:
            pass

    def play(self, env: TrackmaniaEnv):
        """Jazda wyuczoną polityką (bez eksploracji i nauki), aż do Ctrl+C."""
        use_best = self.cfg.train.play_best and self.best_path.exists()
        self._prepare(env, self.best_path if use_best else self.agent_path)
        print("[RL] Jazda agentem " + ("z najlepszym czasem" if use_best else "z ostatniego zapisu")
              + ". Ctrl+C = zakończ po bieżącym przejeździe.")
        with GracefulStop("Zakończę po bieżącym przejeździe (Ctrl+C jeszcze raz = natychmiast).") as stop:
            while not stop.requested:
                st = self.run_episode(learn=False)
                print(f"Koniec: {st.reason}, czas {st.race_time_ms / 1000:.2f} s, postęp {st.progress_pct:.1f}%")

    def plot(self, open_file: bool = False):
        out = plot_training(self.csv_path, self.plot_path, self.cfg.train.moving_avg)
        if out:
            print(f"[RL] Wykresy: {out.resolve()}")
            rows = read_log(self.csv_path)
            fin = [r for r in rows if r["finished"]]
            if fin:
                best = min(fin, key=lambda r: r["race_time_ms"])
                print(f"[RL] Najlepszy czas {best['race_time_ms'] / 1000:.2f} s (epizod {best['episode']}, "
                      f"{best['mode']}), ukończone: {len(fin)}/{len(rows)}")
            if open_file and os.name == "nt":
                # Otwieramy KOPIĘ: przeglądarka zdjęć blokuje otwarty plik i kolejne zapisy wykresu by się nie udawały.
                view = self._viewer_copy(out)
                if view:
                    os.startfile(view)  # noqa: S606 - otwiera PNG w domyślnej przeglądarce obrazów
        return out

    def _viewer_copy(self, src: Path) -> Path | None:
        for name in ("training_plots_podglad.png", f"training_plots_podglad_{time.strftime('%H%M%S')}.png"):
            try:
                dst = src.with_name(name)
                shutil.copyfile(src, dst)
                return dst
            except OSError:
                continue   # poprzednia kopia nadal otwarta w przeglądarce -> nowa nazwa
        return None


def _dump_config(cfg, env_cfg) -> str:
    def section(name, obj):
        items = asdict(obj).items() if is_dataclass(obj) else vars(obj).items()
        return [f"[{name}]"] + [f"{k} = {v}" for k, v in items] + [""]

    lines = [f"algorithm = {cfg.algorithm}", f"actions = {tuple(cfg.actions)}",
             f"random_action_weights = {dict(cfg.random_action_weights)}", ""]
    lines += (section("env", env_cfg) + section(cfg.algorithm, cfg.agent_config)
              + section("exploration", cfg.exploration) + section("sections", cfg.sections)
              + section("train", cfg.train))
    return "\n".join(lines)
