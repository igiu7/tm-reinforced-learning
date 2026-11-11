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
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from tm_env import TrackmaniaEnv

from .agents import Agent, make_agent
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
        print(f"[RL] {self.agent.describe()}  |  akcje: {self.actions}")

        self.episode, self.best_ms = 0, None
        if load_from is not None:
            meta = self.agent.load(load_from, with_buffer=with_buffer)
            self.episode = meta.get("episode", 0)
            self.best_ms = meta.get("best_ms")
            best = f"{self.best_ms / 1000:.2f} s" if self.best_ms else "-"
            print(f"[RL] Wczytano {load_from.name}: epizod {self.episode}, najlepszy czas {best}")

    def _meta(self) -> dict:
        return {"algorithm": self.cfg.algorithm, "actions": list(self.actions),
                "episode": self.episode, "best_ms": self.best_ms}

    def _save(self, with_buffer: bool = False):
        self.agent.save(self.agent_path, self._meta(), with_buffer=with_buffer)

    # --- jeden epizod -------------------------------------------------------
    def run_episode(self, learn: bool) -> EpisodeStats:
        env, agent = self.env, self.agent
        t0 = time.perf_counter()
        obs, info = env.reset()
        agent.start_episode()

        total, steps, speeds = 0.0, 0, []
        done = False
        while not done:
            a = agent.act(obs, explore=learn)
            next_obs, r, terminated, truncated, info = env.step(self.actions[a])
            if learn:
                agent.observe(obs, a, r, next_obs, terminated, truncated)
            obs = next_obs
            done = terminated or truncated
            total += r
            steps += 1
            speeds.append(info["speed_kmh"])
        agent.end_episode()

        finished = info["reason"] == "finish"
        if finished and self.cfg.train.save_ghosts:
            self._save_ghost(learn, int(info["race_time"]))

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
            total_steps=s.get("total_steps", 0), wall_time_s=time.perf_counter() - t0,
        )

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
        end = self.episode + episodes
        print(f"[RL] Trening epizodów {self.episode + 1}..{end}. Ctrl+C = zakończ po bieżącym epizodzie.")

        try:
            with GracefulStop() as stop:
                while self.episode < end and not stop.requested:
                    self.episode += 1
                    self._record(self.run_episode(learn=True), logger)

                    if tc.eval_every and self.episode % tc.eval_every == 0 and not stop.requested:
                        self._record(self.run_episode(learn=False), logger)
                    if tc.checkpoint_every and self.episode % tc.checkpoint_every == 0:
                        self._save()
                    if tc.plot_every and self.episode % tc.plot_every == 0:
                        plot_training(self.csv_path, self.plot_path, tc.moving_avg)
        finally:
            self._save(with_buffer=tc.save_replay_buffer)
            logger.close()
            print(f"[RL] Zapisano trening (epizod {self.episode}) w {self.run_dir}")
            self.plot(open_file=tc.open_plot_at_end)

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
                os.startfile(out)  # noqa: S606 - otwiera PNG w domyślnej przeglądarce obrazów
        return out


def _dump_config(cfg, env_cfg) -> str:
    def section(name, obj):
        items = asdict(obj).items() if is_dataclass(obj) else vars(obj).items()
        return [f"[{name}]"] + [f"{k} = {v}" for k, v in items] + [""]

    lines = [f"algorithm = {cfg.algorithm}", f"actions = {tuple(cfg.actions)}", ""]
    lines += section("env", env_cfg) + section(cfg.algorithm, cfg.agent_config) + section("train", cfg.train)
    return "\n".join(lines)
