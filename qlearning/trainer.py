"""Pętla treningu i jazdy testowej. Środowisko (połączenie z grą) dostarcza wywołujący."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tm_env import TrackmaniaEnv

from .agent import QLearningAgent
from .discretizer import Discretizer
from .plots import plot_training
from .schedules import linear
from .stats import EpisodeLogger, EpisodeStats, format_line, read_log
from .stop import GracefulStop


class Trainer:
    def __init__(self, cfg, run_dir: Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.q_path = self.run_dir / "q_table.npz"
        self.csv_path = self.run_dir / "episodes.csv"
        self.plot_path = self.run_dir / "training_plots.png"
        self.actions = tuple(cfg.agent.actions)

        self.env: TrackmaniaEnv | None = None
        self.disc: Discretizer | None = None
        self.agent: QLearningAgent | None = None
        self.track_len = 1.0
        self.episode = 0       # licznik epizodów treningowych (ciągły między uruchomieniami)
        self.best_ms: int | None = None

    @property
    def has_saved(self) -> bool:
        return self.q_path.exists()

    def archive(self) -> Path | None:
        """Przenosi istniejący trening do kopii zapasowej (przed treningiem od zera)."""
        if not self.run_dir.exists():
            return None
        backup = self.run_dir.with_name(f"{self.run_dir.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(self.run_dir), str(backup))
        return backup

    # --- przygotowanie ------------------------------------------------------
    def _prepare(self, env: TrackmaniaEnv, load: bool):
        self.env = env
        if env.line is None:
            raise RuntimeError("Brak linii referencyjnej dla tej mapy - nagraj ją najpierw.")
        self.track_len = env.line.length

        env_cfg = env.cfg
        self.disc = Discretizer(self.cfg.disc, tuple(env_cfg.lookahead_m), env_cfg.max_deviation_m,
                                env.observation_space.shape[0])
        rng = np.random.default_rng(self.cfg.train.seed + self.episode)
        self.agent = QLearningAgent(self.disc.n_states, len(self.actions), self.cfg.agent, rng)
        print(f"[QL] Stany: {self.disc.describe()}  |  akcje: {len(self.actions)}  |  "
              f"tablica Q: {self.disc.n_states * len(self.actions):,} wartości")

        self.episode, self.best_ms = 0, None
        if load:
            meta = self.agent.load(self.q_path, self._meta())
            self.episode = meta.get("episode", 0)
            self.best_ms = meta.get("best_ms")
            self.agent.rng = np.random.default_rng(self.cfg.train.seed + self.episode)
            best = f"{self.best_ms / 1000:.2f} s" if self.best_ms else "-"
            print(f"[QL] Wczytano trening: epizod {self.episode}, najlepszy czas {best}")

    def _meta(self) -> dict:
        return {"signature": self.disc.signature(), "actions": list(self.actions),
                "episode": self.episode, "best_ms": self.best_ms}

    def _save(self):
        self.agent.save(self.q_path, self._meta())

    # --- jeden epizod -------------------------------------------------------
    def run_episode(self, epsilon: float, alpha: float, learn: bool) -> EpisodeStats:
        env, agent, disc = self.env, self.agent, self.disc
        t0 = time.perf_counter()
        obs, info = env.reset()
        agent.start_episode()
        s = disc(obs)
        a, _ = agent.act(s, epsilon)

        total, steps, speeds = 0.0, 0, []
        done = False
        while not done:
            obs, r, terminated, truncated, info = env.step(self.actions[a])
            s2 = disc(obs)
            done = terminated or truncated
            a2, greedy2 = agent.act(s2, epsilon) if not done else (0, False)
            if learn:
                agent.update(s, a, r, s2, terminal=terminated, next_is_greedy=greedy2 and not done, alpha=alpha)
            s, a = s2, a2
            total += r
            steps += 1
            speeds.append(info["speed_kmh"])

        finished = info["reason"] == "finish"
        if finished and self.cfg.train.save_ghosts:
            self._save_ghost(learn, int(info["race_time"]))
        return EpisodeStats(
            episode=self.episode, mode="train" if learn else "eval", steps=steps, reward=total,
            race_time_ms=int(info["race_time"]), reason=info["reason"] or "?", finished=int(finished),
            progress_m=info["progress_m"],
            progress_pct=100.0 if finished else min(100.0, 100.0 * info["progress_m"] / self.track_len),
            checkpoints=info["checkpoints"], avg_speed_kmh=float(np.mean(speeds)) if speeds else 0.0,
            max_speed_kmh=float(np.max(speeds)) if speeds else 0.0, epsilon=epsilon, alpha=alpha if learn else 0.0,
            visited_states_pct=100.0 * agent.visited_fraction(), wall_time_s=time.perf_counter() - t0,
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
            self._save()
            print(f"[QL] *** Nowy najlepszy czas: {st.race_time_ms / 1000:.2f} s ***")
        logger.log(st)
        print(format_line(st, self.best_ms))

    # --- tryby --------------------------------------------------------------
    def train(self, env: TrackmaniaEnv, episodes: int, resume: bool):
        """Trenuje `episodes` epizodów. resume=True kontynuuje zapisany trening."""
        tc, ac = self.cfg.train, self.cfg.agent
        self._prepare(env, load=resume and self.has_saved)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.txt").write_text(_dump_config(self.cfg, env.cfg), encoding="utf-8")
        logger = EpisodeLogger(self.csv_path)
        end = self.episode + episodes
        print(f"[QL] Trening epizodów {self.episode + 1}..{end}. Ctrl+C = zakończ po bieżącym epizodzie.")

        try:
            with GracefulStop() as stop:
                while self.episode < end and not stop.requested:
                    self.episode += 1
                    eps = linear(ac.epsilon_start, ac.epsilon_end, ac.epsilon_decay_episodes, self.episode - 1)
                    alpha = linear(ac.alpha_start, ac.alpha_end, ac.alpha_decay_episodes, self.episode - 1)
                    self._record(self.run_episode(eps, alpha, learn=True), logger)

                    if tc.eval_every and self.episode % tc.eval_every == 0 and not stop.requested:
                        self._record(self.run_episode(0.0, 0.0, learn=False), logger)
                    if tc.checkpoint_every and self.episode % tc.checkpoint_every == 0:
                        self._save()
                    if tc.plot_every and self.episode % tc.plot_every == 0:
                        plot_training(self.csv_path, self.plot_path, tc.moving_avg)
        finally:
            self._save()
            logger.close()
            print(f"[QL] Zapisano trening (epizod {self.episode}) w {self.run_dir}")
            self.plot(open_file=tc.open_plot_at_end)

    def play(self, env: TrackmaniaEnv):
        """Jazda nauczoną polityką (bez eksploracji i nauki), aż do Ctrl+C."""
        self._prepare(env, load=True)
        print("[QL] Jazda nauczonym agentem. Ctrl+C = zakończ po bieżącym przejeździe.")
        with GracefulStop("Zakończę po bieżącym przejeździe (Ctrl+C jeszcze raz = natychmiast).") as stop:
            while not stop.requested:
                st = self.run_episode(0.0, 0.0, learn=False)
                print(f"Koniec: {st.reason}, czas {st.race_time_ms / 1000:.2f} s, postęp {st.progress_pct:.1f}%")

    def plot(self, open_file: bool = False):
        out = plot_training(self.csv_path, self.plot_path, self.cfg.train.moving_avg)
        if out:
            print(f"[QL] Wykresy: {out.resolve()}")
            rows = read_log(self.csv_path)
            fin = [r for r in rows if r["finished"]]
            if fin:
                best = min(fin, key=lambda r: r["race_time_ms"])
                print(f"[QL] Najlepszy czas {best['race_time_ms'] / 1000:.2f} s (epizod {best['episode']}, "
                      f"{best['mode']}), ukończone: {len(fin)}/{len(rows)}")
            if open_file and os.name == "nt":
                os.startfile(out)  # noqa: S606 - otwiera PNG w domyślnej przeglądarce obrazów
        return out


def _dump_config(cfg, env_cfg) -> str:
    lines = ["[env]"] + [f"{k} = {v}" for k, v in asdict(env_cfg).items()] + [""]
    for section in ("disc", "agent", "train"):
        lines.append(f"[{section}]")
        lines += [f"{k} = {v}" for k, v in asdict(getattr(cfg, section)).items()]
        lines.append("")
    return "\n".join(lines)
