"""
Środowisko Gymnasium dla Trackmania Nations/United Forever przez TMInterface 1.x.

Jak to działa:
  - TMInterface wywołuje `on_run_step` co tick fizyki (10 ms czasu gry).
  - Dopóki callback nie zwróci, gra STOI -> agent ma tyle czasu na decyzję, ile potrzebuje,
    a symulacja jest deterministyczna i nie "ucieka" podczas treningu sieci.
  - Klient (wątek TMInterface) i env.step() (wątek główny) wymieniają się danymi przez kolejki:
        env.step(a)  --cmd_q-->  klient ustawia wejścia, gra jedzie N ticków
        env.step(a) <--obs_q--   klient oddaje (obs, reward, terminated, truncated, info)

Nagroda:
  - jeśli istnieje plik z linią referencyjną (record_line.py) -> nagroda za postęp wzdłuż trasy,
  - w przeciwnym razie -> nagroda za prędkość + bonus za checkpointy (słabsze, ale działa od razu).
"""

from __future__ import annotations

import math
import os
import queue
import threading
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tminterface.client import Client
from tminterface.interface import TMInterface


@dataclass
class EnvConfig:
    server_name: str = "TMInterface0"   # TMInterface0 dla pierwszej instancji gry, TMInterface1 dla drugiej itd.
    game_speed: float = 1.0             # >1 przyspiesza grę (np. 4.0); fizyka nadal liczona tick po ticku
    action_repeat: int = 5              # ile ticków (po 10 ms) trzyma się jedną akcję -> 50 ms na decyzję
    max_episode_ms: int = 60_000        # limit czasu epizodu
    stuck_speed_kmh: float = 5.0        # poniżej tej prędkości liczymy "utknięcie"
    stuck_grace_ms: int = 3_000         # ignoruj utknięcie przez pierwsze X ms (start)
    stuck_timeout_ms: int = 2_000       # ile ms można stać zanim epizod się skończy
    min_height: float = 9.0             # wypadnięcie z mapy (stadion: ziemia ~9, dostosuj do mapy)
    reference_line: str | None = "reference_line.npy"
    max_deviation_m: float = 30.0       # koniec epizodu, gdy auto odjedzie tak daleko od linii
    lookahead_m: tuple = (10, 20, 30, 50, 80)  # punkty linii przed autem podawane w obserwacji
    checkpoint_bonus: float = 10.0
    finish_bonus: float = 100.0
    crash_penalty: float = -10.0
    use_rewind_reset: bool = True       # reset przez rewind do stanu z t=0 (szybciej niż give_up + odliczanie)


# Dyskretne akcje: (gaz, hamulec, lewo, prawo)
DISCRETE_ACTIONS = [
    (False, False, False, False),  # 0 nic
    (True,  False, False, False),  # 1 gaz
    (True,  False, True,  False),  # 2 gaz + lewo
    (True,  False, False, True),   # 3 gaz + prawo
    (False, False, True,  False),  # 4 lewo
    (False, False, False, True),   # 5 prawo
    (False, True,  False, False),  # 6 hamulec
    (False, True,  True,  False),  # 7 hamulec + lewo
    (False, True,  False, True),   # 8 hamulec + prawo
]


# ---------------------------------------------------------------------------
# Linia referencyjna (do nagrody za postęp i obserwacji "co jest przed autem")
# ---------------------------------------------------------------------------
class ReferenceLine:
    def __init__(self, points: np.ndarray):
        self.points = points.astype(np.float64)              # (N, 3) x, y, z
        seg = np.diff(self.points, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = self.cum[-1]
        self.last_idx = 0

    def reset(self):
        self.last_idx = 0

    def project(self, pos: np.ndarray, window: int = 50) -> tuple[float, float]:
        """Zwraca (przejechany dystans wzdłuż linii, odległość od linii). Szuka tylko blisko
        poprzedniego indeksu, żeby nie przeskakiwać na inne fragmenty trasy, które się krzyżują."""
        lo = max(0, self.last_idx - 5)
        hi = min(len(self.points) - 1, self.last_idx + window)
        a = self.points[lo:hi]
        b = self.points[lo + 1:hi + 1]
        ab = b - a
        t = np.einsum("ij,ij->i", pos - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9)
        t = np.clip(t, 0.0, 1.0)
        proj = a + ab * t[:, None]
        d = np.linalg.norm(proj - pos, axis=1)
        k = int(np.argmin(d))
        self.last_idx = lo + k
        progress = self.cum[lo + k] + t[k] * self.seg_len[lo + k]
        return progress, float(d[k])

    def point_at(self, dist: float) -> np.ndarray:
        dist = min(max(dist, 0.0), self.length)
        i = int(np.searchsorted(self.cum, dist, side="right") - 1)
        i = min(i, len(self.seg_len) - 1)
        t = (dist - self.cum[i]) / max(self.seg_len[i], 1e-9)
        return self.points[i] + t * (self.points[i + 1] - self.points[i])


def _to_local(vec_xyz: np.ndarray, yaw: float) -> np.ndarray:
    """Obraca wektor świata (x, y, z) do układu auta wokół osi Y (w TM oś Y jest w górę).
    Znak osi 'bok' nie ma znaczenia dla sieci, ważne że jest spójny."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = vec_xyz
    forward = x * s + z * c
    side = x * c - z * s
    return np.array([side, y, forward], dtype=np.float32)


# ---------------------------------------------------------------------------
# Klient TMInterface
# ---------------------------------------------------------------------------
IDLE, RESETTING, RUNNING = range(3)


class RLClient(Client):
    def __init__(self, cfg: EnvConfig, line: ReferenceLine | None):
        super().__init__()
        self.cfg = cfg
        self.line = line
        self.cmd_q: queue.Queue = queue.Queue()
        self.obs_q: queue.Queue = queue.Queue()
        self.registered = threading.Event()
        self.mode = IDLE
        self.start_state = None
        self._seen_countdown = False
        self._reset_episode_vars()

    def _reset_episode_vars(self):
        self.cp_count = 0
        self.cp_target = 0
        self.new_cps = 0
        self.finished = False
        self.last_progress = 0.0
        self.stuck_ms = 0
        if self.line:
            self.line.reset()

    # --- callbacki TMInterface -------------------------------------------
    def on_registered(self, iface: TMInterface) -> None:
        print(f"[TMI] Połączono z {iface.server_name}")
        # gra czeka na klienta podczas treningu sieci -> duży timeout, inaczej TMInterface nas rozłączy
        iface.set_timeout(600_000)
        iface.set_speed(self.cfg.game_speed)
        self.registered.set()

    def on_deregistered(self, iface: TMInterface) -> None:
        print("[TMI] Rozłączono")

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int) -> None:
        if current > self.cp_count:
            self.new_cps += current - self.cp_count
        self.cp_count, self.cp_target = current, target
        if current == target:
            self.finished = True

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if self.mode == IDLE:
            self._wait_for_command(iface)  # czekamy na pierwszy env.reset()
            return

        if self.mode == RESETTING:
            if _time < 0:
                self._seen_countdown = True
                return
            if not self._seen_countdown:
                return
            # pierwszy tick po odliczaniu: zapamiętaj stan startowy do szybkich resetów
            self.start_state = iface.get_simulation_state()
            self._start_episode(iface, self.start_state)
            return

        # RUNNING
        if _time < 0 or _time % (10 * self.cfg.action_repeat) != 0:
            return
        state = iface.get_simulation_state()
        obs, reward, terminated, truncated, info = self._evaluate(state, _time)
        self.obs_q.put((obs, reward, terminated, truncated, info))
        self._wait_for_command(iface)

    # --- logika -----------------------------------------------------------
    def _start_episode(self, iface: TMInterface, state):
        self._reset_episode_vars()
        self.mode = RUNNING
        obs, _, _, _, info = self._evaluate(state, 0, first=True)
        self.obs_q.put((obs, 0.0, False, False, info))
        self._wait_for_command(iface)

    def _wait_for_command(self, iface: TMInterface):
        cmd, arg = self.cmd_q.get()  # BLOKUJE grę do czasu decyzji agenta
        if cmd == "action":
            gas, brake, left, right = arg
            iface.set_input_state(accelerate=gas, brake=brake, left=left, right=right)
        elif cmd == "analog":
            steer, gas = arg  # steer, gas w [-1, 1]
            iface.set_input_state(steer=int(steer * 65536), gas=int(gas * 65536))
        elif cmd == "reset":
            iface.set_input_state(accelerate=False, brake=False, left=False, right=False)
            if self.cfg.use_rewind_reset and self.start_state is not None:
                iface.rewind_to_state(self.start_state)
                self._start_episode(iface, self.start_state)
            else:
                self.mode = RESETTING
                self._seen_countdown = False
                iface.give_up()
        elif cmd == "close":
            pass  # tylko odblokowuje grę; samo zamknięcie robi TrackmaniaEnv.close()

    def _evaluate(self, state, t: int, first: bool = False):
        pos = np.array(state.position, dtype=np.float64)
        vel = np.array(state.velocity, dtype=np.float64)
        yaw, pitch, roll = state.yaw_pitch_roll
        speed_kmh = np.linalg.norm(vel) * 3.6

        try:
            wheels = [float(w.real_time_state.has_ground_contact) for w in state.simulation_wheels]
        except Exception:
            wheels = [0.0, 0.0, 0.0, 0.0]

        # --- obserwacja ---
        obs = [
            *(_to_local(vel, yaw) / 100.0),     # prędkość w układzie auta
            math.sin(pitch), math.sin(roll),
            *wheels,
        ]
        dist_from_line = 0.0
        progress = 0.0
        if self.line:
            progress, dist_from_line = self.line.project(pos)
            for d in self.cfg.lookahead_m:
                p = self.line.point_at(progress + d)
                obs.extend(_to_local(p - pos, yaw)[[0, 2]] / 100.0)  # (bok, przód) do punktu przed autem
            obs.append(dist_from_line / self.cfg.max_deviation_m)
        obs = np.asarray(obs, dtype=np.float32)

        # --- nagroda ---
        dt_ms = 10 * self.cfg.action_repeat
        if self.line:
            reward = (progress - self.last_progress) / 10.0   # ~ +1 za każde 10 m postępu
            self.last_progress = progress
        else:
            reward = speed_kmh / 100.0 * dt_ms / 1000.0
        reward += self.new_cps * self.cfg.checkpoint_bonus
        self.new_cps = 0

        # --- koniec epizodu ---
        terminated = truncated = False
        reason = ""
        if not first:
            if speed_kmh < self.cfg.stuck_speed_kmh and t > self.cfg.stuck_grace_ms:
                self.stuck_ms += dt_ms
            else:
                self.stuck_ms = 0

            if self.finished:
                terminated, reason = True, "finish"
                reward += self.cfg.finish_bonus
            elif pos[1] < self.cfg.min_height:
                terminated, reason = True, "fell"
                reward += self.cfg.crash_penalty
            elif self.line and dist_from_line > self.cfg.max_deviation_m:
                terminated, reason = True, "off_track"
                reward += self.cfg.crash_penalty
            elif self.stuck_ms >= self.cfg.stuck_timeout_ms:
                terminated, reason = True, "stuck"
                reward += self.cfg.crash_penalty
            elif t >= self.cfg.max_episode_ms:
                truncated, reason = True, "timeout"

        info = {"race_time": t, "speed_kmh": speed_kmh, "checkpoints": self.cp_count,
                "progress_m": progress, "reason": reason}
        return obs, float(reward), terminated, truncated, info


# ---------------------------------------------------------------------------
# Środowisko Gymnasium
# ---------------------------------------------------------------------------
class TrackmaniaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None, continuous: bool = False):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.continuous = continuous

        line = None
        if self.cfg.reference_line and os.path.exists(self.cfg.reference_line):
            line = ReferenceLine(np.load(self.cfg.reference_line))
            print(f"[ENV] Linia referencyjna: {len(line.points)} punktów, {line.length:.0f} m")
        else:
            print("[ENV] Brak linii referencyjnej -> nagroda za prędkość (nagraj ją: python record_line.py)")

        obs_dim = 3 + 2 + 4 + ((2 * len(self.cfg.lookahead_m) + 1) if line else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        if continuous:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)  # steer, gas
        else:
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        self.client = RLClient(self.cfg, line)
        self.iface = TMInterface(self.cfg.server_name)
        self.iface.register(self.client)  # TMInterface odpala własny wątek
        print(f"[ENV] Czekam na grę ({self.cfg.server_name})... uruchom TMNF z TMInterface i załaduj mapę.")
        self.client.registered.wait()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.client.cmd_q.put(("reset", None))
        obs, _, _, _, info = self.client.obs_q.get()
        return obs, info

    def step(self, action):
        if self.continuous:
            a = np.clip(action, -1.0, 1.0)
            self.client.cmd_q.put(("analog", (float(a[0]), float(a[1]))))
        else:
            self.client.cmd_q.put(("action", DISCRETE_ACTIONS[int(action)]))
        return self.client.obs_q.get()

    def close(self):
        if self.iface.registered:
            self.client.cmd_q.put(("close", None))
        self.iface.close()


if __name__ == "__main__":
    # Szybki test: losowy agent
    env = TrackmaniaEnv(EnvConfig(game_speed=2.0))
    try:
        for ep in range(3):
            obs, info = env.reset()
            total, done = 0.0, False
            while not done:
                obs, r, term, trunc, info = env.step(env.action_space.sample())
                total += r
                done = term or trunc
            print(f"Epizod {ep}: nagroda={total:.1f}, powód={info['reason']}, czas={info['race_time']} ms")
    finally:
        env.close()
