"""
Środowisko Gymnasium dla Trackmania Nations/United Forever przez TMInterface 2.

Jak to działa:
  - W grze działa wtyczka tmi_plugin/RLBridge.as. Co `action_repeat` ticków fizyki (po 10 ms)
    wysyła stan auta przez socket i ZATRZYMUJE grę, dopóki env.step() nie odeśle akcji.
    Agent ma więc tyle czasu na decyzję, ile potrzebuje, a symulacja jest deterministyczna.
  - Reset: pierwszy przez restart wyścigu (odliczanie), kolejne przez szybki rewind do stanu z t=0.

Nagroda:
  - jeśli istnieje plik z linią referencyjną (record_line.py) -> nagroda za postęp wzdłuż trasy,
  - w przeciwnym razie -> nagroda za prędkość + bonus za checkpointy (słabsze, ale działa od razu).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tmi2_bridge import TMI2Bridge


@dataclass
class EnvConfig:
    host: str = "127.0.0.1"
    port: int = 8478                    # rl_port wtyczki; druga instancja gry -> np. 8479
    game_speed: float = 1.0             # >1 przyspiesza grę (np. 4.0); fizyka nadal liczona tick po ticku
    action_repeat: int = 5              # ile ticków (po 10 ms) trzyma się jedną akcję -> 50 ms na decyzję
    max_episode_ms: int = 60_000        # limit czasu epizodu
    stuck_speed_kmh: float = 5.0        # poniżej tej prędkości liczymy "utknięcie"
    stuck_grace_ms: int = 3_000         # ignoruj utknięcie przez pierwsze X ms (start)
    stuck_timeout_ms: int = 2_000       # ile ms można stać zanim epizod się skończy
    no_progress_timeout_ms: int = 3_000 # koniec, gdy przez tyle ms auto nie posunie się wzdłuż linii... (0 = wył.)
    no_progress_min_m: float = 5.0      # ...o co najmniej tyle metrów (np. ociera się o bandę albo cofa)
    min_height: float = 9.0             # wypadnięcie z mapy (stadion: ziemia ~9, dostosuj do mapy)
    reference_line: str | None = "reference_line.npy"
    max_deviation_m: float = 30.0       # koniec epizodu, gdy auto odjedzie tak daleko od linii (w poziomie)
    max_vertical_deviation_m: float = 40.0  # ...albo tyle w pionie (spadnięcie z podwyższenia); skoki są OK
    lookahead_m: tuple = (10, 20, 30, 50, 80)  # punkty linii przed autem podawane w obserwacji
    checkpoint_bonus: float = 10.0
    finish_bonus: float = 100.0
    crash_penalty: float = -10.0
    use_rewind_reset: bool = True       # reset przez rewind do stanu z t=0 (szybciej niż restart + odliczanie)


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

    def project(self, pos: np.ndarray, window: int = 50) -> tuple[float, float, float]:
        """Zwraca (przejechany dystans wzdłuż linii, odległość od linii w poziomie, w pionie).
        Szuka tylko blisko poprzedniego indeksu, żeby nie przeskakiwać na inne fragmenty trasy,
        które się krzyżują. Odległość jest rozbita na poziom/pion, bo na skokach i zjazdach auto
        leci inną wysokością niż przy nagrywaniu linii, choć nadal jest na trasie."""
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
        diff = pos - proj[k]
        return progress, float(math.hypot(diff[0], diff[2])), float(abs(diff[1]))

    def point_at(self, dist: float) -> np.ndarray:
        dist = min(max(dist, 0.0), self.length)
        i = int(np.searchsorted(self.cum, dist, side="right") - 1)
        i = min(i, len(self.seg_len) - 1)
        t = (dist - self.cum[i]) / max(self.seg_len[i], 1e-9)
        return self.points[i] + t * (self.points[i + 1] - self.points[i])


# Obserwacja (wektor float32), gdy istnieje linia referencyjna:
#   [0:3]   prędkość w układzie auta (bok, góra, przód) / 100       [m/s]
#   [3:5]   sin(pochylenie), sin(przechył)
#   [5:9]   kontakt kół z podłożem (0/1)
#   [9:12]  prędkość obrotu w układzie auta / 5                      [rad/s]
#   [12:16] ostatnio wciśnięte klawisze: gaz, hamulec, lewo, prawo (0/1)
#   potem dla każdej odległości z lookahead_m: bok / 100, przód / 100, wysokość / 50 punktu linii
#           przed autem względem auta [m]
#   [-1]    odległość od linii w poziomie / max_deviation_m
OBS_BASE_DIM = 16
OBS_PER_LOOKAHEAD = 3


def actions_to_script(action_log: list) -> str:
    """[(czas_ms, (gaz, hamulec, lewo, prawo)), ...] -> skrypt TMInterface ("100 press up" ...)."""
    keys = ("up", "down", "left", "right")
    prev = (False, False, False, False)
    lines = []
    for t, state in action_log:
        for key, now, before in zip(keys, state, prev):
            if now != before:
                lines.append(f"{t} {'press' if now else 'rel'} {key}")
        prev = state
    return "\n".join(lines)


def _to_local(vec_xyz: np.ndarray, yaw: float) -> np.ndarray:
    """Obraca wektor świata (x, y, z) do układu auta wokół osi Y (w TM oś Y jest w górę).
    Znak osi 'bok' nie ma znaczenia dla sieci, ważne że jest spójny."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = vec_xyz
    forward = x * s + z * c
    side = x * c - z * s
    return np.array([side, y, forward], dtype=np.float32)


# ---------------------------------------------------------------------------
# Środowisko Gymnasium
# ---------------------------------------------------------------------------
class TrackmaniaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None, continuous: bool = False,
                 bridge: TMI2Bridge | None = None):
        """bridge: istniejące połączenie z grą (np. z menu). Bez niego env łączy się sam
        i zamyka połączenie w close(); z nim - tylko je usypia (deactivate)."""
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.continuous = continuous

        self.line: ReferenceLine | None = None
        if self.cfg.reference_line and os.path.exists(self.cfg.reference_line):
            self.line = ReferenceLine(np.load(self.cfg.reference_line))
            print(f"[ENV] Linia referencyjna: {len(self.line.points)} punktów, {self.line.length:.0f} m")
        else:
            print("[ENV] Brak linii referencyjnej -> nagroda za prędkość (nagraj ją: python record_line.py)")

        obs_dim = OBS_BASE_DIM + ((OBS_PER_LOOKAHEAD * len(self.cfg.lookahead_m) + 1) if self.line else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        if continuous:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)  # steer, gas
        else:
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        self._owns_bridge = bridge is None
        if bridge is None:
            bridge = TMI2Bridge(self.cfg.host, self.cfg.port)
            bridge.connect()
            print(f"[TMI2] Połączono z grą ({self.cfg.host}:{self.cfg.port}).")
        self.bridge = bridge
        self.bridge.set_speed(self.cfg.game_speed)
        self.bridge.set_period(10 * self.cfg.action_repeat)
        self.bridge.set_active(True)
        self._has_start = False
        self._reset_episode_vars()

    def set_game_speed(self, speed: float):
        self.cfg.game_speed = speed
        self.bridge.set_speed(speed)

    # --- API Gymnasium -----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.cfg.use_rewind_reset and self._has_start:
            self._reset_episode_vars()
            self.bridge.rewind_to_start()
            t, state = self._wait_step()
        else:
            t, state = self._restart_race()
        obs, _, _, _, info = self._evaluate(state, t, first=True)
        return obs, info

    def step(self, action):
        if self.continuous:
            a = np.clip(action, -1.0, 1.0)
            self.bridge.analog(float(a[0]), float(a[1]))
        else:
            keys = DISCRETE_ACTIONS[int(action)]
            self.action_log.append((self._last_t, keys))
            self._last_keys = keys
            self.bridge.action(*keys)
        t, state = self._wait_step()
        return self._evaluate(state, t)

    def episode_inputs_script(self) -> str:
        """Skrypt TMInterface z wejściami bieżącego epizodu (np. do odtworzenia jako powtórka).
        Wywołuj zaraz po końcu epizodu, przed reset(). Najpierw pyta grę (dokładne wejścia),
        a gdy ta nic nie zwróci - buduje skrypt z własnego logu akcji."""
        try:
            text = self.bridge.get_inputs().strip()
        except (OSError, ConnectionError):
            text = ""
        return text or actions_to_script(self.action_log)

    def close(self):
        if not self.bridge.connected:
            return
        if self._owns_bridge:
            self.bridge.close()
        else:
            self.bridge.deactivate()

    # --- komunikacja z grą ---------------------------------------------------
    def _wait_step(self):
        """Czeka na kolejny stan z gry; po drodze obsługuje wiadomości o checkpointach."""
        while True:
            msg = self.bridge.recv()
            if msg[0] == "cp":
                self._on_checkpoint(msg[1], msg[2])
                continue
            return msg[1], msg[2]

    def _restart_race(self):
        """Restart wyścigu i czekanie na koniec odliczania; zapamiętuje stan startowy do rewindów."""
        if not self.bridge.game_waiting:
            self._wait_step()            # gra musi najpierw stanąć na jakimś kroku
        self.bridge.give_up()
        seen_countdown = False
        while True:
            t, state = self._wait_step()
            if t < 0:
                seen_countdown = True
            elif seen_countdown:
                self.bridge.save_start()
                self._has_start = True
                self._reset_episode_vars()
                return t, state
            self.bridge.cont()

    def _on_checkpoint(self, current: int, target: int):
        if current > self.cp_count:
            self.new_cps += current - self.cp_count
        self.cp_count, self.cp_target = current, target
        if target > 0 and current == target:
            self.finished = True

    # --- logika epizodu ------------------------------------------------------
    def _reset_episode_vars(self):
        self.cp_count = 0
        self.cp_target = 0
        self.new_cps = 0
        self.finished = False
        self.last_progress = 0.0
        self.best_progress = 0.0
        self.best_progress_t = 0
        self.stuck_ms = 0
        self.action_log: list = []     # (race_time, (gas, brake, left, right)) dla skryptu powtórki
        self._last_t = 0
        self._last_keys = DISCRETE_ACTIONS[0]
        if self.line:
            self.line.reset()

    def _evaluate(self, state, t: int, first: bool = False):
        self._last_t = t
        pos = np.array(state.position, dtype=np.float64)
        vel = np.array(state.velocity, dtype=np.float64)
        yaw, pitch, roll = state.yaw_pitch_roll
        speed_kmh = np.linalg.norm(vel) * 3.6

        try:
            wheels = [float(w.real_time_state.has_ground_contact) for w in state.simulation_wheels]
        except Exception:
            wheels = [0.0, 0.0, 0.0, 0.0]

        try:
            ang = np.array(state.dyna.current_state.angular_speed, dtype=np.float64)
        except Exception:
            ang = np.zeros(3)

        # --- obserwacja (układ opisany przy OBS_BASE_DIM) ---
        obs = [
            *(_to_local(vel, yaw) / 100.0),     # prędkość w układzie auta
            math.sin(pitch), math.sin(roll),
            *wheels,
            *(_to_local(ang, yaw) / 5.0),       # prędkość obrotu (m.in. czy auto zaczyna zarzucać)
            *(float(k) for k in self._last_keys),  # ostatnio wciśnięte klawisze (gaz, hamulec, lewo, prawo)
        ]
        dist_from_line = vert_from_line = 0.0
        progress = 0.0
        if self.line:
            progress, dist_from_line, vert_from_line = self.line.project(pos)
            for d in self.cfg.lookahead_m:
                side, up, fwd = _to_local(self.line.point_at(progress + d) - pos, yaw)
                # (bok, przód) - zakręty; wysokość - zjazdy, podjazdy, skocznie przed autem
                obs.extend((side / 100.0, fwd / 100.0, up / 50.0))
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

            no_progress = False
            if self.line and self.cfg.no_progress_timeout_ms > 0:
                if progress >= self.best_progress + self.cfg.no_progress_min_m:
                    self.best_progress, self.best_progress_t = progress, t
                elif t > self.cfg.stuck_grace_ms and t - self.best_progress_t > self.cfg.no_progress_timeout_ms:
                    no_progress = True

            if self.finished:
                terminated, reason = True, "finish"
                reward += self.cfg.finish_bonus
            elif pos[1] < self.cfg.min_height:
                terminated, reason = True, "fell"
                reward += self.cfg.crash_penalty
            elif self.line and (dist_from_line > self.cfg.max_deviation_m
                                or vert_from_line > self.cfg.max_vertical_deviation_m):
                terminated, reason = True, "off_track"
                reward += self.cfg.crash_penalty
            elif self.stuck_ms >= self.cfg.stuck_timeout_ms:
                terminated, reason = True, "stuck"
                reward += self.cfg.crash_penalty
            elif no_progress:
                terminated, reason = True, "no_progress"
                reward += self.cfg.crash_penalty
            elif t >= self.cfg.max_episode_ms:
                truncated, reason = True, "timeout"

        info = {"race_time": t, "speed_kmh": speed_kmh, "checkpoints": self.cp_count,
                "progress_m": progress, "reason": reason}
        return obs, float(reward), terminated, truncated, info


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
