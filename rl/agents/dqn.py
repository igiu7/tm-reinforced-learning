"""
DQN (Deep Q-Network) - Q-learning, w którym tablicę Q zastępuje sieć neuronowa.

Reguła uczenia jest ta sama co w Q-learningu:
    Q(s,a) <- r + gamma * max_a' Q(s',a')
tylko zamiast nadpisywać komórkę tablicy, minimalizujemy błąd między Q_sieci(s,a) a tym celem
(funkcja straty Hubera) metodą gradientową.

Elementy, bez których uczenie sieci jest niestabilne (Mnih i in., 2015):
  - bufor powtórek (rl/replay.py): uczymy na losowych, nieskorelowanych próbkach z historii,
  - sieć docelowa (target network): cel r + gamma*maxQ liczony kopią sieci aktualizowaną co
    `target_update_steps` kroków - inaczej sieć "goni własny ogon".
Ulepszenia (każde da się wyłączyć w rl_config.py, żeby porównać w eksperymentach):
  - Double DQN (van Hasselt i in., 2016): akcję w celu wybiera sieć ucząca się, a ocenia ją sieć
    docelowa - zmniejsza systematyczne przeszacowanie wartości Q,
  - Dueling (Wang i in., 2016): osobne gałęzie V(s) i A(s,a) - patrz rl/networks.py,
  - n-krokowe zwroty: szybsze propagowanie kary za wypadek - patrz rl/replay.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..networks import QNetwork
from ..replay import ReplayBuffer
from ..schedules import linear
from .base import Agent


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, obs_dim: int, n_actions: int, cfg, seed: int = 0):
        super().__init__(obs_dim, n_actions, cfg, seed)
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if cfg.device == "auto" else cfg.device))
        self.online = QNetwork(obs_dim, n_actions, tuple(cfg.hidden), cfg.dueling).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, tuple(cfg.hidden), cfg.dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, cfg.gamma, cfg.n_step, self.rng)

        self.total_steps = 0     # kroki środowiska w treningu (od nich zależy epsilon)
        self.updates = 0         # kroki gradientowe
        self._ep_loss: list[float] = []
        self._ep_q: list[float] = []

    # --- wybór akcji ------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        c = self.cfg
        return linear(c.epsilon_start, c.epsilon_end, c.epsilon_decay_steps, self.total_steps)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(x).squeeze(0).cpu().numpy()

    def act(self, obs: np.ndarray, explore: bool) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    # --- nauka --------------------------------------------------------------------
    def start_episode(self) -> None:
        self.buffer.reset_episode()
        self._ep_loss.clear()
        self._ep_q.clear()

    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        self.buffer.add(obs, action, reward, next_obs, terminated, truncated)
        self.total_steps += 1
        c = self.cfg
        if len(self.buffer) < max(c.learning_starts, c.batch_size) or self.total_steps % c.train_freq != 0:
            return
        for _ in range(c.gradient_steps):
            self._update()

    def _update(self) -> None:
        c = self.cfg
        obs, actions, returns, next_obs, discounts = self.buffer.sample(c.batch_size)
        obs = torch.as_tensor(obs, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        returns = torch.as_tensor(returns, device=self.device)
        discounts = torch.as_tensor(discounts, device=self.device)

        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)       # Q(s,a) dla wykonanych akcji
        with torch.no_grad():
            q_next_target = self.target(next_obs)
            if c.double_dqn:
                best = self.online(next_obs).argmax(dim=1, keepdim=True)        # wybór: sieć ucząca się
                q_next = q_next_target.gather(1, best).squeeze(1)              # ocena: sieć docelowa
            else:
                q_next = q_next_target.max(dim=1).values
            target = returns + discounts * q_next                              # discounts = 0 w stanie terminalnym

        loss = F.smooth_l1_loss(q, target)                                     # strata Hubera
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.optimizer.step()

        self.updates += 1
        if self.updates % c.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        self._ep_loss.append(float(loss.item()))
        self._ep_q.append(float(q.mean().item()))

    # --- statystyki ---------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "loss": float(np.mean(self._ep_loss)) if self._ep_loss else float("nan"),
            "q_mean": float(np.mean(self._ep_q)) if self._ep_q else float("nan"),
            "total_steps": self.total_steps,
        }

    def describe(self) -> str:
        c = self.cfg
        parts = [f"DQN  sieć {self.obs_dim}->{'->'.join(map(str, c.hidden))}->{self.n_actions}",
                 "double" if c.double_dqn else "", "dueling" if c.dueling else "",
                 f"n-step={c.n_step}", f"bufor {c.buffer_size:,}", f"urządzenie {self.device}"]
        return "  |  ".join(p for p in parts if p)

    # --- zapis / odczyt -------------------------------------------------------------
    def _signature(self) -> dict:
        c = self.cfg
        return {"algorithm": self.name, "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                "hidden": list(c.hidden), "dueling": c.dueling}

    def save(self, path: Path, meta: dict, with_buffer: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({
            "signature": self._signature(),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "updates": self.updates,
            "meta": json.dumps(meta),
        }, tmp)
        tmp.replace(path)   # atomowo: przerwanie w trakcie zapisu nie psuje poprzedniego pliku
        if with_buffer and len(self.buffer):
            self.buffer.save(path.with_name("replay_buffer.npz"))

    def load(self, path: Path, with_buffer: bool = False) -> dict:
        data = torch.load(path, map_location=self.device, weights_only=False)
        if data["signature"] != self._signature():
            raise ValueError(
                f"Zapis {path.name} pasuje do innej konfiguracji ({data['signature']}). Zmieniłeś sieć, "
                "akcje albo obserwację - ustaw inne run_name albo zacznij nowy trening."
            )
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data["target"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.total_steps, self.updates = data["total_steps"], data["updates"]
        buf = path.with_name("replay_buffer.npz")
        if with_buffer and buf.exists() and self.buffer.load(buf):
            print(f"[DQN] Wczytano bufor powtórek: {len(self.buffer):,} przejść")
        return json.loads(data["meta"])
