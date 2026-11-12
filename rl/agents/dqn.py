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
  - n-krokowe zwroty: szybsze propagowanie kary za wypadek - patrz rl/replay.py,
  - priorytetowy bufor powtórek (Schaul i in., 2016): częściej uczymy się na przejściach z dużym
    błędem TD (wypadki, trudne miejsca) - patrz rl/replay.py: PrioritizedReplayBuffer,
  - miękka aktualizacja sieci docelowej (Lillicrap i in., 2016): target <- (1-tau)*target + tau*online
    po każdym kroku gradientowym zamiast skokowej kopii co N kroków - cel zmienia się płynnie,
  - eksploracja "prawie najlepszą akcją" (pomysł z projektu Linesight dla TMNF): oprócz zwykłych
    losowych akcji część decyzji to argmax(Q + szum). Wybierana jest wtedy akcja niewiele gorsza
    od najlepszej, a nie zupełnie przypadkowa - przy 400 km/h losowy skręt zwykle kończy się wypadkiem,
  - mini-wyścigi (pomysł z projektu Linesight, horizon_steps > 0): zamiast nieskończonej sumy
    zdyskontowanych nagród Q(s, h, a) to suma nagród (bez dyskonta) w ciągu najbliższych h kroków.
    Pozostały horyzont h jest dodatkowym wejściem sieci. Przy uczeniu h jest losowane dla każdej próbki,
    a po h krokach mini-wyścig się kończy (stan "terminalny", brak bootstrapu):
        Q(s, h, a) = r_0 + ... + r_{k-1} + Q(s_k, h - k)          (k = min(n_step, h))
    Agent jedzie z h = pełny horyzont, czyli wybiera akcje, które dadzą największy postęp
    w ciągu np. 10 s - to wprost "jedź jak najdalej jak najszybciej, nie rozbijając się".
    Bez dyskonta wartości nie są zniekształcone, a ograniczony horyzont trzyma je w stałej skali,
  - chroniona pamięć najlepszej jazdy (elite_buffer_size > 0, razem z rl/sections.py): przejazdy sekcji,
    które pobiły jej najlepszy czas, trafiają do osobnego bufora, którego zwykły bufor nie nadpisuje.
    Część każdej paczki uczącej (elite_ratio) pochodzi z niego - sieć nie zapomina najlepszej znanej
    jazdy w każdej sekcji (podobnie jak w self-imitation learning, Oh i in., 2018).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..networks import QNetwork
from ..replay import PrioritizedReplayBuffer, ReplayBuffer
from ..schedules import linear, piecewise
from .base import Agent


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, obs_dim: int, n_actions: int, cfg, seed: int = 0):
        super().__init__(obs_dim, n_actions, cfg, seed)
        torch.manual_seed(seed)
        if cfg.torch_threads > 0:
            # Mała sieć nie zyskuje na wielu wątkach, a wątki PyTorcha "kręcą się" na wszystkich rdzeniach
            # i zabierają procesor grze (spadek FPS). 1 wątek = szybciej i gra płynna.
            torch.set_num_threads(cfg.torch_threads)
        self.device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if cfg.device == "auto" else cfg.device))
        self.horizon = int(cfg.horizon_steps)           # 0 = klasyczne dyskontowane Q
        in_dim = obs_dim + (1 if self.horizon else 0)    # + pozostały horyzont mini-wyścigu (0..1)
        self.online = QNetwork(in_dim, n_actions, tuple(cfg.hidden), cfg.dueling).to(self.device)
        self.target = QNetwork(in_dim, n_actions, tuple(cfg.hidden), cfg.dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        if cfg.prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, obs_dim, cfg.gamma, cfg.n_step, self.rng,
                                                  alpha=cfg.per_alpha, eps=cfg.per_eps)
        else:
            self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, cfg.gamma, cfg.n_step, self.rng)
        self.elite = (ReplayBuffer(cfg.elite_buffer_size, obs_dim, cfg.gamma, cfg.n_step, self.rng)
                      if cfg.elite_buffer_size > 0 else None)

        self.total_steps = 0     # kroki środowiska w treningu (od nich zależy epsilon)
        self.updates = 0         # kroki gradientowe
        self._ep_loss: list[float] = []
        self._ep_q: list[float] = []

    # --- wybór akcji ------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        c = self.cfg
        return piecewise(c.epsilon_schedule, self.total_steps)

    @property
    def boltzmann_epsilon(self) -> float:
        return piecewise(self.cfg.boltzmann_epsilon_schedule, self.total_steps)

    @property
    def lr(self) -> float:
        c = self.cfg
        return linear(c.lr, c.lr_end, c.lr_decay_steps, self.total_steps)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        if self.horizon:
            obs = np.append(obs, 1.0)                    # jazda: zawsze pełny horyzont
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(x).squeeze(0).cpu().numpy()

    def act(self, obs: np.ndarray, explore: bool, epsilon: float | None = None) -> int:
        q = self.q_values(obs)
        if explore:
            eps = self.epsilon if epsilon is None else epsilon
            r = self.rng.random()
            if r < eps:                                            # losowa akcja (wg wag z rl_config.py)
                return self.random_action()
            if r < eps + self.boltzmann_epsilon:                   # akcja "prawie najlepsza"
                return int(np.argmax(q + self.cfg.boltzmann_tau * self.rng.standard_normal(self.n_actions)))
        return int(np.argmax(q))

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

    def add_elite(self, transitions) -> None:
        """Kawałek przejazdu (lista (obs, a, r, next_obs, terminated, truncated)) do chronionej pamięci.
        Ostatnie przejście jest oznaczane jako przerwane (bootstrap z dalszej jazdy), chyba że to koniec epizodu."""
        if self.elite is None or not transitions:
            return
        self.elite.reset_episode()
        last = len(transitions) - 1
        for i, (o, a, r, no, term, trunc) in enumerate(transitions):
            self.elite.add(o, a, r, no, term, trunc or (i == last and not term))

    def _sample(self, buffer, n: int, prioritized: bool):
        if prioritized:
            obs, actions, returns, next_obs, discounts, idx, weights = buffer.sample(n, self.per_beta)
        else:
            obs, actions, returns, next_obs, discounts, idx = buffer.sample(n)
            weights = np.ones(n, dtype=np.float32)
        if self.horizon:
            obs, returns, next_obs, discounts = self._mini_race_targets(buffer, obs, next_obs, idx)
        return obs, actions, returns, next_obs, discounts, idx, weights

    def _mini_race_targets(self, b, obs, next_obs, idx):
        """Losuje pozostały horyzont h dla każdej próbki i składa zwrot z pojedynczych nagród.
        Zwraca wejścia sieci z dopisanym h oraz (zwrot, mnożnik bootstrapu) jak w zwykłym DQN."""
        c, H = self.cfg, self.horizon
        n = len(idx)
        # połowa próbek z pełnym horyzontem (z nim agent jedzie), reszta równomiernie 1..H
        h = np.where(self.rng.random(n) < 0.5, H, self.rng.integers(1, H + 1, size=n))
        k = b.n_steps[idx].astype(np.int64)
        m = np.minimum(k, h)                                     # ile nagród mieści się w horyzoncie
        g = c.horizon_gamma ** np.arange(b.n_step)
        returns = (b.step_rewards[idx] * g * (np.arange(b.n_step) < m[:, None])).sum(axis=1)
        # bootstrap tylko gdy zmieściło się całe przejście, epizod trwa dalej i zostało coś z horyzontu
        boot = (m == k) & ~b.terminal[idx] & (h > k)
        discounts = np.where(boot, c.horizon_gamma ** k, 0.0)
        h_next = np.where(boot, h - k, 0)
        obs = np.concatenate([obs, (h / H)[:, None]], axis=1).astype(np.float32)
        next_obs = np.concatenate([next_obs, (h_next / H)[:, None]], axis=1).astype(np.float32)
        return obs, returns.astype(np.float32), next_obs, discounts.astype(np.float32)

    @property
    def per_beta(self) -> float:
        c = self.cfg
        return linear(c.per_beta_start, 1.0, c.per_beta_steps, self.total_steps)

    def _update(self) -> None:
        c = self.cfg
        n_elite = int(c.batch_size * c.elite_ratio) if self.elite is not None and len(self.elite) >= c.batch_size else 0
        n_main = c.batch_size - n_elite
        obs, actions, returns, next_obs, discounts, idx, weights = self._sample(self.buffer, n_main,
                                                                                c.prioritized_replay)
        if n_elite:   # doklejamy próbki z chronionej pamięci najlepszej jazdy (waga 1, bez priorytetów)
            e = self._sample(self.elite, n_elite, False)
            obs, actions, returns, next_obs, discounts = (np.concatenate([x, y]) for x, y in
                                                          zip((obs, actions, returns, next_obs, discounts), e[:5]))
            weights = np.concatenate([weights, np.ones(n_elite, dtype=np.float32)])
        weights = torch.as_tensor(weights, device=self.device)
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

        # strata Hubera; przy PER każda próbka ważona korektą importance sampling
        loss = (weights * F.smooth_l1_loss(q, target, reduction="none")).mean()
        if c.prioritized_replay:
            self.buffer.update_priorities(idx, (target - q)[:n_main].detach().cpu().numpy())   # nowy błąd TD = nowy priorytet
        for group in self.optimizer.param_groups:                              # malejący współczynnik uczenia
            group["lr"] = self.lr
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.optimizer.step()

        self.updates += 1
        if c.target_tau > 0:                                                   # miękko, po każdym kroku
            with torch.no_grad():
                for t, o in zip(self.target.parameters(), self.online.parameters()):
                    t.lerp_(o, c.target_tau)
        elif self.updates % c.target_update_steps == 0:                        # skokowo co N kroków
            self.target.load_state_dict(self.online.state_dict())
        self._ep_loss.append(float(loss.item()))
        self._ep_q.append(float(q.mean().item()))

    # --- statystyki ---------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "boltzmann_epsilon": self.boltzmann_epsilon,
            "loss": float(np.mean(self._ep_loss)) if self._ep_loss else float("nan"),
            "q_mean": float(np.mean(self._ep_q)) if self._ep_q else float("nan"),
            "total_steps": self.total_steps,
        }

    def describe(self) -> str:
        c = self.cfg
        parts = [f"DQN  sieć {self.obs_dim}->{'->'.join(map(str, c.hidden))}->{self.n_actions}",
                 "double" if c.double_dqn else "", "dueling" if c.dueling else "",
                 f"PER(alpha={c.per_alpha})" if c.prioritized_replay else "",
                 f"n-step={c.n_step}",
                 f"mini-wyścigi {self.horizon} kroków" if self.horizon else f"gamma={c.gamma}", f"bufor {c.buffer_size:,}", f"urządzenie {self.device}"]
        return "  |  ".join(p for p in parts if p)

    # --- zapis / odczyt -------------------------------------------------------------
    def _signature(self) -> dict:
        c = self.cfg
        sig = {"algorithm": self.name, "obs_dim": self.obs_dim, "n_actions": self.n_actions,
               "hidden": list(c.hidden), "dueling": c.dueling}
        if self.horizon:
            sig["horizon_steps"] = self.horizon
        return sig

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
        if with_buffer and self.elite is not None and len(self.elite):
            self.elite.save(path.with_name("elite_buffer.npz"))

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
        ebuf = path.with_name("elite_buffer.npz")
        if with_buffer and self.elite is not None and ebuf.exists() and self.elite.load(ebuf):
            print(f"[DQN] Wczytano pamięć najlepszej jazdy: {len(self.elite):,} przejść")
        return json.loads(data["meta"])
