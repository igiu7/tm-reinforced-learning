"""
Wszystkie parametry w jednym miejscu. Zmiany działają od razu (menu wczytuje ten plik przy każdym
wyświetleniu).

    python main.py      # menu: wykrywa mapę w grze, trening / kontynuacja / jazda / wykresy / duchy

Dane każdej mapy (linia referencyjna + treningi) są w maps/<mapa>/.
"""

from dataclasses import dataclass, field

from tm_env import EnvConfig


# ---------------------------------------------------------------------------
# Środowisko (gra)
# ---------------------------------------------------------------------------
ENV = EnvConfig(
    port=8478,               # rl_port wtyczki RL Bridge w grze
    game_speed=20.0,         # przyspieszenie gry podczas treningu (fizyka bez zmian)
    action_repeat=5,         # 5 ticków x 10 ms = decyzja co 50 ms
    max_episode_ms=120_000,  # limit czasu przejazdu; musi wyraźnie przekraczać czas jazdy wolnego agenta
    max_deviation_m=15.0,    # koniec epizodu, gdy auto odjedzie tak daleko od linii w poziomie
    max_vertical_deviation_m=40.0,  # ...albo w pionie (spadnięcie z podwyższenia); skoki i zjazdy nie są karane
    no_progress_timeout_ms=3_000,  # koniec, gdy przez 3 s auto nie posunie się o no_progress_min_m
    no_progress_min_m=5.0,         # (ocieranie się o bandę, jazda w złą stronę) - oszczędza mnóstwo czasu
    crash_penalty=-10.0,
    checkpoint_bonus=10.0,
    finish_bonus=100.0,
    # punkty linii przed autem w obserwacji; najdalsze ~3 s przy 270 km/h, żeby zdążyć przed zjazdem/podjazdem
    lookahead_m=(10, 20, 30, 50, 80, 120, 170, 230),
)
PLAY_GAME_SPEED = 1.0        # prędkość gry w trybie jazdy pokazowej

# Folder wtyczek TMInterface 2. None = automatycznie: Dokumenty\TMInterface\Plugins.
# Menu przy starcie kopiuje tam tmi_plugin/RLBridge.as, jeśli brakuje jej albo jest nieaktualna.
TMI_PLUGINS_DIR: str | None = None
# Folder skryptów TMInterface (zmienna scripts_folder w grze). None = Dokumenty\TMInterface\Scripts.
# Menu -> Duchy kopiuje tam przejazdy agenta, żeby odtworzyć je komendą `load`.
TMI_SCRIPTS_DIR: str | None = None


# ---------------------------------------------------------------------------
# Algorytm i akcje
# ---------------------------------------------------------------------------
ALGORITHM = "dqn"            # klucz z rl/agents/__init__.py: REGISTRY

# Akcje (indeksy z tm_env.DISCRETE_ACTIONS):
#   0 nic, 1 gaz, 2 gaz+lewo, 3 gaz+prawo, 4 lewo, 5 prawo, 6 hamulec, 7 hamulec+lewo, 8 hamulec+prawo
# Domyślnie gaz (prosto/lewo/prawo) + puszczenie gazu. Na technicznych mapach można dodać 6 (hamulec).
ACTIONS = (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    # sieć
    hidden: tuple = (256, 256)   # warstwy ukryte MLP
    dueling: bool = True         # osobne gałęzie V(s) i A(s,a)
    double_dqn: bool = True      # Double DQN: mniejsze przeszacowanie Q
    device: str = "auto"         # "auto" (GPU, jeśli jest), "cpu", "cuda"
    torch_threads: int = 1       # wątki PyTorcha na CPU; więcej zabiera procesor grze (0 = domyślnie wszystkie)

    # cel uczenia
    gamma: float = 0.99          # dyskonto; decyzja co 50 ms -> horyzont ~ 1/(1-gamma) = 100 kroków = 5 s
    n_step: int = 3              # n-krokowe zwroty (1 = klasyczny DQN)

    # priorytetowy bufor powtórek (PER)
    prioritized_replay: bool = True   # False = zwykły bufor jednorodny (do porównania w eksperymentach)
    per_alpha: float = 0.6            # siła priorytetów: 0 = losowanie jednorodne, 1 = w pełni wg błędu TD
    per_beta_start: float = 0.4       # korekta importance sampling na starcie...
    per_beta_steps: int = 300_000     # ...rośnie liniowo do 1.0 przez tyle kroków
    per_eps: float = 1e-3             # minimalny priorytet (każde przejście ma szansę zostać wylosowane)

    # optymalizacja
    lr: float = 2.5e-4           # współczynnik uczenia Adama na starcie...
    lr_end: float = 5e-5         # ...liniowo maleje do tej wartości (stabilniejsza polityka pod koniec)
    lr_decay_steps: int = 300_000
    batch_size: int = 128
    buffer_size: int = 200_000   # pojemność bufora powtórek (przejść)
    learning_starts: int = 5_000 # ile przejść zebrać losowo, zanim ruszy uczenie
    train_freq: int = 1          # co ile kroków środowiska aktualizacja sieci
    gradient_steps: int = 1      # ile kroków gradientowych na aktualizację
    target_update_steps: int = 2_000  # co ile kroków gradientowych kopiować sieć do sieci docelowej
    grad_clip: float = 10.0      # obcięcie normy gradientu (stabilność)

    # eksploracja epsilon-greedy: liniowo od start do end przez decay_steps kroków środowiska
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02    # przy 260 km/h każda losowa akcja to ryzyko wypadku
    epsilon_decay_steps: int = 60_000   # ~300-400 epizodów; przy 150k agent jeździł losowo do ~850. epizodu


# Parametry każdego algorytmu z REGISTRY. Nowy algorytm = nowy wpis.
AGENT_CONFIGS = {
    "dqn": DQNConfig(),
}


# ---------------------------------------------------------------------------
# Trening, zapisy, wykresy
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    maps_dir: str = "maps"        # dane per mapa: maps/<mapa>/runs/<run_name>/
    run_name: str = ""            # pusty = nazwa algorytmu; zmień, by trzymać kilka wariantów obok siebie
    episodes: int = 1000          # domyślna liczba epizodów proponowana w menu
    seed: int = 0

    eval_every: int = 10          # co ile epizodów przejazd testowy (bez eksploracji i nauki); 0 = wyłączone
    checkpoint_every: int = 25    # co ile epizodów zapis agenta
    plot_every: int = 50          # co ile epizodów odświeżać plik z wykresami; 0 = tylko na końcu
    open_plot_at_end: bool = True # otwórz wykres po zakończeniu treningu
    moving_avg: int = 20          # okno średniej kroczącej na wykresach
    save_ghosts: bool = True      # zapisuj wejścia każdego ukończonego przejazdu (menu -> Duchy)
    save_replay_buffer: bool = True  # zapisuj bufor powtórek przy końcu treningu (szybsze wznowienie)
    play_best: bool = True        # jazda pokazowa agentem z najlepszym czasem (best.pt) zamiast ostatnim


@dataclass
class Config:
    algorithm: str = ALGORITHM
    actions: tuple = ACTIONS
    env: EnvConfig = field(default_factory=lambda: ENV)
    agents: dict = field(default_factory=lambda: dict(AGENT_CONFIGS))
    train: TrainConfig = field(default_factory=TrainConfig)
    play_game_speed: float = PLAY_GAME_SPEED

    @property
    def agent_config(self):
        if self.algorithm not in self.agents:
            raise ValueError(f"Brak parametrów dla algorytmu '{self.algorithm}' w AGENT_CONFIGS")
        return self.agents[self.algorithm]

    @property
    def run_name(self) -> str:
        return self.train.run_name or self.algorithm


CONFIG = Config()
