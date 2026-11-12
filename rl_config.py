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
# Ustawienia jak w najlepszym dotąd treningu (A01, 11.11 21:56: 162 mety, najlepszy czas 25.83 s), z jedną
# zmianą: mini-wyścigi (DQNConfig.horizon_steps). Wartości z późniejszych eksperymentów są w komentarzach
# "później:" - do porównań włączaj je pojedynczo, każdy wariant z innym TrainConfig.run_name.
ENV = EnvConfig(
    port=8478,               # rl_port wtyczki RL Bridge w grze
    game_speed=10.0,         # przyspieszenie gry podczas treningu (fizyka bez zmian; o 21:56 było 20)
    action_repeat=5,         # 5 ticków x 10 ms = decyzja co 50 ms
    max_episode_ms=120_000,  # limit czasu przejazdu; musi wyraźnie przekraczać czas jazdy wolnego agenta
    max_deviation_m=15.0,    # koniec epizodu, gdy auto odjedzie tak daleko od linii w poziomie
    max_vertical_deviation_m=40.0,  # ...albo w pionie (spadnięcie z podwyższenia); skoki i zjazdy nie są karane
    max_airborne_deviation_m=15.0,  # limit w poziomie w locie; później: 50 (szybki agent wylatuje ze szczytu inaczej niż linia)
    no_progress_timeout_ms=3_000,  # koniec, gdy przez 3 s auto nie posunie się o no_progress_min_m
    no_progress_min_m=5.0,         # (ocieranie się o bandę, jazda w złą stronę) - oszczędza mnóstwo czasu
    # nagrody
    progress_reward_per_m=0.1,  # ~0.35 pkt na krok przy 250 km/h
    time_penalty_per_s=0.0,     # później: 1.0 (kara za każdą sekundę jazdy)
    # Kara za uderzenie w ścianę: spadek prędkości poziomej o więcej niż hit_speed_loss_kmh w jednym kroku (50 ms).
    # Za każdy km/h ponad próg: uderzenie tracące 60 km/h = (60 - 10) * 0.1 = -5 (połowa kary za wypadek).
    hit_speed_loss_kmh=10.0,
    hit_penalty_per_kmh=0.1,    # 0 = wyłączone
    crash_penalty=-10.0,        # później: -5
    checkpoint_bonus=10.0,      # później: 2
    finish_bonus=100.0,         # później: 20
    # punkty linii przed autem w obserwacji
    lookahead_m=(10, 20, 30, 50, 80, 120),  # później: (10, 20, 30, 45, 60, 80, 100, 125, 150, 180, 210, 250, 290, 340, 400)
    lookahead_height=False,                 # później: True (wysokość punktów - zjazdy, podjazdy)
    # dodatkowe wejścia: przyspieszenie, poślizg kół, kontakt ze ścianą, strona linii, kąt do linii, krzywizna
    # trasy przed autem (podgląd na żywo: python check_inputs.py). False = obserwacja jak wcześniej.
    extra_inputs=True,
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

# Wagi losowania akcji przy eksploracji (klucz = indeks z tm_env.DISCRETE_ACTIONS; brak = 1).
# Przy równych szansach co 4. losowa akcja puszczała gaz i auto ledwo się rozpędzało.
RANDOM_ACTION_WEIGHTS = {
    # równe szanse jak o 21:56; później: 0: 1, 1: 4, 2: 2, 3: 2, 4: 1, 5: 1, 6: 0.5, 7: 0.25, 8: 0.25
    0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0,
}


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
    # Mini-wyścigi (Linesight): Q = suma nagród bez dyskonta w najbliższych horizon_steps krokach.
    # 200 kroków x 50 ms = 10 s: obejmuje rozpęd przed wzgórzem (~5 s) i jego skutek na szczycie.
    # 0 = klasyczne Q z dyskontem gamma (do porównania w raporcie).
    horizon_steps: int = 200    # JEDYNA zmiana względem 21:56 (wtedy: 0, czyli klasyczne Q z gamma)
    horizon_gamma: float = 1.0   # dyskonto wewnątrz mini-wyścigu (1 = brak, jak w Linesight)
    # dyskonto; decyzja co 50 ms -> horyzont ~ 1/(1-gamma) kroków: 0.99 = 5 s, 0.995 = 10 s.
    # Dłuższy horyzont pozwala powiązać rozpędzenie się kilka sekund wcześniej z udanym wjazdem/skokiem.
    gamma: float = 0.99          # używane tylko przy horizon_steps = 0; później: 0.995
    n_step: int = 3              # n-krokowe zwroty (1 = klasyczny DQN); później: 5

    # priorytetowy bufor powtórek (PER)
    prioritized_replay: bool = False  # później: True; False = zwykły bufor jednorodny (do porównania w eksperymentach)
    per_alpha: float = 0.6            # siła priorytetów: 0 = losowanie jednorodne, 1 = w pełni wg błędu TD
    per_beta_start: float = 0.4       # korekta importance sampling na starcie...
    per_beta_steps: int = 300_000     # ...rośnie liniowo do 1.0 przez tyle kroków
    per_eps: float = 1e-3             # minimalny priorytet (każde przejście ma szansę zostać wylosowane)

    # optymalizacja
    lr: float = 2.5e-4           # współczynnik uczenia Adama na starcie...
    lr_end: float = 2.5e-4       # = lr, czyli stały współczynnik; później: 5e-5 (malejący)
    lr_decay_steps: int = 300_000
    batch_size: int = 128
    buffer_size: int = 200_000   # pojemność bufora powtórek (przejść)
    learning_starts: int = 5_000 # ile przejść zebrać losowo, zanim ruszy uczenie
    train_freq: int = 1          # co ile kroków środowiska aktualizacja sieci
    gradient_steps: int = 1      # ile kroków gradientowych na aktualizację
    target_tau: float = 0.0      # 0 = skokowa kopia co target_update_steps; później: 0.005 (miękko po każdym kroku)
    target_update_steps: int = 2_000  # używane tylko przy target_tau = 0
    grad_clip: float = 10.0      # obcięcie normy gradientu (stabilność)

    # chroniona pamięć najlepszej jazdy w każdej sekcji (rl/sections.py); 0 = wyłączona
    elite_buffer_size: int = 30_000
    elite_ratio: float = 0.25    # taka część każdej paczki uczącej pochodzi z tej pamięci

    # Eksploracja. Harmonogramy [(krok, wartość), ...], liniowo pomiędzy punktami (~430 kroków na epizod).
    # Wzorowane na Linesight: szybko do 0.1, potem długo i powoli do 0.03 - agent dłużej szuka lepszej jazdy.
    epsilon_schedule: tuple = ((0, 1.0), (60_000, 0.05))   # później: ((0, 1.0), (5_000, 1.0), (100_000, 0.1), (500_000, 0.03))
    # dodatkowo tyle decyzji to "prawie najlepsza akcja": argmax(Q + boltzmann_tau * szum)
    boltzmann_epsilon_schedule: tuple = ((0, 0.0),)   # wyłączone; później: ((0, 0.15), (500_000, 0.03))
    boltzmann_tau: float = 0.2   # skala szumu; typowa różnica Q między 1. a 2. akcją to ~0.25, a 1. a ostatnią ~1


# Parametry każdego algorytmu z REGISTRY. Nowy algorytm = nowy wpis.
AGENT_CONFIGS = {
    "dqn": DQNConfig(),
}


# ---------------------------------------------------------------------------
# Adaptacyjna eksploracja w strefie porażki (rl/exploration.py) - działa z każdym algorytmem epsilon-greedy
# ---------------------------------------------------------------------------
@dataclass
class ExplorationConfig:
    enabled: bool = False             # później: True; False = zwykły epsilon wg harmonogramu (do porównania w eksperymentach)
    window_episodes: int = 50         # z tylu ostatnich epizodów liczymy zasięg agenta i miejsca porażek
    patience_episodes: int = 50       # tyle epizodów bez poprawy zasięgu = zastój
    min_improvement_m: float = 25.0   # o tyle musi wzrosnąć zasięg (90. percentyl postępu), żeby liczyć to jako poprawę
    failure_bin_m: float = 25.0       # dokładność wyznaczania najczęstszego miejsca porażki
    zone_before_m: float = 300.0      # strefa zaczyna się tyle metrów przed miejscem porażki (np. rozpęd przed wzgórzem)...
    zone_after_m: float = 50.0        # ...i kończy tyle za nim
    boost_epsilon: float = 0.25       # epsilon w strefie
    min_active_episodes: int = 30     # strefa działa co najmniej tyle epizodów; z tylu ostatnich ocenia się efekt
    solved_share_ratio: float = 0.5   # wyłącz, gdy w strefie kończy się <= połowa tylu przejazdów co przy włączaniu
    max_active_episodes: int = 150    # po tylu epizodach bez efektu strefa się wyłącza i moduł znów czeka na zastój


# ---------------------------------------------------------------------------
# Sekcje toru (rl/sections.py): poprawianie jazdy po kawałku i zatrzymywanie tylko udanych poprawek
# ---------------------------------------------------------------------------
@dataclass
class SectionConfig:
    enabled: bool = True              # False = zwykła eksploracja na całej trasie (do porównania)
    count: int = 10                   # na ile równych części dzielić trasę
    window_episodes: int = 30         # z tylu ostatnich przejazdów sekcji liczymy wypadki i stratę czasu
    start_below_epsilon: float = 0.05 # ćwiczenie sekcji zaczyna się, gdy epsilon z harmonogramu spadnie do tej wartości
    focus_epsilon: float = 0.15       # losowe akcje w ćwiczonej sekcji...
    outside_epsilon: float = 0.0      # ...i poza nią (0 = najlepsza znana jazda)
    lead_in_m: float = 30.0           # eksploracja zaczyna się tyle metrów przed sekcją (inny wjazd w nią)
    failure_weight: float = 1.0       # wybór sekcji: waga udziału przejazdów, które się w niej kończą...
    time_loss_weight: float = 5.0     # ...i względnej straty do najlepszego czasu (5% straty = 0.25)
    min_potential: float = 0.05       # każda sekcja ma choć trochę szansy na ćwiczenie
    survive_after_m: float = 50.0     # poprawka sekcji liczy się, jeśli auto przejechało jeszcze tyle za nią
    seed: int = 0


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
    random_action_weights: dict = field(default_factory=lambda: dict(RANDOM_ACTION_WEIGHTS))
    env: EnvConfig = field(default_factory=lambda: ENV)
    agents: dict = field(default_factory=lambda: dict(AGENT_CONFIGS))
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    sections: SectionConfig = field(default_factory=SectionConfig)
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
