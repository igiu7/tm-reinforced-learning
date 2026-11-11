"""
Wszystkie parametry treningu Q-learning w jednym miejscu. Zmieniaj tylko ten plik.

    python train_qlearning.py          # menu: wykrywa mapę w grze, trening / kontynuacja / jazda / wykresy

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
    max_deviation_m=15.0,    # koniec epizodu, gdy auto odjedzie tak daleko od linii (szybka kara za spadnięcie z drogi)
    no_progress_timeout_ms=3_000,  # koniec, gdy przez 3 s auto nie posunie się o no_progress_min_m
    no_progress_min_m=5.0,         # (ocieranie się o bandę, jazda w złą stronę) - oszczędza mnóstwo czasu
    crash_penalty=-10.0,
    checkpoint_bonus=10.0,
    finish_bonus=100.0,
)
PLAY_GAME_SPEED = 1.0        # prędkość gry w trybie "play"

# Folder wtyczek TMInterface 2. None = automatycznie: Dokumenty\TMInterface\Plugins.
# Menu przy starcie kopiuje tam tmi_plugin/RLBridge.as, jeśli brakuje jej albo jest nieaktualna.
TMI_PLUGINS_DIR: str | None = None
# Folder skryptów TMInterface (zmienna scripts_folder w grze). None = Dokumenty\TMInterface\Scripts.
# Menu -> Duchy kopiuje tam przejazdy agenta, żeby odtworzyć je komendą `load`.
TMI_SCRIPTS_DIR: str | None = None


# ---------------------------------------------------------------------------
# Dyskretyzacja stanu (ciągła obserwacja -> numer komórki tablicy Q)
# ---------------------------------------------------------------------------
# Każda cecha jest dzielona na przedziały wg podanych granic: N granic = N+1 przedziałów.
# Liczba stanów = iloczyn liczby przedziałów wszystkich cech. Więcej przedziałów = dokładniejsza
# jazda, ale WOLNIEJSZA nauka (każdą komórkę trzeba wiele razy odwiedzić).
@dataclass
class DiscretizerConfig:
    # Kąt (w stopniach) między kierunkiem auta a punktem linii referencyjnej X metrów przed autem.
    # Odległość X musi być jedną z wartości ENV.lookahead_m (domyślnie 10, 20, 30, 50, 80).
    #   - punkt bliski  -> gdzie jestem względem linii (korekta toru)
    #   - punkt daleki  -> jaki zakręt nadchodzi
    angle_features: tuple = (
        (10, (-30.0, -12.0, -4.0, 4.0, 12.0, 30.0)),   # 7 przedziałów
        (30, (-35.0, -12.0, 12.0, 35.0)),              # 5 przedziałów
        (80, (-40.0, -12.0, 12.0, 40.0)),              # 5 przedziałów
    )
    # prędkość do przodu: 6 przedziałów. Najniższy (<15 km/h) odróżnia start / stanie od jazdy.
    speed_edges_kmh: tuple = (15.0, 60.0, 110.0, 170.0, 240.0)
    deviation_edges_m: tuple = (2.5, 6.0, 10.0)            # odległość od linii: 4 przedziały (czuje zbliżanie się do krawędzi)
    slip_edges_kmh: tuple = ()          # poślizg boczny, np. (-20, 20); puste = cecha wyłączona
    use_airborne: bool = False          # dodatkowa cecha "auto w powietrzu" (x2 liczba stanów)


# ---------------------------------------------------------------------------
# Agent Q(lambda) (Watkins)
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    gamma: float = 0.99          # dyskonto; przy decyzji co 50 ms horyzont ~ 1/(1-gamma) = 100 kroków = 5 s
    lam: float = 0.9            # ślady eligibility; 0 = zwykły Q-learning, ~0.8 = dużo szybsza nauka
    trace_min: float = 0.01      # ślady mniejsze od tego są usuwane (wydajność)
    q_init: float = 0.0          # początkowa wartość Q; >0 = "optymizm", wymusza próbowanie nowych akcji

    # Współczynnik uczenia: liniowo od start do end przez decay_episodes epizodów
    alpha_start: float = 0.15
    alpha_end: float = 0.05
    alpha_decay_episodes: int = 600

    # Eksploracja epsilon-greedy: liniowo od start do end przez decay_episodes epizodów
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1500   # długa trasa = dłuższa eksploracja (przy 400 agent "zastygał" za wcześnie)

    # Używane akcje (indeksy z tm_env.DISCRETE_ACTIONS). Mniej akcji = szybsza nauka.
    #   0 nic, 1 gaz, 2 gaz+lewo, 3 gaz+prawo, 4 lewo, 5 prawo, 6 hamulec, 7 hamulec+lewo, 8 hamulec+prawo
    # Domyślnie tylko gaz (prosto/lewo/prawo) + hamulec: losowa eksploracja naprawdę jedzie do przodu.
    # Gdy agent już dojeżdża do mety, można spróbować dodać 0 (puszczenie gazu) w nowym treningu.
    actions: tuple = (1, 2, 3, 6)


# ---------------------------------------------------------------------------
# Trening, zapisy, wykresy
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    maps_dir: str = "maps"        # dane per mapa: maps/<mapa>/runs/<run_name>/
    run_name: str = "qlearning_v3"  # zmień, gdy zmieniasz dyskretyzację/akcje (stara tablica Q nie pasuje)
    episodes: int = 1000          # domyślna liczba epizodów proponowana w menu
    seed: int = 0

    eval_every: int = 10          # co ile epizodów przejazd testowy (bez eksploracji i nauki); 0 = wyłączone
    checkpoint_every: int = 25    # co ile epizodów zapis tablicy Q
    plot_every: int = 50          # co ile epizodów odświeżać plik z wykresami; 0 = tylko na końcu
    open_plot_at_end: bool = True # otwórz wykres po zakończeniu treningu
    save_ghosts: bool = True      # zapisuj wejścia każdego ukończonego przejazdu (menu -> Duchy)
    moving_avg: int = 20          # okno średniej kroczącej na wykresach


@dataclass
class QLConfig:
    env: EnvConfig = field(default_factory=lambda: ENV)
    disc: DiscretizerConfig = field(default_factory=DiscretizerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    play_game_speed: float = PLAY_GAME_SPEED


CONFIG = QLConfig()
