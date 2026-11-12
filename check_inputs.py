"""
Podgląd wejść sieci na żywo: jedziesz sam, a skrypt co pół sekundy wypisuje to, co widzi agent
(prędkość, kontakt i poślizg kół, kontakt ze ścianą, przyspieszenie, strona linii, kąt do linii, krzywizna).
Służy do sprawdzenia, czy dane ze stanu gry mają sens (np. czy poślizg zapala się w zakręcie,
a kontakt ze ścianą przy otarciu o bandę).

    python check_inputs.py [port]        # w grze: wyścig na mapie z nagraną linią; Ctrl+C = koniec
"""

from __future__ import annotations

import dataclasses
import math
import sys

import numpy as np

import rl_config
from maps import MapStore
from tm_env import OBS_BASE_DIM, OBS_PER_LOOKAHEAD, ReferenceLine, TrackmaniaEnv
from tmi2_bridge import TMI2Bridge


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else rl_config.ENV.port
    bridge = TMI2Bridge(port=port)
    bridge.connect()
    info = bridge.get_map_info()
    store = MapStore(rl_config.CONFIG.train.maps_dir)
    if not info.loaded or not store.has_line(info):
        print("Załaduj w grze mapę, która ma nagraną linię referencyjną.")
        return
    cfg = dataclasses.replace(rl_config.ENV, extra_inputs=True, reference_line=str(store.line_path(info)))

    # środowisko bez połączenia - używamy tylko jego funkcji liczącej obserwację z surowego stanu gry
    env = TrackmaniaEnv.__new__(TrackmaniaEnv)
    env.cfg = cfg
    env.line = ReferenceLine(np.load(cfg.reference_line))
    env._reset_episode_vars()
    per_point = OBS_PER_LOOKAHEAD if cfg.lookahead_height else OBS_PER_LOOKAHEAD - 1
    x = OBS_BASE_DIM + per_point * len(cfg.lookahead_m) + 1          # początek dodatkowych wejść

    bridge.set_speed(1.0)
    bridge.set_period(50)
    bridge.set_active(True)
    print(f"Mapa: {info.name}. Jedź - co 0,5 s wypisuję wejścia sieci. Ctrl+C = koniec.\n")
    last_print = -10_000
    try:
        while True:
            msg = bridge.recv()
            if msg[0] != "step":
                continue
            _, t, state = msg
            if t < 0:
                env._reset_episode_vars()
            else:
                obs, *_ = env._evaluate(state, t)
                if t - last_print >= 500:
                    last_print = t
                    v = np.linalg.norm(state.velocity) * 3.6
                    contact = "".join("x" if w else "." for w in obs[5:9])
                    slide = "".join("S" if w else "." for w in obs[x + 2:x + 6])
                    wall = "ŚCIANA" if obs[x + 6] else "-"
                    ang = math.degrees(math.atan2(obs[x + 8], obs[x + 9]))
                    curv = " ".join(f"{math.degrees(c):+4.0f}" for c in obs[x + 10:])
                    print(f"t {t / 1000:6.1f}s | {v:4.0f} km/h | koła {contact} poślizg {slide} | {wall:6s} | "
                          f"przysp. bok {obs[x] * 50:+5.1f} przód {obs[x + 1] * 50:+5.1f} m/s² | "
                          f"strona linii {obs[x + 7] * cfg.max_deviation_m:+5.1f} m | kąt {ang:+4.0f}° | "
                          f"zakręty [°/20 m] {curv}")
            bridge.cont()       # nie zmieniamy wejść - jedziesz ty
    except KeyboardInterrupt:
        pass
    finally:
        if bridge.connected:
            bridge.deactivate()
        bridge.close()


if __name__ == "__main__":
    main()
