"""
Nagrywa linię referencyjną trasy (TMInterface 2 + wtyczka RL Bridge): przejedź mapę sam,
a skrypt co ~1 m zapisze pozycję auta. Po przekroczeniu mety zapisuje linię do katalogu mapy
(maps/<mapa>/reference_line.npy). Zwykle uruchamiane z menu: python train_qlearning.py

    python record_line.py [port]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tmi2_bridge import TMI2Bridge


def record_line(bridge: TMI2Bridge, out_path: Path) -> bool:
    """Nagrywa jeden przejazd do mety. Zwraca True, jeśli zapisano linię. Ctrl+C przerywa."""
    bridge.set_speed(1.0)
    bridge.set_period(10)      # stan co tick; gra czeka na odpowiedź, ale lokalnie to ułamek ms
    bridge.set_active(True)
    print("Restartuję wyścig - po odliczaniu jedź do mety. Restart w trakcie (Delete) = nagrywanie od nowa. "
          "Ctrl+C = anuluj.")

    points: list[np.ndarray] = []
    restarted = False
    try:
        while True:
            msg = bridge.recv()
            if msg[0] == "cp":
                _, current, target = msg
                if current > 0:
                    print(f"Checkpoint {current}/{target}")
                if target > 0 and current == target and len(points) > 1:
                    pts = np.array(points)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_path, pts)
                    length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
                    print(f"Zapisano {len(pts)} punktów ({length:.0f} m) do {out_path}")
                    return True
                continue

            _, t, state = msg
            if not restarted:
                bridge.give_up()    # zaczynamy zawsze od startu, niezależnie od tego, gdzie stoi auto
                restarted = True
                continue
            if t < 0:
                if points:
                    print("Restart - nagrywam od nowa")
                    points = []
            else:
                pos = np.array(state.position, dtype=np.float64)
                if not points or np.linalg.norm(pos - points[-1]) >= 1.0:
                    points.append(pos)
            bridge.cont()       # nie zmieniamy wejść - jedziesz ty
    except KeyboardInterrupt:
        print("Anulowano - linia nie została zapisana.")
        raise
    finally:
        if bridge.connected:
            bridge.deactivate()


def main():
    from maps import MapStore

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8478
    bridge = TMI2Bridge(port=port)
    bridge.connect()
    try:
        info = bridge.get_map_info()
        if not info.loaded:
            print("W grze nie jest załadowana żadna mapa.")
            return
        print(f"Mapa: {info.name} ({info.uid})")
        record_line(bridge, MapStore().line_path(info))
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
