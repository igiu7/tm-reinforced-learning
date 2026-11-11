"""
Nagrywa linię referencyjną trasy: przejedź mapę sam (albo puść replay/TAS w TMInterface),
a skrypt co ~1 m zapisze pozycję auta. Po przekroczeniu mety zapisuje reference_line.npy.

    python record_line.py [plik_wyjściowy.npy]
"""

import sys
import time

import numpy as np
from tminterface.client import Client
from tminterface.interface import TMInterface


class Recorder(Client):
    def __init__(self, out_path: str):
        super().__init__()
        self.out_path = out_path
        self.points = []
        self.done = False

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Połączono z {iface.server_name}. Zrestartuj wyścig (Delete) i jedź do mety.")

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time < 0:
            if self.points:  # restart w trakcie nagrywania -> zaczynamy od nowa
                print("Restart - nagrywam od nowa")
                self.points = []
            return
        pos = np.array(iface.get_simulation_state().position)
        if not self.points or np.linalg.norm(pos - self.points[-1]) >= 1.0:
            self.points.append(pos)

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int) -> None:
        print(f"Checkpoint {current}/{target}")
        if current == target and not self.done:
            pts = np.array(self.points)
            np.save(self.out_path, pts)
            length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            print(f"Zapisano {len(pts)} punktów ({length:.0f} m) do {self.out_path}")
            self.done = True


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "reference_line.npy"
    iface = TMInterface("TMInterface0")
    rec = Recorder(out)
    iface.register(rec)
    try:
        while not rec.done:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        iface.close()


if __name__ == "__main__":
    main()
