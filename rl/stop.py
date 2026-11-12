"""Łagodne przerywanie: pierwszy Ctrl+C = zakończ po bieżącym epizodzie, drugi = natychmiast."""

from __future__ import annotations

import signal


class GracefulStop:
    def __init__(self, message: str = "Zakończę po bieżącym epizodzie (Ctrl+C jeszcze raz = natychmiast)."):
        self.message = message
        self.requested = False
        self._old = None

    def _handler(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print(f"\n[RL] Otrzymano Ctrl+C. {self.message}")

    def __enter__(self):
        self.requested = False
        self._old = signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._old)
        return False
