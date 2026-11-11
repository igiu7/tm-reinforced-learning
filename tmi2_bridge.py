"""
Klient Pythona dla wtyczki tmi_plugin/RLBridge.as (TMInterface 2).

Wtyczka w trybie aktywnym (set_active(True)) po każdym kroku (co `period` ms czasu gry) wysyła
stan symulacji i ZATRZYMUJE grę, dopóki nie dostanie komendy, która ją puszcza
(action / analog / cont / give_up / disconnect). W trybie nieaktywnym gra jedzie normalnie.
Stan to bajty SimulationState.ToArray() - ten sam format, który czyta tminterface.structs.SimStateData.
"""

from __future__ import annotations

import re
import socket
import struct
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

from tminterface.structs import SimStateData


class Msg(IntEnum):
    S_STEP = 1
    S_CHECKPOINT = 2
    S_MAP_INFO = 3
    S_INPUTS = 4

    C_ACTION = 10
    C_ANALOG = 11
    C_CONTINUE = 12
    C_GIVE_UP = 13
    C_REWIND_START = 14
    C_SAVE_START = 15
    C_SET_SPEED = 16
    C_SET_PERIOD = 17
    C_EXECUTE = 18
    C_DISCONNECT = 19
    C_SET_ACTIVE = 20
    C_GET_MAP = 21
    C_GET_INPUTS = 22


class BridgeError(ConnectionError):
    pass


_TM_FORMAT = re.compile(r"\$([0-9a-fA-F]{1,3}|[wnoitsgzhlpWNOITSGZHLP<>\[\]]|\$)")


@dataclass(frozen=True)
class MapInfo:
    uid: str
    name: str          # nazwa bez kodów formatowania TrackManii ($fff, $o, ...)
    author: str

    @property
    def loaded(self) -> bool:
        return bool(self.uid)


class TMI2Bridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 8478):
        self.host, self.port = host, port
        self.sock: socket.socket | None = None
        self.game_waiting = False   # True po odebraniu S_STEP, dopóki nie puścimy gry
        self._pending: deque = deque()

    # --- połączenie -----------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self.sock is not None

    def connect(self, wait: bool = True, retry_s: float = 1.0) -> bool:
        """Łączy się z wtyczką. wait=True czeka, aż gra będzie dostępna; wait=False próbuje raz."""
        hinted = False
        while True:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=2.0)
                break
            except OSError:
                if not wait:
                    return False
                if not hinted:
                    print(f"[TMI2] Czekam na wtyczkę RL Bridge na {self.host}:{self.port}... "
                          "Uruchom TMNF z TMInterface 2 (wtyczka w Dokumenty\\TMInterface\\Plugins).")
                    hinted = True
                time.sleep(retry_s)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.game_waiting = False
        self._pending.clear()
        return True

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self._send(struct.pack("<i", Msg.C_DISCONNECT))  # puszcza grę, jeśli czeka
        except OSError:
            pass
        self.sock.close()
        self.sock = None
        self.game_waiting = False
        self._pending.clear()

    # --- niskopoziomowe -----------------------------------------------------
    def _send(self, payload: bytes) -> None:
        if self.sock is None:
            raise BridgeError("Brak połączenia z grą")
        self.sock.sendall(payload)

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                self.sock.close()
                self.sock = None
                raise BridgeError("Gra zamknęła połączenie (timeout wtyczki albo wyłączona gra)")
            buf += chunk
        return bytes(buf)

    def _recv_i32(self) -> int:
        return struct.unpack("<i", self._recv_exact(4))[0]

    def _recv_str(self) -> str:
        n = self._recv_i32()
        return self._recv_exact(n).decode("utf-8", errors="replace") if n > 0 else ""

    def _recv_raw(self):
        if self.sock is None:
            raise BridgeError("Brak połączenia z grą")
        kind = self._recv_i32()
        if kind == Msg.S_STEP:
            t = self._recv_i32()
            n = self._recv_i32()
            state = SimStateData(self._recv_exact(n))
            self.game_waiting = True
            return "step", t, state
        if kind == Msg.S_CHECKPOINT:
            cur, tgt = struct.unpack("<ii", self._recv_exact(8))
            return "cp", cur, tgt
        if kind == Msg.S_MAP_INFO:
            uid, name, author = self._recv_str(), self._recv_str(), self._recv_str()
            return "map", MapInfo(uid, _TM_FORMAT.sub("", name).strip(), _TM_FORMAT.sub("", author).strip())
        if kind == Msg.S_INPUTS:
            return "inputs", self._recv_str()
        raise BridgeError(f"Nieznana wiadomość od wtyczki: {kind}")

    def recv(self):
        """Zwraca ("step", race_time, SimStateData) albo ("cp", current, target)."""
        if self._pending:
            return self._pending.popleft()
        while True:
            msg = self._recv_raw()
            if msg[0] in ("step", "cp"):
                return msg

    def _request(self, payload: bytes, answer: str, keep_other: bool = True):
        """Wysyła zapytanie i czeka na odpowiedź danego typu; inne wiadomości odkłada na później."""
        self._send(payload)
        while True:
            msg = self._recv_raw()
            if msg[0] == answer:
                return msg[1]
            if keep_other and msg[0] in ("step", "cp"):
                self._pending.append(msg)

    # --- tryb pracy ---------------------------------------------------------
    def set_active(self, on: bool) -> None:
        self._send(struct.pack("<iB", Msg.C_SET_ACTIVE, int(on)))

    def deactivate(self) -> None:
        """Wyłącza wysyłanie kroków i puszcza grę (np. powrót do menu). Potem odrzuca zaległe
        kroki/checkpointy, które wtyczka zdążyła wysłać, żeby następny tryb zaczął od czystego stanu."""
        if self.sock is None:
            return
        self.set_active(False)          # wtyczka sama puszcza grę, jeśli czekała
        self._pending.clear()
        self.get_map_info(keep_other=False)
        self.game_waiting = False

    def get_map_info(self, keep_other: bool = True) -> MapInfo:
        return self._request(struct.pack("<i", Msg.C_GET_MAP), "map", keep_other)

    def get_inputs(self) -> str:
        """Wejścia bieżącego przejazdu jako skrypt TMInterface (do odtworzenia komendą `load`)."""
        return self._request(struct.pack("<i", Msg.C_GET_INPUTS), "inputs")

    # --- komendy puszczające grę -------------------------------------------
    def action(self, gas: bool, brake: bool, left: bool, right: bool) -> None:
        self._send(struct.pack("<iBBBB", Msg.C_ACTION, left, right, gas, brake))
        self.game_waiting = False

    def analog(self, steer: float, gas: float) -> None:
        """steer, gas w [-1, 1]."""
        self._send(struct.pack("<iii", Msg.C_ANALOG, int(steer * 65536), int(gas * 65536)))
        self.game_waiting = False

    def cont(self) -> None:
        self._send(struct.pack("<i", Msg.C_CONTINUE))
        self.game_waiting = False

    def give_up(self) -> None:
        self._send(struct.pack("<i", Msg.C_GIVE_UP))
        self.game_waiting = False

    # --- komendy, po których gra dalej czeka --------------------------------
    def rewind_to_start(self) -> None:
        """Wtyczka cofa do zapisanego startu i odsyła S_STEP ze stanem startowym."""
        self._send(struct.pack("<i", Msg.C_REWIND_START))

    def save_start(self) -> None:
        self._send(struct.pack("<i", Msg.C_SAVE_START))

    def set_speed(self, speed: float) -> None:
        self._send(struct.pack("<if", Msg.C_SET_SPEED, speed))

    def set_period(self, ms: int) -> None:
        self._send(struct.pack("<ii", Msg.C_SET_PERIOD, ms))

    def execute(self, command: str) -> None:
        data = command.encode()
        self._send(struct.pack("<ii", Msg.C_EXECUTE, len(data)) + data)
