from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from typing import Any

from gym_torcs.constants import TRACK_SENSOR_ANGLES

DATA_SIZE = 2**17


def _parse_value(values: list[str]) -> float | list[float] | str:
    if len(values) == 1:
        try:
            return float(values[0])
        except ValueError:
            return values[0]
    return [float(v) for v in values]


@dataclass(slots=True)
class ServerState:
    data: dict[str, Any] = field(default_factory=dict)

    def parse(self, message: str) -> dict[str, Any]:
        body = message.strip().lstrip("(").rstrip(")")
        parsed: dict[str, Any] = {}
        for item in body.split(")("):
            if not item:
                continue
            parts = item.split()
            parsed[parts[0]] = _parse_value(parts[1:])
        self.data = parsed
        return parsed


@dataclass(slots=True)
class DriverAction:
    steer: float = 0.0
    accel: float = 0.0
    brake: float = 0.0
    clutch: float = 0.0
    gear: int = 1
    meta: int = 0
    focus: tuple[float, ...] = (-90.0, -45.0, 0.0, 45.0, 90.0)

    def reset(self) -> None:
        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.clutch = 0.0
        self.gear = 1
        self.meta = 0

    def encode(self) -> bytes:
        steer = min(max(float(self.steer), -1.0), 1.0)
        accel = min(max(float(self.accel), 0.0), 1.0)
        brake = min(max(float(self.brake), 0.0), 1.0)
        clutch = min(max(float(self.clutch), 0.0), 1.0)
        gear = int(self.gear) if int(self.gear) in {-1, 0, 1, 2, 3, 4, 5, 6} else 0
        meta = int(self.meta) if int(self.meta) in {0, 1} else 0
        focus = " ".join(str(x) for x in self.focus)
        return (
            f"(accel {accel:.3f})"
            f"(brake {brake:.3f})"
            f"(clutch {clutch:.3f})"
            f"(gear {gear})"
            f"(steer {steer:.3f})"
            f"(focus {focus})"
            f"(meta {meta})"
        ).encode()


class TorcsClient:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 3001,
        client_id: str = "SCR",
        timeout: float = 2.0,
        connect_attempts: int = 60,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.connect_attempts = connect_attempts
        self.state = ServerState()
        self.action = DriverAction()
        self.socket: socket.socket | None = None

    def connect(self) -> None:
        self.close()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(self.timeout)

        angles = " ".join(str(a) for a in TRACK_SENSOR_ANGLES)
        init = f"{self.client_id}(init {angles})".encode()

        for _ in range(self.connect_attempts):
            self.socket.sendto(init, (self.host, self.port))
            try:
                payload, _ = self.socket.recvfrom(DATA_SIZE)
            except socket.timeout:
                time.sleep(0.1)
                continue
            if "***identified***" in payload.decode("utf-8", errors="replace"):
                return

        raise TimeoutError(
            f"TORCS SCR server did not answer at {self.host}:{self.port}. "
            "The race is not running, the SCR robot is not selected, or the port is wrong."
        )

    def receive(self, *, max_wait: float | None = None, keepalive: bool = False) -> dict[str, Any]:
        """Receive one SCR sensor packet.

        GUI startup can be slow and TORCS may print "Timeout for client answer"
        while the Python side is still waiting for the first telemetry packet.
        When ``keepalive`` is enabled, we send the current neutral action after
        every socket timeout. This keeps the SCR server moving until the first
        real sensor packet arrives.
        """
        if self.socket is None:
            raise RuntimeError("connect() must be called before receive().")

        deadline = None if max_wait is None else time.monotonic() + max_wait
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for TORCS telemetry at {self.host}:{self.port}. "
                    "The SCR race is running, but no sensor packet was received."
                )

            try:
                payload, _ = self.socket.recvfrom(DATA_SIZE)
            except socket.timeout:
                if keepalive:
                    self.send()
                continue

            msg = payload.decode("utf-8", errors="replace")
            if not msg or "***identified***" in msg:
                continue
            if "***shutdown***" in msg or "***restart***" in msg:
                raise ConnectionError(msg.strip())
            return self.state.parse(msg)

    def send(self) -> None:
        if self.socket is None:
            raise RuntimeError("connect() must be called before send().")
        self.socket.sendto(self.action.encode(), (self.host, self.port))

    def restart_race(self) -> None:
        self.action.meta = 1
        self.send()
        self.action.meta = 0

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
