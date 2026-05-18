from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_torcs.client import TorcsClient
from gym_torcs.constants import (
    DEFAULT_SPEED,
    DEFAULT_TORCS_EXECUTABLE,
    MAX_SPEED,
    MAX_TRACK,
    MAX_RPM,
    MAX_WHEEL_SPIN_VEL,
    TERMINAL_JUDGE_START,
    TERMINATION_LIMIT_PROGRESS,
)
from .server import RaceConfig, TorcsServer, scr_idx_from_port


class TorcsEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": [None, "human"], "render_fps": 50}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        port: int = 3001,
        host: str = "127.0.0.1",
        client_id: str = "SCR",
        executable: str = DEFAULT_TORCS_EXECUTABLE,
        race_type: str = "practice",
        track_name: str = "michigan",
        track_category: str = "oval",
        laps: int = 20,
        template_xml: str | None = None,
        template_practice_xml: str | None = None,  # backwards-compatible alias
        throttle: bool = True,
        max_episode_steps: int = 100_000,
        reset_strategy: str = "relaunch",
        auto_start_server: bool = True,
        startup_sleep: float = 1.0,
        client_connect_attempts: int | None = None,
        debug: bool = False,
        gui_auto_start: bool = True,
        gui_start_delay: float = 2.0,
        gui_key_delay: float = 0.10,
        gui_auto_start_keys: tuple[str, ...] = ("Return", "Return", "Up", "Up", "Return", "Return"),
    ) -> None:
        """
        Create a Gymnasium-compatible TORCS/SCR environment.

        Parameters
        ----------
        render_mode : str | None, default=None
            Rendering mode of the environment.

            - None:
                Run TORCS in headless/text mode. This is the recommended mode
                for RL training.
            - "human":
                Start the TORCS graphical interface. This is mainly useful for
                visual debugging. If `gui_auto_start=True`, the environment
                tries to navigate the TORCS menu automatically.

        port : int, default=3001
            UDP port used by the SCR server and Python client.

            Standard SCR ports are usually mapped as:

            - 3001 -> scr_server index 0
            - 3002 -> scr_server index 1
            - 3003 -> scr_server index 2
            - ...

            For multiple environments, use different ports.

        host : str, default="127.0.0.1"
            Host address of the TORCS SCR server. Usually localhost.

        client_id : str, default="SCR"
            Client identifier sent during the SCR UDP handshake. For the
            standard SCR server this should usually remain "SCR".

        executable : str, default=DEFAULT_TORCS_EXECUTABLE
            Path to the TORCS executable or launcher. On this setup the stable
            default is usually "/usr/local/bin/torcs".

        race_type : str, default="practice"
            TORCS race manager type to generate. Common values are "practice"
            and "race". The generated XML file is written into the temporary
            TORCS config directory before launch.

        track_name : str, default="michigan"
            Name of the TORCS track to select, for example "forza",
            "michigan", or "aalborg".

        track_category : str, default="oval"
            TORCS track category, for example "road", "oval", or "dirt".
            The pair `(track_category, track_name)` determines the selected
            track.

        laps : int, default=20
            Number of laps written into the generated race configuration.

        template_xml : str | None, default=None
            Optional path to a custom TORCS race XML template. If provided,
            the environment uses this template and patches relevant fields
            such as track, race type, laps, and SCR driver index.

        template_practice_xml : str | None, default=None
            Backwards-compatible alias for `template_xml`. Prefer using
            `template_xml` in new code.

        throttle : bool, default=True
            Whether the agent controls throttle.

            - True:
                Action space is usually two-dimensional:
                `[steer, throttle_or_brake]`.
            - False:
                Action space is usually one-dimensional:
                `[steer]`, while the environment applies a simple automatic
                throttle controller.

        max_episode_steps : int, default=100_000
            Maximum number of environment steps before the episode is
            truncated.

        reset_strategy : str, default="relaunch"
            Strategy used when resetting after an episode.

            - "relaunch":
                Stop the env-owned TORCS process and launch a fresh one.
                This is slower but usually the most stable.
            - "meta":
                Send `meta=1` to ask TORCS/SCR to restart internally.
                This can be faster, but some TORCS builds segfault on restart.
            - "none":
                Do not actively restart TORCS. Mostly useful for debugging.

        auto_start_server : bool, default=True
            Whether the environment should start the TORCS process itself.

            - True:
                The env launches TORCS during `reset()`.
            - False:
                The env only connects to an already running TORCS/SCR server.
                Useful if you start TORCS manually.

        startup_sleep : float, default=1.0
            Seconds to wait after launching TORCS before attempting the SCR
            client connection.

        client_connect_attempts : int | None, default=None
            Number of attempts used for the SCR UDP connection handshake.
            If None, the environment chooses a suitable default depending on
            render mode.

        debug : bool, default=False
            If True, print diagnostic information such as the TORCS command,
            generated race file path, selected track, port, SCR index, and
            startup behavior. Useful while debugging setup issues.

        gui_auto_start : bool, default=True
            Only relevant when `render_mode="human"`.

            If True, the environment sends keyboard input to the TORCS GUI
            after startup to navigate the menu and start the selected race
            automatically. Requires a working GUI automation tool such as
            `xdotool` or `xte`.

        gui_start_delay : float, default=2.0
            Seconds to wait after launching the TORCS GUI before sending
            automatic menu-navigation keys.

        gui_key_delay : float, default=0.10
            Delay in seconds between individual GUI auto-start key presses.

        gui_auto_start_keys : tuple[str, ...], default=("Return", "Return", "Up", "Up", "Return", "Return")
            Key sequence used for automatic GUI menu navigation. The default
            sequence is based on the classic gym_torcs autostart behavior.
            Adjust this if your TORCS menu selection differs.

        Notes
        -----
        Recommended training setup:

        >>> env = gym.make(
        ...     "TorcsSCR-v0",
        ...     render_mode=None,
        ...     port=3001,
        ...     race_type="practice",
        ...     track_name="forza",
        ...     track_category="road",
        ...     reset_strategy="relaunch",
        ... )

        Recommended visual debugging setup:

        >>> env = gym.make(
        ...     "TorcsSCR-v0",
        ...     render_mode="human",
        ...     port=3001,
        ...     race_type="practice",
        ...     track_name="forza",
        ...     track_category="road",
        ...     gui_auto_start=True,
        ... )
        """ 
        super().__init__()
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be None or 'human', got {render_mode!r}")

        self.render_mode = render_mode
        self.port = int(port)
        self.scr_idx = scr_idx_from_port(self.port)
        self.host = host
        self.client_id = client_id
        self.throttle = throttle
        self.max_episode_steps = max_episode_steps
        if reset_strategy not in {"relaunch", "meta"}:
            raise ValueError("reset_strategy must be 'relaunch' or 'meta'.")
        self.reset_strategy = reset_strategy
        self.auto_start_server = auto_start_server
        self.debug = debug
        self.client_connect_attempts = client_connect_attempts
        self.time_step = 0

        action_dim = 2 if throttle else 1
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(29,), dtype=np.float32)

        if template_xml is None and template_practice_xml is not None:
            template_xml = template_practice_xml
        
        self.race_config = RaceConfig(
            race_type=race_type,
            track_name=track_name,
            track_category=track_category,
            laps=laps,
            scr_idx=self.scr_idx,
        )
        self.server = TorcsServer(
            port=self.port,
            render_mode=render_mode,
            executable=executable,
            template_xml=template_xml,
            race_config=self.race_config,
            startup_sleep=startup_sleep,
            debug=debug,
            gui_auto_start=gui_auto_start,
            gui_start_delay=gui_start_delay,
            gui_key_delay=gui_key_delay,
            gui_auto_start_keys=gui_auto_start_keys,
        )
        self.client: TorcsClient | None = None
        self._started_once = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        # First reset must start TORCS. Subsequent resets should NOT use the
        # SCR internal meta restart by default: on some TORCS/SCR builds that
        # prints "******* RESTARTING *****" and then segfaults torcs-bin.
        # The stable default is to terminate only our owned process and launch
        # a fresh TORCS process for the next episode.
        relaunch = bool(options.get("relaunch", not self._started_once))
        if self._started_once and self.reset_strategy == "relaunch":
            relaunch = True

        if "track" in options:
            track = options["track"]
            self.race_config = RaceConfig(
                race_type=str(options.get("race_type", self.race_config.race_type)),
                track_name=track["name"],
                track_category=track["category"],
                laps=int(options.get("laps", self.race_config.laps)),
                scr_idx=self.scr_idx,
            )
            relaunch = True
        elif {"race_type", "track_name", "track_category", "laps"} & set(options):
            self.race_config = RaceConfig(
                race_type=str(options.get("race_type", self.race_config.race_type)),
                track_name=str(options.get("track_name", self.race_config.track_name)),
                track_category=str(options.get("track_category", self.race_config.track_category)),
                laps=int(options.get("laps", self.race_config.laps)),
                scr_idx=self.scr_idx,
            )
            relaunch = True

        if self.client is not None:
            if self.reset_strategy == "meta" and not relaunch:
                try:
                    self.client.restart_race()
                except Exception:
                    relaunch = True
            self.client.close()
            self.client = None

        if self.debug and self._started_once:
            print(f"[gym_torcs] reset_strategy={self.reset_strategy} relaunch={relaunch}", flush=True)

        if self.auto_start_server:
            if relaunch or not self.server.running:
                self.server.restart(self.race_config)
                self._started_once = True
        elif relaunch:
            raise RuntimeError("Cannot relaunch TORCS when auto_start_server=False.")

        attempts = self.client_connect_attempts
        if attempts is None:
            attempts = 600 if self.render_mode == "human" else 60
        self.client = TorcsClient(
            host=self.host,
            port=self.port,
            client_id=self.client_id,
            connect_attempts=attempts,
        )
        self.client.connect()

        # After GUI autostart the SCR server can take a few seconds before
        # the first telemetry frame is delivered. Sending a neutral keepalive
        # action on socket timeouts prevents the server from stalling with
        # "Timeout for client answer" while reset() waits.
        startup_wait = 60.0 if self.render_mode == "human" else 15.0
        raw = self.client.receive(max_wait=startup_wait, keepalive=True)
        self.time_step = 0
        return self._obs(raw), {
            "race_type": self.race_config.race_type,
            "track_name": self.race_config.track_name,
            "track_category": self.race_config.track_category,
            "laps": self.race_config.laps,
            "port": self.port,
            "scr_idx": self.scr_idx,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("Call reset() before step().")

        action = np.asarray(action, dtype=np.float32)
        action = action.clip(-1.0, 1.0)

        prev = dict(self.client.state.data)
        self._apply_action(action)
        self.client.send()
        raw = self.client.receive()

        reward = self._reward(raw, prev, action)

        terminated = self._terminated(raw, prev)
        truncated = self._truncated(raw)
        info = self._info(raw)

        self.time_step += 1

        return self._obs(raw), reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.server.close()

    def _apply_action(self, action: np.ndarray) -> None:
        assert self.client is not None
        cmd = self.client.action
        state = self.client.state.data
        cmd.steer = float(action[0])

        if self.throttle is True:
            throttle = float(action[1])
            cmd.accel = max(throttle, 0.0)
            cmd.brake = max(-throttle, 0.0)
        
        else:
            target = DEFAULT_SPEED
            cmd.brake = 0.0
            cmd.accel += 0.01 if state["speedX"] < target - cmd.steer * 50.0 else -0.01
            cmd.accel = min(cmd.accel, 0.2)
            if state["speedX"] < 10.0:
                cmd.accel += 1.0 / (state["speedX"] + 0.1)

        cmd.gear = self._automatic_gear(int(state["gear"]), float(state["rpm"]))
        # cmd.gear = self._gear(float(state["speedX"]))

    def _obs(self, raw: dict[str, Any]) -> np.ndarray:
        speed = np.asarray([
            raw["speedX"] / MAX_SPEED, 
            raw["speedY"] / MAX_SPEED, 
            raw["speedZ"] / MAX_SPEED
        ], dtype=np.float32)
        
        rpm = np.asarray([raw["rpm"]], dtype=np.float32) / MAX_RPM
        angle = np.asarray([raw["angle"]], dtype=np.float32) / np.pi
        track_pos = np.asarray([raw["trackPos"]], dtype=np.float32)
        wheel_spin = np.asarray(raw["wheelSpinVel"], dtype=np.float32) / MAX_WHEEL_SPIN_VEL
        
        track = np.asarray(raw["track"], dtype=np.float32) / MAX_TRACK
        track = track + np.random.normal(0.0, 0.005, size=track.shape).astype(np.float32)
        track = track.clip(0.0, 1.0)

        obs = np.concatenate([speed, rpm, angle, track_pos, track, wheel_spin]).astype(np.float32)

        return obs

    def _reward(self, obs: dict[str, Any], prev: dict[str, Any], action: np.ndarray) -> float:
        speed = float(obs["speedX"]) / MAX_SPEED
        angle = float(obs["angle"])
        trackPos = float(obs["trackPos"])
        track = np.asarray(obs["track"], dtype=np.float32)
        steering = float(action[0])

        # Forward progress aligned with road direction
        reward = speed * np.cos(angle) - abs(speed * np.sin(angle)) - speed * abs(trackPos)

        # Small steering regularization
        reward -= 0.1 * steering**2

        # Collision
        if float(obs["damage"]) - float(prev["damage"]) > 0.0:
            reward -= 1.0

        # Off-track
        if track.min() < 0.0 or abs(trackPos) > 1.0:
            reward -= 1.0

        return float(reward)

    def _terminated(self, obs: dict[str, Any], prev: dict[str, Any]) -> bool:
        speed = float(obs["speedX"])
        angle = float(obs["angle"])
        trackPos = float(obs["trackPos"])
        track = np.asarray(obs["track"], dtype=np.float32)
        progress = speed * np.cos(angle)
        if track.min() < 0.0 or abs(trackPos) > 1.0:
            return True
        if self.time_step > TERMINAL_JUDGE_START and progress < TERMINATION_LIMIT_PROGRESS:
            return True
        if np.cos(float(obs["angle"])) < 0.0:
            return True
        if float(obs["damage"]) - float(prev["damage"]) > 0.0: 
            return True
        return False
    
    def _truncated(self, obs: dict[str, Any]) -> bool:
        # Truncate after one successful lap
        if float(obs["lastLapTime"]) > 0.0:
            return True
        if self.time_step >= self.max_episode_steps:
            return True 
        return False

    def _info(self, obs: dict[str, Any]) -> dict[str, Any]:
        info = {}
        info["successfulLap"] = bool(float(obs["lastLapTime"]) > 0.0)
        info["offTrack"] = bool(np.asarray(obs["track"], dtype=np.float32).min() < 0.0)
        info["distRaced"] = obs["distRaced"]
        info["timeAlive"] = obs["curLapTime"]
        info["speedX"] = obs["speedX"]
        info["speedY"] = obs["speedY"]
        return info

    @staticmethod
    def _gear(speed_x: float) -> int:
        if speed_x > 170.0:
            return 6
        if speed_x > 140.0:
            return 5
        if speed_x > 110.0:
            return 4
        if speed_x > 80.0:
            return 3
        if speed_x > 50.0:
            return 2
        return 1

    @staticmethod
    def _automatic_gear(gear: int, rpm: float) -> int:
        if gear < 1:
            return 1

        if rpm > 8000.0 and gear < 6:
            return gear + 1

        if rpm < 3500.0 and gear > 1:
            return gear - 1

        return gear