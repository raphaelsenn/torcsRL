from __future__ import annotations
from typing import Any
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_torcs.client import TorcsClient
from gym_torcs.constants import (
    DEFAULT_TORCS_EXECUTABLE,
    MAX_SPEED,
    MAX_ACCEL,
    MAX_TRACK,
    MAX_RPM,
    MAX_WHEEL_SPIN_VEL,
    CURVATURE,
    TERMINAL_JUDGE_START,
    TERMINATION_LIMIT_PROGRESS,
    RACING_LINES
)
from gym_torcs.server import RaceConfig, TorcsServer, scr_idx_from_port
from gym_torcs.curvature import RacingLineCurvatureMap

"""
NOTE: This code is mostly taken from: 
https://github.com/ugo-nama-kun/gym_torcs
https://github.com/giuse/vtorcs
https://github.com/dosssman/GymTorcs
"""


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
        torcs_max_episode_steps: int = 25_000,
        reset_strategy: str = "relaunch",
        observation_noise : bool = True,
        auto_start_server: bool = True,
        startup_sleep: float = 1.0,
        truncate_on_successful_lap: bool = True,
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
        self.torcs_max_episode_steps = torcs_max_episode_steps
        self.observation_noise = observation_noise 
        if reset_strategy not in {"relaunch", "meta"}:
            raise ValueError("reset_strategy must be 'relaunch' or 'meta'.")
        self.reset_strategy = reset_strategy
        self.auto_start_server = auto_start_server
        self.debug = debug
        self.client_connect_attempts = client_connect_attempts
        self.time_step = 0
        self.truncate_on_successful_lap = truncate_on_successful_lap

        # NOTE: Limiting the steering angle to i.e. +\- 0.4 (corresponding to +/- 18 deg) 
        # reflects the physically realistic Ackermann steering range.
        # Read more here: 
        # Deep Reinforcement Learning for Local Path Following of an Autonomous Formula SAE Vehicle, Merton et al., 2024
        # https://arxiv.org/abs/2401.02903v1
        # action_dim = 2  # [steering, throttle]
        #self.act_low = np.array([steer_min, throttle_min], dtype=np.float32)
        #self.act_high = np.array([steer_max, throttle_max], dtype=np.float32)

        action_dim = 2  # [steering, throttle]
        self.act_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.act_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(self.act_low, self.act_high, shape=(action_dim,), dtype=np.float32)

        # Old TORCS features 29 features, this version adds 16 new features (12 LAC values, racing line diff, accel xyz)
        obs_dim = 29 + 16
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

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

        self.racing_line_curvature = None
        self.racing_line_track_name = None
        self.racing_line_path = None
        self._racing_line_cache: dict[str, RacingLineCurvatureMap] = {}

        # Assetto-Corsa-Gym-style 12 values over 300 m:
        # 0, 25, 50, ..., 275
        self.lac_distances = np.arange(0, 300, 25.0, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})

        next_race_config = self.race_config

        if "track" in options:
            track = options["track"]
            next_race_config = RaceConfig(
                race_type=str(options.get("race_type", self.race_config.race_type)),
                track_name=str(track["name"]),
                track_category=str(track["category"]),
                laps=int(options.get("laps", self.race_config.laps)),
                scr_idx=self.scr_idx,
            )

        elif {"race_type", "track_name", "track_category", "laps"} & set(options):
            next_race_config = RaceConfig(
                race_type=str(options.get("race_type", self.race_config.race_type)),
                track_name=str(options.get("track_name", self.race_config.track_name)),
                track_category=str(options.get("track_category", self.race_config.track_category)),
                laps=int(options.get("laps", self.race_config.laps)),
                scr_idx=self.scr_idx,
            )

        self.race_config = next_race_config
        self.racing_line_curvature = None
        self.racing_line_track_name = None
        self.racing_line_path = None
        track_length = None

        curv_map, racing_line_path, track_length = self._get_racing_line(
            self.race_config.track_name
        )

        if curv_map is not None:
            self.racing_line_curvature = curv_map
            self.racing_line_track_name = self.race_config.track_name
            self.racing_line_path = racing_line_path

        if self.client is not None:
            self.client.close()
            self.client = None

        if not self.auto_start_server:
            raise RuntimeError(
                "auto_start_server=False, but full restart reset needs auto_start_server=True."
            )

        self.server.restart(self.race_config)
        self._started_once = True

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

        startup_wait = 60.0 if self.render_mode == "human" else 15.0
        raw = self.client.receive(max_wait=startup_wait, keepalive=True)

        self.time_step = 0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        tita_steer = float(raw["titaSteer"])
        tita_gas = float(raw["titaGas"])
        tita_brake = float(raw["titaBrake"])
        tita_throttle = tita_gas - tita_brake
        action_tita = np.array([tita_steer, tita_throttle], dtype=np.float32)

        return self._obs(raw), {
            "race_type": self.race_config.race_type,
            "track_name": self.race_config.track_name,
            "track_category": self.race_config.track_category,
            "track_length": track_length,
            "racing_line_track_name": self.racing_line_track_name,
            "laps": self.race_config.laps,
            "port": self.port,
            "scr_idx": self.scr_idx,
            "full_restart": True,
            "action_tita": action_tita,
            "distFromStart": raw.get("distFromStart", 0.0),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("Call reset() before step().")

        action = np.asarray(action, dtype=np.float32)
        # action = np.clip(action, self.act_low, self.act_high)
        action = np.clip(action, -1.0, 1.0)

        prev = dict(self.client.state.data)
        self._apply_action(action)
        self.client.send()
        # raw = self.client.receive(max_wait=10.0)
        try:
            raw = self.client.receive(max_wait=10.0)
        except TimeoutError as e:
            print(
                "[TORCS TIMEOUT] no telemetry received. "
                f"track={self.race_config.track_category}/{self.race_config.track_name}, "
                f"port={self.port}, "
                f"step={self.time_step}"
            )

            if self.client is not None:
                self.client.close()
                self.client = None

            obs = getattr(
                self,
                "_last_obs",
                np.zeros(self.observation_space.shape, dtype=np.float32),
            )

            info = {
                "torcs_timeout": True,
                "track_name": self.race_config.track_name,
                "track_category": self.race_config.track_category,
                "port": self.port,
                "scr_idx": self.scr_idx,
                "done_td": True,
                "offTrack": False,
                "crashed": True,
                "damage": np.nan,
                "damage_delta": np.nan,
                "distRaced": np.nan,
                "timeAlive": np.nan,
                "speedX": np.nan,
                "lastLapTime": 0.0,
                "action_tita": np.zeros(3, dtype=np.float32),
                "globalX": np.nan,
                "globalY": np.nan,
                "distFromStart": np.nan,
            }

            return obs, 0.0, False, True, info

        reward = self._reward(raw, prev, action)
        terminated = self._terminated(raw, prev)
        truncated = self._truncated(raw)
        info = self._info(raw, prev)

        self.time_step += 1
        self.prev_action = action.copy()

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

        steer = float(action[0])
        throttle = float(action[1])

        cmd.steer = steer
        cmd.accel = max(throttle, 0.0)
        cmd.brake = max(-throttle, 0.0)
        cmd.gear = self._gear(int(state["gear"]), float(state["rpm"]))

    def _obs(self, raw: dict[str, Any]) -> np.ndarray:
        speed = np.asarray([
            raw["speedX"] / MAX_SPEED,
            raw["speedY"] / MAX_SPEED,
            raw["speedZ"] / MAX_SPEED
        ], dtype=np.float32)                                                                            # [-1, 1]

        accel = np.asarray([
            raw["accelX"] / MAX_ACCEL,
            raw["accelY"] / MAX_ACCEL,
            raw["accelZ"] / MAX_ACCEL
        ], dtype=np.float32)                                                                            # [-1, 1]

        rpm = (np.asarray([raw["rpm"]], dtype=np.float32) / MAX_RPM) * 2 - 1                            # [-1, 1]
        angle = np.asarray([raw["angle"]], dtype=np.float32) / np.pi                                    # [-1, 1]
        track_pos = np.asarray([raw["trackPos"]], dtype=np.float32)                                     # [-1, 1]
        wheel_spin = (np.asarray(raw["wheelSpinVel"], dtype=np.float32) / MAX_WHEEL_SPIN_VEL) * 2 - 1   # [-1, 1]
        track = (np.asarray(raw["track"], dtype=np.float32) / MAX_TRACK) * 2 - 1                        # [-1, 1]

        target_track_pos = -1.0 * float(raw["titaLineTrackPos"])
        racing_line_diff = np.asarray([(float(raw["trackPos"]) - target_track_pos) / 2], dtype=np.float32)
        racing_line_diff = np.clip(racing_line_diff, -1.0, 1.0)                                         # [-1, 1]

        # Look ahead curvature, read more here:
        # (1): AssettoCorsaGym, https://assetto-corsa-gym.github.io/
        # (2): A Simulation Benchmark for Autonomous Racing with Large-Scale Human Data, https://arxiv.org/abs/2407.16680
        # (3): Formula RL: Deep Reinforcement Learning for Autonomous Racing using Telemetry Data, https://arxiv.org/abs/2104.11106
        if self.racing_line_curvature is not None and "distFromStart" in raw: 
            lac = self.racing_line_curvature.lookahead(
                dist_from_start=float(raw["distFromStart"]),
                distances=self.lac_distances,
            )
            lac = lac / CURVATURE
            # lac = np.clip(lac, -1.0, 1.0)

        else: 
            print("Wrong LAC...")
            lac = np.asarray([
                raw["curv0"],
                raw["curv25"],
                raw["curv50"],
                raw["curv75"],
                raw["curv100"],
                raw["curv125"],
                raw["curv150"],
                raw["curv175"],
                raw["curv200"],
                raw["curv225"],
                raw["curv250"],
                raw["curv275"],
            ], dtype=np.float32) / CURVATURE
            # lac = np.clip(lac, -1.0, 1.0)

        # dims = [3 + 3 + 1 + 1 + 1 + 19 + 4 + 1 + 12] = [45]
        obs = np.concatenate([speed, accel, rpm, angle, track_pos, track, wheel_spin, racing_line_diff, lac]).astype(np.float32)
        
        if self.observation_noise is True: 
            obs_noise = self.np_random.normal(0.0, 0.025, size=obs.shape).astype(np.float32)
            obs = (obs + obs_noise).astype(np.float32)
        
        obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
        return obs

    def _reward(self, obs: dict[str, Any], prev: dict[str, Any], action: np.ndarray) -> float:
        speed_kmh = float(obs["speedX"]) 
        speed = speed_kmh / MAX_SPEED
        angle = float(obs["angle"])
        track = np.asarray(obs["track"], dtype=np.float32)
        track_pos = float(obs["trackPos"])

        # NOTE: Target track pos from tita bot is flipped (i.e. left = -1.0 and right = 1.0),
        # but track_pos is encoded as -1.0 = right and 1.0 = left...
        target_track_pos = -1.0 * float(obs["titaLineTrackPos"])
        racing_line_diff = abs(track_pos - target_track_pos) / 2.0  # [0, 1]
        # print(f"TrackPos: {track_pos:.2f}\tTargetTrackPos: {target_track_pos:.2f}")

        # Main reward function
        reward = speed * (1.0 - racing_line_diff)
        # reward = speed * (np.cos(angle) - np.sin(angle) - racing_line_diff)

        # Collision
        damage_delta = float(obs["damage"]) - float(prev["damage"])
        if damage_delta > 0.0:
            return -10.0

        if abs(track_pos) > 1.0 or track.min() < 0.0:
            return -10.0
        
        # Moving backward
        if np.cos(angle) < 0.0:
            return -10.0
        
        # Terminal judge start
        if self.time_step > TERMINAL_JUDGE_START and speed_kmh < TERMINATION_LIMIT_PROGRESS:
            return -10.0

        return float(reward)

    def _terminated(self, obs: dict[str, Any], prev: dict[str, Any]) -> bool:
        speed_kmh = float(obs["speedX"])
        angle = float(obs["angle"])
        track_pos = float(obs["trackPos"])
        track = np.asarray(obs["track"], dtype=np.float32)

        if abs(track_pos) > 1.0 or track.min() < 0.0:
            return True

        # Terminal judge start 
        if self.time_step > TERMINAL_JUDGE_START and speed_kmh < TERMINATION_LIMIT_PROGRESS:
            return True
        
        # Moving backwards 
        if np.cos(angle) < 0.0:
            return True
        
        # Damage 
        damage_delta = float(obs["damage"]) - float(prev["damage"])
        if damage_delta > 0.0:
            return True

        return False
    
    def _truncated(self, obs: dict[str, Any]) -> bool:
        if float(obs["lastLapTime"]) > 0.0 and self.truncate_on_successful_lap is True:
            return True
        if self.time_step + 1 >= self.torcs_max_episode_steps:
            return True
        return False

    def _done_td(self, obs: dict[str, Any], prev: dict[str, Any]) -> bool:
        track = np.asarray(obs["track"], dtype=np.float32)
        damage_delta = float(obs["damage"]) - float(prev["damage"])

        crashed = bool(damage_delta > 0.0)
        off_track = bool(track.min() < 0.0) or bool(abs(float(obs["trackPos"])) > 1.0)

        return crashed or off_track

    def _info(self, obs: dict[str, Any], prev: dict[str, Any]) -> dict[str, Any]:
        track = np.asarray(obs["track"], dtype=np.float32)
        damage_delta = float(obs["damage"]) - float(prev["damage"])

        off_track = track.min() < 0.0 or abs(float(obs["trackPos"])) > 1.0
        crashed = damage_delta > 0.0

        tita_steer = float(obs["titaSteer"])
        tita_gas = float(obs["titaGas"])
        tita_brake = float(obs["titaBrake"])
        tita_throttle = tita_gas - tita_brake
        action_tita = np.array([tita_steer, tita_throttle], dtype=np.float32)

        return {
            "successfulLap": bool(float(obs["lastLapTime"]) > 0.0),
            "offTrack": off_track,
            "crashed": crashed,
            "damage": float(obs["damage"]),
            "damage_delta": damage_delta,
            "distRaced": obs["distRaced"],
            "timeAlive": obs["curLapTime"],
            "speedX": obs["speedX"],
            "done_td": crashed or off_track,
            "lastLapTime" : float(obs["lastLapTime"]),
            "action_tita" : action_tita,     # The action tita bot might take (nearly expert an action :D)
            "globalX" : obs["x"],
            "globalY" : obs["y"],
            "distFromStart" : float(obs["distFromStart"])
        }

    @staticmethod
    def _gear(gear: int, rpm: float) -> int:
        if gear < 1:
            return 1

        if rpm > 8000.0 and gear < 6:
            return gear + 1

        if rpm < 3500.0 and gear > 1:
            return gear - 1

        return gear

    def _get_racing_line(self, track_name: str):
        racing_line_path = RACING_LINES.get(track_name)
        if racing_line_path is None:
            return None, None, None

        racing_line_path = str(racing_line_path)

        if not os.path.exists(racing_line_path) or not racing_line_path.endswith(".csv"):
            return None, None, None

        if track_name not in self._racing_line_cache:
            self._racing_line_cache[track_name] = RacingLineCurvatureMap.from_csv(racing_line_path)

        curv_map = self._racing_line_cache[track_name]
        track_length = curv_map.track_length

        return curv_map, racing_line_path, track_length
