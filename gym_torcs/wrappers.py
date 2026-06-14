from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True, slots=True)
class TrackSpec:
    name: str
    category: str


class TimedTrackSelectionWrapper(gym.Wrapper):
    """
    Cyclically switches to the next TORCS track every `switch_every_steps`.

    Important:
    The wrapper does NOT interrupt the current episode.
    Once the step budget is reached, the next natural reset switches the track.

    Track order:

        tracks[0] -> tracks[1] -> tracks[2] -> ... -> tracks[0] -> ...

    If a TrackSpec contains `racing_line_csv`, the matching racing line is loaded
    on reset together with the selected track.
    """

    def __init__(
        self,
        env: gym.Env,
        tracks: Iterable[TrackSpec],
        switch_every_steps: int,
    ) -> None:
        super().__init__(env)

        self.tracks = tuple(tracks)
        if not self.tracks:
            raise ValueError("At least one track is required.")

        if switch_every_steps <= 0:
            raise ValueError("switch_every_steps must be positive.")

        self.switch_every_steps = int(switch_every_steps)

        self.global_step = 0
        self.steps_on_current_track = 0

        self.current_track_idx: int | None = None
        self._needs_track_switch = True

    def _current_track(self) -> TrackSpec:
        assert self.current_track_idx is not None
        return self.tracks[self.current_track_idx]

    def _track_dict(self, idx: int) -> dict[str, str]:
        track = self.tracks[idx]
        return {
            "name": track.name,
            "category": track.category,
        }

    def _next_track_idx(self) -> int:
        """
        Deterministically move to the next track.

        First reset starts at track 0.
        Afterwards:

            0 -> 1 -> 2 -> ... -> 0
        """
        if self.current_track_idx is None:
            return 0

        return (self.current_track_idx + 1) % len(self.tracks)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        options = dict(options or {})

        track_switch = False
        options["relaunch"] = True

        if self._needs_track_switch:
            self.current_track_idx = self._next_track_idx()

            self.steps_on_current_track = 0
            self._needs_track_switch = False
            track_switch = True

            track = self._track_dict(self.current_track_idx)

            options["track"] = track

        obs, info = self.env.reset(seed=seed, options=options)

        track = self._track_dict(self.current_track_idx)

        info = dict(info)
        info["global_step"] = self.global_step
        info["track_idx"] = self.current_track_idx
        info["track"] = track
        info["track_switch"] = track_switch
        info["steps_on_current_track"] = self.steps_on_current_track

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.global_step += 1
        self.steps_on_current_track += 1

        track_switch_due = self.steps_on_current_track >= self.switch_every_steps

        if track_switch_due:
            self._needs_track_switch = True

        info = dict(info)
        info["global_step"] = self.global_step
        info["steps_on_current_track"] = self.steps_on_current_track
        info["track_idx"] = self.current_track_idx
        info["track"] = self._track_dict(self.current_track_idx)
        info["track_switch_pending"] = self._needs_track_switch
        info["track_switch_due"] = track_switch_due

        return obs, reward, terminated, truncated, info


class HistoryWrapper(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """
    NOTE: This wrapper was taken from: 
    https://rl-baselines3-zoo.readthedocs.io/en/master/_modules/rl_zoo3/wrappers.html#HistoryWrapper 
    
    Stack past observations and actions to give an history to the agent.

    :param env:
    :param horizon: Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon_obs: int = 3, horizon_act: int = 2):
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Box)

        wrapped_obs_space = env.observation_space
        wrapped_action_space = env.action_space

        low_obs = np.tile(wrapped_obs_space.low, horizon_obs)
        high_obs = np.tile(wrapped_obs_space.high, horizon_obs)

        low_action = np.tile(wrapped_action_space.low, horizon_act)
        high_action = np.tile(wrapped_action_space.high, horizon_act)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space = spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)  # type: ignore[arg-type]

        super().__init__(env)

        self.horizon_obs = horizon_obs
        self.horizon_act = horizon_act
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self) -> np.ndarray:
        return np.concatenate((self.obs_history, self.action_history))
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs, info = self.env.reset(seed=seed, options=options)
        self.obs_history[..., -obs.shape[-1] :] = obs
        return self._create_obs_from_history(), info

    def step(self, action) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1] :] = action
        return self._create_obs_from_history(), reward, terminated, truncated, info


class ActionSmoothingWrapper(gym.Wrapper):
    """
    NOTE: This wrapper was taken from: 
    https://rl-baselines3-zoo.readthedocs.io/en/master/_modules/rl_zoo3/wrappers.html#HistoryWrapper 
     
    Smooth the action using exponential moving average.

    :param env:
    :param smoothing_coef: Smoothing coefficient (0 no smoothing, 1 very smooth)
    """

    def __init__(self, env: gym.Env, smoothing_coef: float = 0.0):
        super().__init__(env)
        self.smoothing_coef = smoothing_coef
        self.smoothed_action = None
        # from https://github.com/rail-berkeley/softlearning/issues/3
        # for smoothing latent space
        # self.alpha = self.smoothing_coef
        # self.beta = np.sqrt(1 - self.alpha ** 2) / (1 - self.alpha)


    
    def reset(self, seed: int | None = None, options: dict | None = None):
            self.smoothed_action = None
            # assert options is None, "Options not supported for now"
            return self.env.reset(seed=seed, options=options)


    def step(self, action):
            if self.smoothed_action is None:
                self.smoothed_action = np.zeros_like(action)
            assert self.smoothed_action is not None
            self.smoothed_action = self.smoothing_coef * self.smoothed_action + (1 - self.smoothing_coef) * action
            return self.env.step(self.smoothed_action)