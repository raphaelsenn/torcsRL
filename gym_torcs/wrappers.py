from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import gymnasium as gym


@dataclass(frozen=True, slots=True)
class TrackSpec:
    name: str
    category: str
    racing_line_csv: str | None = None


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

            track_spec = self._current_track()
            track = self._track_dict(self.current_track_idx)

            options["track"] = track
            options["racing_line_csv"] = track_spec.racing_line_csv

        obs, info = self.env.reset(seed=seed, options=options)

        track_spec = self._current_track()
        track = self._track_dict(self.current_track_idx)

        info = dict(info)
        info["global_step"] = self.global_step
        info["track_idx"] = self.current_track_idx
        info["track"] = track
        info["track_switch"] = track_switch
        info["steps_on_current_track"] = self.steps_on_current_track
        info["racing_line_csv"] = track_spec.racing_line_csv

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
        info["racing_line_csv"] = self._current_track().racing_line_csv
        info["track_switch_pending"] = self._needs_track_switch
        info["track_switch_due"] = track_switch_due

        return obs, reward, terminated, truncated, info