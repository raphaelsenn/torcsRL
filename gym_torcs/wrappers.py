from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym


@dataclass(frozen=True, slots=True)
class TrackSpec:
    name: str
    category: str


class TrackSelectionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, tracks: Iterable[TrackSpec], mode: str = "cycle") -> None:
        super().__init__(env)
        self.tracks = tuple(tracks)
        if not self.tracks:
            raise ValueError("At least one track is required.")
        if mode not in {"fixed", "cycle", "random"}:
            raise ValueError("mode must be 'fixed', 'cycle', or 'random'.")
        self.mode = mode
        self._cycle = itertools.cycle(self.tracks)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = dict(options or {})
        track = options.pop("track", None)

        if track is None:
            if self.mode == "fixed":
                track = self.tracks[0]
            elif self.mode == "random":
                track = random.choice(self.tracks)
            else:
                track = next(self._cycle)

        if isinstance(track, TrackSpec):
            track = {"name": track.name, "category": track.category}

        options["track"] = track
        options["relaunch"] = True
        return self.env.reset(seed=seed, options=options)
