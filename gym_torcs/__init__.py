from __future__ import annotations

from gymnasium.envs.registration import register

from .env import TorcsEnv
from .wrappers import TrackSelectionWrapper, TrackSpec

register(
    id="TorcsSCR-v0",
    entry_point=f"{__name__}.env:TorcsEnv",
)

__all__ = ["TorcsEnv", "TrackSelectionWrapper", "TrackSpec"]
