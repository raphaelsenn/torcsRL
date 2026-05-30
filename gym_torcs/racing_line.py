from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RacingLine:
    dist: np.ndarray
    target_track_pos: np.ndarray
    track_length: float

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        track_length: float | None = None,
        bin_size: float = 0.5,
        min_speed: float = 1.0,
    ) -> "RacingLine":
        df = pd.read_csv(path)

        required = {"distFromStart", "targetTrackPos", "speedX"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in racing line CSV: {missing}")

        # If csv contains repeated header lines.
        for col in ["distFromStart", "targetTrackPos", "speedX"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[["distFromStart", "targetTrackPos", "speedX"]].dropna()

        # Remove stationary/warmup rows before the race actually starts.
        df = df[df["speedX"].abs() >= min_speed]

        if len(df) < 2:
            raise ValueError("Racing line CSV is too small after filtering.")

        if track_length is None:
            # Approximation. Better: pass the real track length explicitly.
            track_length = float(df["distFromStart"].max())

        if track_length <= 1.0:
            raise ValueError(f"Invalid track_length: {track_length}")

        # Normalize distance into [0, track_length).
        df["dist"] = df["distFromStart"] % track_length

        # Bin distances so multiple laps collapse into one mean racing line.
        df["bin"] = (df["dist"] / bin_size).round() * bin_size

        df = (
            df.groupby("bin", as_index=False)["targetTrackPos"]
            .mean()
            .rename(columns={"bin": "dist"})
            .sort_values("dist")
        )

        dist = df["dist"].to_numpy(dtype=np.float32)
        target = df["targetTrackPos"].to_numpy(dtype=np.float32)

        if len(dist) < 2:
            raise ValueError("Racing line CSV has too few unique distance bins.")

        return cls(
            dist=dist,
            target_track_pos=target,
            track_length=float(track_length),
        )

    def get(self, dist_from_start: float) -> float:
        # Periodic interpolation handles the start line wrap correctly.
        return float(
            np.interp(
                dist_from_start % self.track_length,
                self.dist,
                self.target_track_pos,
                period=self.track_length,
            )
        )