from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

"""
NOTE: This implementation is completly taken from
* https://github.com/dergestaler/estimate_curvatures
"""

@dataclass(slots=True)
class RacingLineCurvatureMap:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    curvature: np.ndarray
    track_length: float

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        x_col: str = "x",
        y_col: str = "y",
        dist_col: str = "distFromStart",
        spacing: float = 1.0,
        smoothing_window: int = 21,
        max_step_xy: float = 25.0,
        min_lap_coverage: float = 0.90,
        outlier_sigma: float = 3.5,
        max_abs_curvature: float = 0.5,
    ) -> "RacingLineCurvatureMap":
        df = pd.read_csv(csv_path)

        missing = [c for c in (x_col, y_col, dist_col) if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path} is missing columns {missing}. Found: {list(df.columns)}"
            )

        x_raw = df[x_col].to_numpy(dtype=np.float64)
        y_raw = df[y_col].to_numpy(dtype=np.float64)
        d_raw = df[dist_col].to_numpy(dtype=np.float64)

        valid = np.isfinite(x_raw) & np.isfinite(y_raw) & np.isfinite(d_raw)
        x_raw = x_raw[valid]
        y_raw = y_raw[valid]
        d_raw = d_raw[valid]

        if len(d_raw) < 50:
            raise ValueError(f"Not enough valid racing-line points in {csv_path}")

        # Estimate track length
        rough_track_length = float(np.nanpercentile(d_raw, 99.9))
        if rough_track_length <= 1.0:
            raise ValueError(f"Invalid track length from {csv_path}: {rough_track_length}")

        wraps = np.where(np.diff(d_raw) < -0.5 * rough_track_length)[0] + 1
        lap_starts = np.concatenate([[0], wraps])
        lap_ends = np.concatenate([wraps, [len(d_raw)]])

        # Use the median of completed lap maxima as track length
        lap_maxima = []
        for start, end in zip(lap_starts, lap_ends):
            d_lap = d_raw[start:end]
            if len(d_lap) >= 50:
                lap_maxima.append(float(np.max(d_lap)))

        if not lap_maxima:
            raise ValueError(f"No usable laps found in {csv_path}")

        track_length = float(np.median(lap_maxima))
        if track_length <= 1.0:
            raise ValueError(f"Invalid track length from {csv_path}: {track_length}")

        s_grid = np.arange(0.0, track_length, spacing, dtype=np.float64)

        lap_xs: list[np.ndarray] = []
        lap_ys: list[np.ndarray] = []

        for start, end in zip(lap_starts, lap_ends):
            d_lap = d_raw[start:end]
            x_lap = x_raw[start:end]
            y_lap = y_raw[start:end]

            if len(d_lap) < 50:
                continue

            # Sort by distFromStart inside lap
            order = np.argsort(d_lap)
            d_lap = d_lap[order]
            x_lap = x_lap[order]
            y_lap = y_lap[order]

            # Remove duplicate or non-increasing distance samples
            keep = np.concatenate([[True], np.diff(d_lap) > 1e-6])
            d_lap = d_lap[keep]
            x_lap = x_lap[keep]
            y_lap = y_lap[keep]

            if len(d_lap) < 50:
                continue

            # Reject incomplete laps
            coverage = (float(np.max(d_lap)) - float(np.min(d_lap))) / track_length
            if coverage < min_lap_coverage:
                continue

            # Reject laps with impossible XY jumps
            step_xy = np.hypot(np.diff(x_lap), np.diff(y_lap))
            if np.nanmax(step_xy) > max_step_xy:
                continue

            # Periodic interpolation
            d_ext = np.concatenate([[d_lap[-1] - track_length], d_lap, [d_lap[0] + track_length]])
            x_ext = np.concatenate([[x_lap[-1]], x_lap, [x_lap[0]]])
            y_ext = np.concatenate([[y_lap[-1]], y_lap, [y_lap[0]]])

            x_interp = np.interp(s_grid, d_ext, x_ext)
            y_interp = np.interp(s_grid, d_ext, y_ext)

            if np.all(np.isfinite(x_interp)) and np.all(np.isfinite(y_interp)):
                lap_xs.append(x_interp)
                lap_ys.append(y_interp)

        if len(lap_xs) < 2:
            raise ValueError(
                f"Only {len(lap_xs)} usable laps found in {csv_path}. "
                f"CSV likely contains broken/incomplete laps."
            )

        xs = np.stack(lap_xs, axis=0)
        ys = np.stack(lap_ys, axis=0)

        # Robust centerline: median first.
        x_med = np.median(xs, axis=0)
        y_med = np.median(ys, axis=0)

        # Reject per-sample lap outliers based on distance to median position
        dist_to_med = np.hypot(xs - x_med[None, :], ys - y_med[None, :])
        med_dist = np.median(dist_to_med, axis=0)
        mad_dist = np.median(np.abs(dist_to_med - med_dist[None, :]), axis=0)
        robust_std = 1.4826 * mad_dist + 1e-6

        inlier = dist_to_med <= (med_dist[None, :] + outlier_sigma * robust_std[None, :])

        xs_clean = np.where(inlier, xs, np.nan)
        ys_clean = np.where(inlier, ys, np.nan)

        x_path = np.nanmedian(xs_clean, axis=0)
        y_path = np.nanmedian(ys_clean, axis=0)

        # Fallback if all values were NaN somewhere.
        bad = ~np.isfinite(x_path) | ~np.isfinite(y_path)
        x_path[bad] = x_med[bad]
        y_path[bad] = y_med[bad]

        if smoothing_window > 1:
            x_path = cls._circular_moving_average(x_path, smoothing_window)
            y_path = cls._circular_moving_average(y_path, smoothing_window)

        curvature = cls._finite_difference_curvature(
            x_path,
            y_path,
            spacing=spacing,
        )

        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
        curvature = np.clip(curvature, -max_abs_curvature, max_abs_curvature)

        return cls(
            s=s_grid.astype(np.float32),
            x=x_path.astype(np.float32),
            y=y_path.astype(np.float32),
            curvature=curvature.astype(np.float32),
            track_length=track_length,
        )

    @staticmethod
    def _circular_moving_average(values: np.ndarray, window: int) -> np.ndarray:
        window = int(window)
        if window <= 1:
            return values.copy()

        # Force odd window for centered smoothing.
        if window % 2 == 0:
            window += 1

        pad = window // 2
        padded = np.concatenate([values[-pad:], values, values[:pad]])
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _finite_difference_curvature(x: np.ndarray, y: np.ndarray, *, spacing: float) -> np.ndarray:
        # Periodic central differences.
        x_prev = np.roll(x, 1)
        x_next = np.roll(x, -1)
        y_prev = np.roll(y, 1)
        y_next = np.roll(y, -1)

        dx = (x_next - x_prev) / (2.0 * spacing)
        dy = (y_next - y_prev) / (2.0 * spacing)
        ddx = (x_next - 2.0 * x + x_prev) / (spacing * spacing)
        ddy = (y_next - 2.0 * y + y_prev) / (spacing * spacing)

        denom = (dx * dx + dy * dy) ** 1.5
        curvature = np.zeros_like(x, dtype=np.float64)
        good = denom > 1e-6
        curvature[good] = (dx[good] * ddy[good] - dy[good] * ddx[good]) / denom[good]

        return curvature

    def lookahead(self, dist_from_start: float, distances: float | Sequence[float] | np.ndarray) -> np.ndarray:
        distances = np.asarray(distances, dtype=np.float64)
        query = (float(dist_from_start) + distances) % self.track_length
        return np.interp(
            query,
            self.s,
            self.curvature,
            period=self.track_length,
        ).astype(np.float32)
