"""AFML Chapter 4 sample weighting utilities.

The core idea is sample uniqueness: if many labeled events overlap in time,
each event contributes less independent information than a non-overlapping event.
This module computes exact interval uniqueness for trade/event intervals.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def compute_interval_uniqueness(
    intervals: pd.DataFrame,
    *,
    by: Sequence[str] = ("symbol",),
    t0_col: str = "t0",
    t1_col: str = "t1",
) -> pd.DataFrame:
    """Compute exact average uniqueness for interval-labeled samples.

    For each group defined by ``by``, concurrency is piecewise constant across all
    interval endpoints. An event's uniqueness is the duration-weighted average of
    ``1 / concurrency`` across its active interval.

    Parameters
    ----------
    intervals:
        DataFrame with start/end timestamps.
    by:
        Columns that define independent concurrency pools. Use ``[]`` to pool all
        events portfolio-wide.
    t0_col / t1_col:
        Start and end timestamp columns.
    """

    if intervals.empty:
        return intervals.copy().assign(
            uniqueness=pd.Series(dtype=float),
            concurrency_mean=pd.Series(dtype=float),
            duration_minutes=pd.Series(dtype=float),
        )
    required = {t0_col, t1_col, *by}
    missing = sorted(required - set(intervals.columns))
    if missing:
        raise ValueError(f"intervals missing required columns: {missing}")

    frame = intervals.copy()
    frame[t0_col] = pd.to_datetime(frame[t0_col], utc=True)
    frame[t1_col] = pd.to_datetime(frame[t1_col], utc=True)
    if (frame[t1_col] < frame[t0_col]).any():
        raise ValueError("interval end time precedes start time")

    outputs: list[pd.DataFrame] = []
    if by:
        grouped = frame.groupby(list(by), dropna=False, sort=False)
        for _key, group in grouped:
            outputs.append(_compute_group_uniqueness(group, t0_col=t0_col, t1_col=t1_col))
    else:
        outputs.append(_compute_group_uniqueness(frame, t0_col=t0_col, t1_col=t1_col))
    return pd.concat(outputs).sort_index().reset_index(drop=True)


def summarize_weighted_pnl(
    weighted: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    pnl_col: str = "pnl",
    weight_col: str = "uniqueness",
) -> pd.DataFrame:
    """Summarize raw and uniqueness-weighted PnL by groups."""

    required = {pnl_col, weight_col, *group_cols}
    missing = sorted(required - set(weighted.columns))
    if missing:
        raise ValueError(f"weighted frame missing required columns: {missing}")
    if weighted.empty:
        return pd.DataFrame(
            columns=list(group_cols)
            + [
                "raw_n",
                "effective_n",
                "mean_uniqueness",
                "raw_pnl_sum",
                "raw_pnl_mean",
                "weighted_pnl_sum",
                "weighted_pnl_mean",
                "win_rate",
                "weighted_win_rate",
                "mean_duration_minutes",
            ]
        )

    frame = weighted.copy()
    frame[pnl_col] = pd.to_numeric(frame[pnl_col], errors="raise")
    frame[weight_col] = pd.to_numeric(frame[weight_col], errors="raise")

    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        weights = group[weight_col].astype(float)
        pnl = group[pnl_col].astype(float)
        effective_n = float(weights.sum())
        row: dict[str, object] = {col: value for col, value in zip(group_cols, key, strict=True)}
        row.update(
            {
                "raw_n": int(len(group)),
                "effective_n": effective_n,
                "mean_uniqueness": float(weights.mean()) if len(group) else np.nan,
                "raw_pnl_sum": float(pnl.sum()),
                "raw_pnl_mean": float(pnl.mean()) if len(group) else np.nan,
                "weighted_pnl_sum": float((pnl * weights).sum()),
                "weighted_pnl_mean": float((pnl * weights).sum() / effective_n) if effective_n else np.nan,
                "win_rate": float((pnl > 0).mean()) if len(group) else np.nan,
                "weighted_win_rate": float(((pnl > 0).astype(float) * weights).sum() / effective_n) if effective_n else np.nan,
                "mean_duration_minutes": float(group["duration_minutes"].mean()) if "duration_minutes" in group.columns else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_group_uniqueness(group: pd.DataFrame, *, t0_col: str, t1_col: str) -> pd.DataFrame:
    out = group.copy()
    endpoints = sorted(set(out[t0_col].tolist() + out[t1_col].tolist()))
    if len(endpoints) <= 1:
        out["duration_minutes"] = 0.0
        out["uniqueness"] = 1.0
        out["concurrency_mean"] = 1.0
        return out

    uniqueness: list[float] = []
    concurrency_mean: list[float] = []
    durations_minutes: list[float] = []
    for row in out.itertuples(index=False):
        t0 = getattr(row, t0_col)
        t1 = getattr(row, t1_col)
        duration_seconds = max((t1 - t0).total_seconds(), 0.0)
        durations_minutes.append(duration_seconds / 60.0)
        if duration_seconds == 0.0:
            instant_concurrency = int(((out[t0_col] <= t0) & (out[t1_col] >= t0)).sum())
            concurrency = max(instant_concurrency, 1)
            uniqueness.append(1.0 / concurrency)
            concurrency_mean.append(float(concurrency))
            continue

        unique_integral = 0.0
        concurrency_integral = 0.0
        for left, right in zip(endpoints[:-1], endpoints[1:], strict=True):
            if right <= t0 or left >= t1:
                continue
            segment_start = max(left, t0)
            segment_end = min(right, t1)
            segment_seconds = (segment_end - segment_start).total_seconds()
            if segment_seconds <= 0:
                continue
            active = (out[t0_col] < segment_end) & (out[t1_col] > segment_start)
            concurrency = int(active.sum())
            if concurrency <= 0:
                continue
            unique_integral += segment_seconds / concurrency
            concurrency_integral += segment_seconds * concurrency
        uniqueness.append(float(unique_integral / duration_seconds))
        concurrency_mean.append(float(concurrency_integral / duration_seconds))

    out["duration_minutes"] = durations_minutes
    out["uniqueness"] = uniqueness
    out["concurrency_mean"] = concurrency_mean
    return out
