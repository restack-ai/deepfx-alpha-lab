"""Purged and embargoed time-series cross-validation utilities.

AFML Ch07 warns that financial labels often span intervals, not points. Train
samples whose intervals overlap test labels leak information and must be purged;
nearby samples after the test window can also be embargoed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    """One purged walk-forward fold."""

    label: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    diagnostics: dict[str, int | str]


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    """Expanding walk-forward split with interval purging and embargo.

    By default, monthly test folds are created from event ``t0`` timestamps. Each
    fold trains on prior periods only, then removes train rows whose ``t0/t1``
    intervals overlap any test interval and rows inside the post-test embargo.
    """

    t0_col: str = "t0"
    t1_col: str = "t1"
    test_freq: str = "MS"
    min_train_periods: int = 1
    embargo: pd.Timedelta = pd.Timedelta(0)

    def split(self, events: pd.DataFrame) -> Iterator[PurgedFold]:
        frame = _normalize_events(events, self.t0_col, self.t1_col)
        if frame.empty:
            return
        event_periods = frame[self.t0_col].dt.tz_convert(None).dt.to_period(_period_alias(self.test_freq))
        periods = sorted(event_periods.unique())
        for period_idx, period in enumerate(periods):
            if period_idx < self.min_train_periods:
                continue
            test_mask = event_periods == period
            train_candidate_mask = event_periods < period
            test = frame[test_mask]
            train_candidates = frame[train_candidate_mask]
            if test.empty or train_candidates.empty:
                continue
            keep_candidate_mask, diagnostics = apply_purge_embargo(
                train_candidates,
                test,
                embargo=self.embargo,
                t0_col=self.t0_col,
                t1_col=self.t1_col,
            )
            train_indices = train_candidates.index[keep_candidate_mask].to_numpy(dtype=int)
            test_indices = test.index.to_numpy(dtype=int)
            diagnostics.update(
                {
                    "fold": f"test_{period}",
                    "candidate_train_rows": int(len(train_candidates)),
                    "train_rows": int(len(train_indices)),
                    "test_rows": int(len(test_indices)),
                }
            )
            yield PurgedFold(
                label=f"test_{period}",
                train_indices=train_indices,
                test_indices=test_indices,
                diagnostics=diagnostics,
            )


def interval_overlaps(
    starts: pd.Series | pd.DatetimeIndex,
    ends: pd.Series | pd.DatetimeIndex,
    test_starts: pd.Series | pd.DatetimeIndex,
    test_ends: pd.Series | pd.DatetimeIndex,
) -> pd.Series:
    """Return whether each half-open interval overlaps any test interval.

    Intervals are treated as ``[start, end)``. Boundary touching is not overlap.
    """

    starts_s = pd.Series(pd.to_datetime(starts, utc=True))
    ends_s = pd.Series(pd.to_datetime(ends, utc=True))
    test_starts_s = pd.Series(pd.to_datetime(test_starts, utc=True)).reset_index(drop=True)
    test_ends_s = pd.Series(pd.to_datetime(test_ends, utc=True)).reset_index(drop=True)
    out = pd.Series(False, index=starts_s.index)
    for test_start, test_end in zip(test_starts_s, test_ends_s, strict=True):
        out |= (starts_s < test_end) & (ends_s > test_start)
    return out


def apply_purge_embargo(
    train_candidates: pd.DataFrame,
    test: pd.DataFrame,
    *,
    embargo: pd.Timedelta = pd.Timedelta(0),
    t0_col: str = "t0",
    t1_col: str = "t1",
) -> tuple[pd.Series, dict[str, int]]:
    """Return train-candidate keep mask plus purge/embargo diagnostics."""

    train = _normalize_events(train_candidates, t0_col, t1_col)
    test_frame = _normalize_events(test, t0_col, t1_col)
    if train.empty:
        return pd.Series(dtype=bool), {"purged_overlap_rows": 0, "embargo_rows": 0}
    if test_frame.empty:
        return pd.Series(True, index=train.index), {"purged_overlap_rows": 0, "embargo_rows": 0}

    overlap_mask = interval_overlaps(train[t0_col], train[t1_col], test_frame[t0_col], test_frame[t1_col])
    overlap_mask.index = train.index

    embargo_mask = pd.Series(False, index=train.index)
    if embargo > pd.Timedelta(0):
        test_end = test_frame[t1_col].max()
        embargo_end = test_end + embargo
        embargo_mask = (train[t0_col] >= test_end) & (train[t0_col] < embargo_end)

    keep = ~(overlap_mask | embargo_mask)
    return keep, {
        "purged_overlap_rows": int(overlap_mask.sum()),
        "embargo_rows": int((embargo_mask & ~overlap_mask).sum()),
    }


def _normalize_events(events: pd.DataFrame, t0_col: str, t1_col: str) -> pd.DataFrame:
    missing = sorted({t0_col, t1_col} - set(events.columns))
    if missing:
        raise ValueError(f"events missing required columns: {missing}")
    frame = events.copy()
    frame[t0_col] = pd.to_datetime(frame[t0_col], utc=True)
    frame[t1_col] = pd.to_datetime(frame[t1_col], utc=True)
    frame.loc[frame[t1_col] < frame[t0_col], t1_col] = frame.loc[frame[t1_col] < frame[t0_col], t0_col]
    return frame


def _period_alias(freq: str) -> str:
    # pandas Period does not accept 'MS' directly; monthly periods use 'M'.
    if freq.upper() in {"MS", "ME"}:
        return "M"
    return freq
