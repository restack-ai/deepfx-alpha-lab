from __future__ import annotations

import pandas as pd


def add_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    *,
    num_days: int | float = 1,
) -> pd.Series:
    """Map each event timestamp to the first close index at or after t0 + num_days."""
    close = close.dropna().sort_index()
    close.index = pd.DatetimeIndex(close.index).as_unit("ns")
    t_events = pd.DatetimeIndex(t_events).as_unit("ns")
    if close.empty or len(t_events) == 0:
        return pd.Series(dtype="datetime64[ns]")

    barrier_times = t_events + pd.Timedelta(days=num_days)
    positions = close.index.searchsorted(barrier_times)
    valid = positions < close.shape[0]
    return pd.Series(close.index[positions[valid]], index=t_events[valid])
