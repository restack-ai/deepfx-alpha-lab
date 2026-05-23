from __future__ import annotations

import numpy as np
import pandas as pd


def symmetric_cusum_filter(close: pd.Series, threshold: float | pd.Series) -> pd.DatetimeIndex:
    """Sample events when cumulative log returns exceed a symmetric threshold."""
    close = close.dropna().sort_index()
    log_returns = np.log(close).diff().dropna()
    if log_returns.empty:
        return pd.DatetimeIndex([])

    if isinstance(threshold, pd.Series):
        threshold = threshold.reindex(log_returns.index, method="ffill")
    else:
        threshold = pd.Series(float(threshold), index=log_returns.index)

    threshold = threshold.dropna()
    log_returns = log_returns.reindex(threshold.index).dropna()
    threshold = threshold.reindex(log_returns.index)

    positive_sum = 0.0
    negative_sum = 0.0
    events: list[pd.Timestamp] = []

    for timestamp, value in log_returns.items():
        h = float(threshold.loc[timestamp])
        if not np.isfinite(h) or h <= 0:
            continue

        positive_sum = max(0.0, positive_sum + float(value))
        negative_sum = min(0.0, negative_sum + float(value))

        if negative_sum < -h:
            negative_sum = 0.0
            events.append(timestamp)
        elif positive_sum > h:
            positive_sum = 0.0
            events.append(timestamp)

    return pd.DatetimeIndex(events)
