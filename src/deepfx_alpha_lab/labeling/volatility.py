from __future__ import annotations

import pandas as pd


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Estimate daily volatility using EWM standard deviation of 1-day returns."""
    close = close.dropna().sort_index()
    if close.empty:
        return pd.Series(dtype="float64")

    one_day_ago = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    valid = one_day_ago > 0
    positions = one_day_ago[valid]
    current_index = close.index[valid]
    previous_index = close.index[positions - 1]

    daily_returns = pd.Series(
        close.loc[current_index].to_numpy() / close.loc[previous_index].to_numpy() - 1.0,
        index=current_index,
    )
    return daily_returns.ewm(span=span).std()
