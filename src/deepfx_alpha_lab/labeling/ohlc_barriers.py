from __future__ import annotations

import pandas as pd


_VALID_AMBIGUOUS_POLICIES = {"sl_first", "pt_first", "ambiguous"}


def _empty_ohlc_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["t1", "trgt", "type", "ret", "bin", "ambiguous"])


def _select_same_bar_touch(
    *,
    timestamp: pd.Timestamp,
    pt_touched: bool,
    sl_touched: bool,
    upper_ret: float,
    lower_ret: float,
    ambiguous_policy: str,
) -> tuple[pd.Timestamp | None, str | None, float | None, bool]:
    if pt_touched and sl_touched:
        if ambiguous_policy == "pt_first":
            return timestamp, "pt", upper_ret, True
        if ambiguous_policy == "sl_first":
            return timestamp, "sl", lower_ret, True
        return timestamp, "ambiguous", 0.0, True
    if pt_touched:
        return timestamp, "pt", upper_ret, False
    if sl_touched:
        return timestamp, "sl", lower_ret, False
    return None, None, None, False


def resolve_ohlc_events(
    entry_close: pd.Series,
    path_bars: pd.DataFrame,
    t_events: pd.DatetimeIndex,
    t1: pd.Series,
    trgt: pd.Series,
    *,
    pt: float,
    sl: float,
    min_ret: float,
    ambiguous_policy: str = "sl_first",
) -> pd.DataFrame:
    """Resolve triple-barrier events using intrabar OHLC high/low touches.

    This is execution-aware relative to close-path labeling: profit-taking and
    stop-loss are considered touched if the path bar's high or low crosses the
    corresponding barrier before the vertical barrier.

    Same-bar high/low ambiguity is unavoidable with OHLC bars because the order
    within the bar is unknown. The default `sl_first` policy is conservative.
    """
    if ambiguous_policy not in _VALID_AMBIGUOUS_POLICIES:
        raise ValueError(f"ambiguous_policy must be one of {sorted(_VALID_AMBIGUOUS_POLICIES)}")

    required = {"high", "low", "close"}
    missing = required.difference(path_bars.columns)
    if missing:
        raise ValueError(f"path_bars missing required columns: {sorted(missing)}")

    entry_close = entry_close.dropna().sort_index()
    path_bars = path_bars.dropna(subset=["high", "low", "close"]).sort_index()
    trgt = trgt.reindex(t_events).dropna()
    trgt = trgt[trgt > min_ret]
    t1 = t1.reindex(trgt.index).dropna()
    event_index = trgt.index.intersection(t1.index).intersection(entry_close.index)

    rows: list[dict[str, object]] = []
    for t0 in event_index:
        entry_price = float(entry_close.loc[t0])
        target = float(trgt.loc[t0])
        vertical_time = t1.loc[t0]
        path = path_bars.loc[(path_bars.index > t0) & (path_bars.index <= vertical_time)]
        if path.empty:
            continue

        upper_ret = float(pt * target)
        lower_ret = float(-sl * target)
        first_time: pd.Timestamp | None = None
        first_type: str | None = None
        first_ret: float | None = None
        ambiguous = False

        for timestamp, row in path.iterrows():
            high_ret = float(row["high"] / entry_price - 1.0)
            low_ret = float(row["low"] / entry_price - 1.0)
            pt_touched = pt > 0 and high_ret >= upper_ret
            sl_touched = sl > 0 and low_ret <= lower_ret
            first_time, first_type, first_ret, ambiguous = _select_same_bar_touch(
                timestamp=timestamp,
                pt_touched=pt_touched,
                sl_touched=sl_touched,
                upper_ret=upper_ret,
                lower_ret=lower_ret,
                ambiguous_policy=ambiguous_policy,
            )
            if first_time is not None:
                break

        if first_time is None:
            last_bar = path.iloc[-1]
            first_time = path.index[-1]
            first_type = "t1"
            first_ret = float(last_bar["close"] / entry_price - 1.0)
            ambiguous = False

        if first_type == "ambiguous":
            bin_value = 0
        else:
            bin_value = int(1 if first_ret > 0 else -1 if first_ret < 0 else 0)

        rows.append(
            {
                "t0": t0,
                "t1": first_time,
                "trgt": target,
                "type": first_type,
                "ret": first_ret,
                "bin": bin_value,
                "ambiguous": ambiguous,
            }
        )

    if not rows:
        return _empty_ohlc_events()
    return pd.DataFrame(rows).set_index("t0").sort_index()
