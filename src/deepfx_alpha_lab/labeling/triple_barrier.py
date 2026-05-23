from __future__ import annotations

import numpy as np
import pandas as pd


def _apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float] | list[float],
) -> pd.DataFrame:
    out = events[["t1"]].copy()
    out["pt"] = pd.NaT
    out["sl"] = pd.NaT

    profit_taking = float(pt_sl[0]) if len(pt_sl) > 0 else 0.0
    stop_loss = float(pt_sl[1]) if len(pt_sl) > 1 else 0.0

    for t0, event in events.iterrows():
        t1 = event["t1"]
        if pd.isna(t1):
            price_path = close.loc[t0:]
        else:
            price_path = close.loc[t0:t1]
        if price_path.empty:
            continue

        side = float(event.get("side", 1.0))
        path_returns = (price_path / close.loc[t0] - 1.0) * side
        target = float(event["trgt"])

        if profit_taking > 0:
            touched = path_returns[path_returns > profit_taking * target]
            if not touched.empty:
                out.at[t0, "pt"] = touched.index[0]
        if stop_loss > 0:
            touched = path_returns[path_returns < -stop_loss * target]
            if not touched.empty:
                out.at[t0, "sl"] = touched.index[0]

    return out


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float] | list[float],
    trgt: pd.Series,
    min_ret: float,
    t1: pd.Series | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Create triple-barrier events and resolve the first touched barrier."""
    close = close.dropna().sort_index()
    trgt = trgt.reindex(t_events).dropna()
    trgt = trgt[trgt > min_ret]
    if trgt.empty:
        return pd.DataFrame(columns=["t1", "trgt", "type"])

    if t1 is None:
        t1 = pd.Series(pd.NaT, index=trgt.index)
    else:
        t1 = t1.reindex(trgt.index)

    events = pd.DataFrame({"t1": t1, "trgt": trgt})
    if side is not None:
        events["side"] = side.reindex(events.index)
        events = events.dropna(subset=["side"])

    touches = _apply_pt_sl_on_t1(close, events, pt_sl)
    first_touch = touches[["t1", "pt", "sl"]].min(axis=1)
    touch_type = []
    for t0, row in touches.iterrows():
        candidates = {
            "pt": row["pt"],
            "sl": row["sl"],
            "t1": row["t1"],
        }
        first = first_touch.loc[t0]
        selected = [name for name, value in candidates.items() if pd.notna(value) and value == first]
        touch_type.append(selected[0] if selected else "t1")

    events["t1"] = first_touch
    events["type"] = touch_type
    return events


def get_bins(close: pd.Series, events: pd.DataFrame, *, zero_on_vertical: bool = False) -> pd.DataFrame:
    """Compute returns and labels for resolved triple-barrier events."""
    if events.empty:
        return pd.DataFrame(columns=["ret", "bin", "type"])

    close = close.dropna().sort_index()
    events = events.dropna(subset=["t1"])
    valid_t1 = events["t1"].isin(close.index)
    events = events.loc[valid_t1]
    if events.empty:
        return pd.DataFrame(columns=["ret", "bin", "type"])

    out = pd.DataFrame(index=events.index)
    out["ret"] = close.reindex(events["t1"].to_numpy()).to_numpy() / close.reindex(events.index).to_numpy() - 1.0
    if "side" in events:
        out["ret"] *= events["side"].to_numpy()

    out["bin"] = np.sign(out["ret"]).astype("int64")
    if zero_on_vertical:
        out.loc[events["type"] == "t1", "bin"] = 0
    if "side" in events:
        out.loc[out["ret"] <= 0, "bin"] = 0

    out["type"] = events["type"]
    out["t1"] = events["t1"]
    return out
