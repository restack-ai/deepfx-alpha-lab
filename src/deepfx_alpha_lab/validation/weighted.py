"""Sample-weight helpers for interval-aware validation.

This module bridges AFML Ch04 sample uniqueness with AFML Ch07 purged
validation. Use the returned ``sample_weight`` for model fitting inside each
train fold, never for selecting test rows.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from deepfx_alpha_lab.sample_weights import compute_interval_uniqueness

WeightMode = Literal["none", "symbol", "portfolio"]


def add_interval_uniqueness_weights(
    events: pd.DataFrame,
    *,
    mode: WeightMode = "symbol",
    symbol_col: str = "symbol",
    t0_col: str = "t0",
    t1_col: str = "t1",
) -> pd.DataFrame:
    """Return events with Ch04 uniqueness as ``sample_weight``.

    Parameters
    ----------
    events:
        Event DataFrame with interval columns.
    mode:
        ``none`` assigns unit weights, ``symbol`` computes uniqueness inside
        each symbol pool, and ``portfolio`` pools all events together.
    symbol_col:
        Symbol column required by ``mode='symbol'``.
    """

    if mode not in {"none", "symbol", "portfolio"}:
        raise ValueError(f"unsupported weight mode: {mode}")

    out = events.copy()
    if out.empty:
        out["sample_weight"] = pd.Series(dtype=float)
        out["effective_sample_size"] = pd.Series(dtype=float)
        return out

    if mode == "none":
        out["sample_weight"] = 1.0
        out["effective_sample_size"] = float(len(out))
        return out

    by = [symbol_col] if mode == "symbol" else []
    if mode == "symbol" and symbol_col not in out.columns:
        raise ValueError(f"symbol weight mode requires column: {symbol_col}")

    weighted = compute_interval_uniqueness(out, by=by, t0_col=t0_col, t1_col=t1_col)
    weighted["sample_weight"] = weighted["uniqueness"].astype(float)
    weighted["effective_sample_size"] = float(weighted["sample_weight"].sum())
    return weighted
