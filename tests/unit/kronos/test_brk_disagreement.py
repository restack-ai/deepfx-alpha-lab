from __future__ import annotations

import pandas as pd

from deepfx_alpha_lab.kronos.brk_disagreement import (
    add_ohlcv_features,
    attach_latest_prior_signals,
    classify_cross_asset_regime,
    filter_brk_trades,
    summarize_alignment,
)


def test_attach_latest_prior_signal_is_prior_only_and_windowed() -> None:
    signals = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-06-01T09:00:00Z", "2026-06-01T09:50:00Z", "2026-06-01T10:01:00Z"], utc=True
            ),
            "symbol": ["XAUUSD", "XAUUSD", "XAUUSD"],
            "direction": ["LONG", "SHORT", "LONG"],
            "predicted_bps": [20.0, -30.0, 40.0],
            "horizon": [12, 12, 12],
            "lookback": [256, 256, 256],
            "model": ["m", "m", "m"],
            "tag": ["t", "t", "t"],
        }
    )
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2026-06-01T10:00:00Z"], utc=True),
            "symbol": ["XAUUSD"],
            "side": ["LONG"],
            "pnl": [1.0],
            "reason": ["BRK BRK"],
            "family": ["BRK"],
        }
    )

    aligned = attach_latest_prior_signals(signals, trades, window_minutes=15)

    assert aligned.loc[0, "signal_time"] == pd.Timestamp("2026-06-01T09:50:00Z")
    assert aligned.loc[0, "signal_direction"] == "SHORT"
    assert aligned.loc[0, "alignment"] == "opposed"


def test_attach_latest_prior_signal_marks_no_recent_when_outside_window() -> None:
    signals = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-06-01T09:00:00Z"], utc=True),
            "symbol": ["XAUUSD"],
            "direction": ["LONG"],
            "predicted_bps": [20.0],
            "horizon": [12],
            "lookback": [256],
            "model": ["m"],
            "tag": ["t"],
        }
    )
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2026-06-01T10:00:00Z"], utc=True),
            "symbol": ["XAUUSD"],
            "side": ["LONG"],
            "pnl": [1.0],
            "reason": ["BRK BRK"],
            "family": ["BRK"],
        }
    )

    aligned = attach_latest_prior_signals(signals, trades, window_minutes=30)

    assert aligned.loc[0, "alignment"] == "no_recent_kronos"
    assert pd.isna(aligned.loc[0, "signal_time"])


def test_filter_brk_trades_uses_family_or_reason_prefix() -> None:
    trades = pd.DataFrame(
        {
            "family": ["BRK", "INF", ""],
            "reason": ["whatever", "BRK fallback", "INF thing"],
            "pnl": [1, 2, 3],
        }
    )

    brk = filter_brk_trades(trades)

    assert brk["pnl"].tolist() == [1, 2]


def test_ohlcv_features_and_partial_regime() -> None:
    times = pd.date_range("2026-06-01T00:00:00Z", periods=40, freq="5min")
    ohlcv = pd.DataFrame(
        {
            "time": list(times) * 2,
            "symbol": ["XAUUSD"] * len(times) + ["XAGUSD"] * len(times),
            "open": [100 + i * 0.1 for i in range(len(times))] + [50 + i * 0.05 for i in range(len(times))],
            "high": [100.2 + i * 0.1 for i in range(len(times))] + [50.1 + i * 0.05 for i in range(len(times))],
            "low": [99.8 + i * 0.1 for i in range(len(times))] + [49.9 + i * 0.05 for i in range(len(times))],
            "close": [100.1 + i * 0.1 for i in range(len(times))] + [50.05 + i * 0.05 for i in range(len(times))],
        }
    )

    featured = add_ohlcv_features(ohlcv)
    regime = classify_cross_asset_regime(featured, times[-1])

    assert "atr" in featured.columns
    assert "bb_width_pct" in featured.columns
    assert regime in {"metals_up_partial", "mixed_partial"}


def test_summarize_alignment_includes_pnl_and_win_rate() -> None:
    frame = pd.DataFrame(
        {
            "window_minutes": [120, 120, 120],
            "alignment": ["opposed", "opposed", "aligned"],
            "pnl": [10.0, -2.0, 1.0],
            "r_multiple": [1.0, -0.2, 0.1],
            "atr_norm_return": [0.5, -0.1, 0.2],
        }
    )

    summary = summarize_alignment(frame, ["window_minutes", "alignment"])
    opposed = summary[summary["alignment"] == "opposed"].iloc[0]

    assert opposed["trades"] == 2
    assert opposed["pnl_sum"] == 8.0
    assert opposed["win_rate"] == 0.5
