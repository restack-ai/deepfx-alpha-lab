from __future__ import annotations

import pandas as pd

from deepfx_alpha_lab.sample_weights import compute_interval_uniqueness, summarize_weighted_pnl


def test_compute_interval_uniqueness_uses_exact_overlap_integral() -> None:
    intervals = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "symbol": ["XAUUSD", "XAUUSD", "XAUUSD"],
            "t0": pd.to_datetime(
                ["2026-06-01T10:00:00Z", "2026-06-01T10:05:00Z", "2026-06-01T10:30:00Z"], utc=True
            ),
            "t1": pd.to_datetime(
                ["2026-06-01T10:10:00Z", "2026-06-01T10:15:00Z", "2026-06-01T10:40:00Z"], utc=True
            ),
            "pnl": [10.0, -2.0, 1.0],
        }
    )

    result = compute_interval_uniqueness(intervals, by=["symbol"])

    weights = dict(zip(result["event_id"], result["uniqueness"], strict=True))
    assert weights["a"] == 0.75
    assert weights["b"] == 0.75
    assert weights["c"] == 1.0
    assert result["concurrency_mean"].round(6).tolist() == [1.5, 1.5, 1.0]


def test_compute_interval_uniqueness_can_pool_across_symbols() -> None:
    intervals = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "symbol": ["XAUUSD", "XAGUSD"],
            "t0": pd.to_datetime(["2026-06-01T10:00:00Z", "2026-06-01T10:05:00Z"], utc=True),
            "t1": pd.to_datetime(["2026-06-01T10:10:00Z", "2026-06-01T10:15:00Z"], utc=True),
            "pnl": [10.0, -2.0],
        }
    )

    by_symbol = compute_interval_uniqueness(intervals, by=["symbol"])
    pooled = compute_interval_uniqueness(intervals, by=[])

    assert by_symbol["uniqueness"].tolist() == [1.0, 1.0]
    assert pooled["uniqueness"].tolist() == [0.75, 0.75]


def test_summarize_weighted_pnl_reports_effective_sample_size() -> None:
    weighted = pd.DataFrame(
        {
            "family": ["BRK", "BRK", "INF"],
            "alignment": ["opposed", "opposed", "aligned"],
            "pnl": [10.0, -2.0, 5.0],
            "uniqueness": [0.75, 0.75, 1.0],
            "duration_minutes": [10.0, 10.0, 20.0],
        }
    )

    summary = summarize_weighted_pnl(weighted, group_cols=["family", "alignment"])
    brk = summary[(summary["family"] == "BRK") & (summary["alignment"] == "opposed")].iloc[0]

    assert brk["raw_n"] == 2
    assert brk["effective_n"] == 1.5
    assert brk["raw_pnl_sum"] == 8.0
    assert brk["weighted_pnl_sum"] == 6.0
    assert brk["weighted_pnl_mean"] == 4.0
