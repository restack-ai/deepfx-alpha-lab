from __future__ import annotations

import pandas as pd
import pytest

from deepfx_alpha_lab.validation.weighted import add_interval_uniqueness_weights


def test_add_interval_uniqueness_weights_uses_symbol_pools() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "symbol": ["XAUUSD", "XAUUSD", "XAGUSD"],
            "t0": pd.to_datetime(
                ["2026-01-01T10:00:00Z", "2026-01-01T10:05:00Z", "2026-01-01T10:05:00Z"], utc=True
            ),
            "t1": pd.to_datetime(
                ["2026-01-01T10:10:00Z", "2026-01-01T10:15:00Z", "2026-01-01T10:15:00Z"], utc=True
            ),
        }
    )

    weighted = add_interval_uniqueness_weights(events, mode="symbol")

    weights = dict(zip(weighted["event_id"], weighted["sample_weight"], strict=True))
    assert weights == {"a": 0.75, "b": 0.75, "c": 1.0}
    assert weighted["effective_sample_size"].iloc[0] == pytest.approx(2.5)


def test_add_interval_uniqueness_weights_can_pool_portfolio_wide() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "symbol": ["XAUUSD", "XAGUSD"],
            "t0": pd.to_datetime(["2026-01-01T10:00:00Z", "2026-01-01T10:05:00Z"], utc=True),
            "t1": pd.to_datetime(["2026-01-01T10:10:00Z", "2026-01-01T10:15:00Z"], utc=True),
        }
    )

    weighted = add_interval_uniqueness_weights(events, mode="portfolio")

    assert weighted["sample_weight"].tolist() == [0.75, 0.75]
    assert weighted["effective_sample_size"].iloc[0] == pytest.approx(1.5)


def test_add_interval_uniqueness_weights_supports_no_weight_mode() -> None:
    events = pd.DataFrame(
        {
            "t0": pd.to_datetime(["2026-01-01T10:00:00Z", "2026-01-01T10:05:00Z"], utc=True),
            "t1": pd.to_datetime(["2026-01-01T10:10:00Z", "2026-01-01T10:15:00Z"], utc=True),
        }
    )

    weighted = add_interval_uniqueness_weights(events, mode="none")

    assert weighted["sample_weight"].tolist() == [1.0, 1.0]
    assert weighted["effective_sample_size"].iloc[0] == pytest.approx(2.0)
