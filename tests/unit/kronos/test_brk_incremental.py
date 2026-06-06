from __future__ import annotations

import pandas as pd

from deepfx_alpha_lab.kronos.brk_incremental import (
    add_feature_buckets,
    evaluate_feature_sets,
    leave_one_day_out_group_mean,
    summarize_kronos_lift,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                [
                    "2026-06-01T10:00:00Z",
                    "2026-06-01T11:00:00Z",
                    "2026-06-02T10:00:00Z",
                    "2026-06-02T11:00:00Z",
                    "2026-06-03T10:00:00Z",
                    "2026-06-03T11:00:00Z",
                ],
                utc=True,
            ),
            "entry_day": ["2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02", "2026-06-03", "2026-06-03"],
            "symbol": ["XAUUSD"] * 6,
            "side": ["SHORT"] * 6,
            "risk_regime": ["mixed", "mixed", "mixed", "mixed", "mixed", "mixed"],
            "alignment": ["opposed", "aligned", "opposed", "aligned", "opposed", "aligned"],
            "signal_direction": ["LONG", "SHORT", "LONG", "SHORT", "LONG", "SHORT"],
            "signal_age_minutes": [10.0, 20.0, 15.0, 25.0, 12.0, 30.0],
            "signal_predicted_bps": [30.0, -12.0, 35.0, -10.0, 40.0, -11.0],
            "bb_width_pct": [0.2, 0.2, 0.3, 0.4, 0.1, 0.5],
            "atr_pct": [0.7, 0.8, 0.6, 0.7, 0.9, 0.6],
            "body_atr": [1.0, 0.1, 1.2, 0.2, 1.1, 0.3],
            "pnl": [10.0, -2.0, 9.0, -1.0, 11.0, -3.0],
        }
    )


def test_add_feature_buckets_creates_discrete_features() -> None:
    frame = add_feature_buckets(_sample_frame())

    assert set(frame["bb_width_bucket"]) <= {"low", "mid", "high", "unknown"}
    assert set(frame["signal_age_bucket"]) <= {"fresh", "recent", "stale", "none", "unknown"}
    assert "pred_abs_bucket" in frame.columns


def test_leave_one_day_out_group_mean_uses_training_days_only() -> None:
    frame = add_feature_buckets(_sample_frame())

    pred = leave_one_day_out_group_mean(frame, ["symbol", "side", "risk_regime", "alignment"], min_train_group_size=2)
    day1_opposed = pred[(pred["entry_day"] == "2026-06-01") & (pred["alignment"] == "opposed")].iloc[0]

    assert day1_opposed["prediction"] == 10.0  # mean of day2/day3 opposed: (9 + 11) / 2
    assert day1_opposed["fallback_level"] == "symbol+side+risk_regime+alignment"


def test_evaluate_feature_sets_returns_ranked_metrics() -> None:
    frame = add_feature_buckets(_sample_frame())

    result = evaluate_feature_sets(frame, min_train_group_size=2)

    assert set(result["feature_set"]) >= {"global_mean", "brk_regime", "brk_kronos_alignment"}
    assert result["mae"].notna().all()
    kronos = result[result["feature_set"] == "brk_kronos_alignment"].iloc[0]
    baseline = result[result["feature_set"] == "brk_regime"].iloc[0]
    assert kronos["mae"] < baseline["mae"]


def test_summarize_kronos_lift_adds_alignment_buckets() -> None:
    frame = add_feature_buckets(_sample_frame())

    lift = summarize_kronos_lift(frame)

    aligned_rows = lift[lift["bucket_family"] == "brk_regime_alignment"]
    assert set(aligned_rows["alignment"]) == {"aligned", "opposed"}
    assert aligned_rows[aligned_rows["alignment"] == "opposed"]["pnl_mean"].iloc[0] == 10.0
