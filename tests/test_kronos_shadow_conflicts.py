import pandas as pd
import pytest

from deepfx_alpha_lab.kronos.shadow_conflicts import apply_shadow_conflict_policy


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "XAUUSD",
                "time": pd.Timestamp("2026-06-01 00:00", tz="UTC"),
                "direction": "LONG",
                "predicted_bps": 12.0,
            },
            {
                "symbol": "XAUUSD",
                "time": pd.Timestamp("2026-06-01 00:00", tz="UTC"),
                "direction": "SHORT",
                "predicted_bps": -15.0,
            },
            {
                "symbol": "XAGUSD",
                "time": pd.Timestamp("2026-06-01 00:05", tz="UTC"),
                "direction": "BUY",
                "predicted_bps": 11.0,
            },
            {
                "symbol": "XAGUSD",
                "time": pd.Timestamp("2026-06-01 00:05", tz="UTC"),
                "direction": "LONG",
                "predicted_bps": 20.0,
            },
            {
                "symbol": "XAUUSD",
                "time": pd.Timestamp("2026-06-01 00:10", tz="UTC"),
                "direction": "SELL",
                "predicted_bps": -18.0,
            },
        ]
    )


def test_discard_conflict_removes_bidirectional_groups_and_keeps_strongest_duplicate():
    resolved, summary = apply_shadow_conflict_policy(_signals(), policy="discard_conflict")

    assert summary.input_rows == 5
    assert summary.groups == 3
    assert summary.conflict_groups == 1
    assert summary.same_direction_duplicate_groups == 1
    assert summary.discarded_groups == 1
    assert summary.output_rows == 2
    assert summary.discarded_rows == 3

    assert resolved["symbol"].tolist() == ["XAGUSD", "XAUUSD"]
    assert resolved["direction"].tolist() == ["LONG", "SHORT"]
    assert resolved["predicted_bps"].tolist() == [20.0, -18.0]
    assert resolved.loc[0, "group_size"] == 2
    assert resolved.loc[0, "conflict_group"] == False  # noqa: E712


def test_strongest_abs_pred_keeps_one_row_per_group_and_marks_conflict_metadata():
    resolved, summary = apply_shadow_conflict_policy(_signals(), policy="strongest_abs_pred")

    assert summary.output_rows == 3
    xau_conflict = resolved[(resolved["symbol"] == "XAUUSD") & (resolved["time"] == pd.Timestamp("2026-06-01 00:00", tz="UTC"))].iloc[0]
    assert xau_conflict["direction"] == "SHORT"
    assert xau_conflict["predicted_bps"] == -15.0
    assert xau_conflict["conflict_group"] == True  # noqa: E712
    assert xau_conflict["long_count"] == 1
    assert xau_conflict["short_count"] == 1
    assert xau_conflict["gross_predicted_bps"] == 27.0
    assert xau_conflict["net_predicted_bps"] == -3.0


def test_net_pred_collapses_groups_by_signed_sum_and_optional_threshold():
    resolved, summary = apply_shadow_conflict_policy(_signals(), policy="net_pred", threshold_bps=10.0)

    assert summary.output_rows == 2
    assert summary.discarded_groups == 1
    assert resolved["symbol"].tolist() == ["XAGUSD", "XAUUSD"]
    assert resolved["direction"].tolist() == ["LONG", "SHORT"]
    assert resolved["predicted_bps"].tolist() == [31.0, -18.0]
    assert resolved["conflict_policy"].tolist() == ["net_pred", "net_pred"]


def test_rejects_unknown_direction_values():
    signals = _signals()
    signals.loc[0, "direction"] = "FLAT"

    with pytest.raises(ValueError, match="unsupported direction"):
        apply_shadow_conflict_policy(signals, policy="strongest_abs_pred")


def test_rejects_non_finite_predictions_and_sign_mismatch():
    signals = _signals()
    signals.loc[0, "predicted_bps"] = float("nan")

    with pytest.raises(ValueError, match="finite numeric"):
        apply_shadow_conflict_policy(signals, policy="strongest_abs_pred")

    signals = _signals()
    signals.loc[0, "predicted_bps"] = -12.0

    with pytest.raises(ValueError, match="sign is inconsistent"):
        apply_shadow_conflict_policy(signals, policy="strongest_abs_pred")


def test_empty_input_keeps_diagnostic_schema():
    signals = _signals().iloc[0:0]

    resolved, summary = apply_shadow_conflict_policy(signals, policy="strongest_abs_pred")

    assert summary.input_rows == 0
    assert resolved.empty
    assert "conflict_group" in resolved.columns
    assert "conflict_policy" in resolved.columns


def test_net_pred_reports_zero_and_threshold_discards():
    signals = pd.DataFrame(
        [
            {"symbol": "XAUUSD", "time": pd.Timestamp("2026-06-01", tz="UTC"), "direction": "LONG", "predicted_bps": 12.0},
            {"symbol": "XAUUSD", "time": pd.Timestamp("2026-06-01", tz="UTC"), "direction": "SHORT", "predicted_bps": -12.0},
            {"symbol": "XAGUSD", "time": pd.Timestamp("2026-06-01", tz="UTC"), "direction": "LONG", "predicted_bps": 7.0},
        ]
    )

    resolved, summary = apply_shadow_conflict_policy(signals, policy="net_pred", threshold_bps=10.0)

    assert resolved.empty
    assert summary.discarded_groups == 2
    assert summary.zero_net_discarded_groups == 1
    assert summary.threshold_discarded_groups == 1


def test_rejects_negative_threshold():
    with pytest.raises(ValueError, match="threshold_bps must be non-negative"):
        apply_shadow_conflict_policy(_signals(), policy="net_pred", threshold_bps=-1.0)
