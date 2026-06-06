import pandas as pd

from deepfx_alpha_lab.kronos.shadow_report import (
    ShadowReportConfig,
    align_trades_to_signals,
    build_shadow_report,
    load_live_trades,
    load_shadow_signals,
    summarize_brk_capture,
    summarize_direction_edge,
    summarize_trade_alignment,
)


def test_load_shadow_signals_normalizes_aliases_and_absolute_predictions(tmp_path):
    path = tmp_path / "signals.csv"
    pd.DataFrame(
        [
            {"ts": "2026-06-01T00:00:00Z", "symbol": "xauusd", "side": "BUY", "predicted_bps": 12.0},
            {"ts": "2026-06-01T00:05:00Z", "symbol": "xauusd", "side": "SELL", "predicted_bps": 15.0},
        ]
    ).to_csv(path, index=False)

    signals = load_shadow_signals(path, ShadowReportConfig(predicted_is_absolute=True))

    assert signals["symbol"].tolist() == ["XAUUSD", "XAUUSD"]
    assert signals["direction"].tolist() == ["LONG", "SHORT"]
    assert signals["predicted_bps"].tolist() == [12.0, -15.0]


def test_summarize_direction_edge_uses_direction_adjusted_realized_bps():
    signals = pd.DataFrame(
        [
            {"time": pd.Timestamp("2026-06-01", tz="UTC"), "symbol": "XAUUSD", "direction": "LONG", "predicted_bps": 12.0, "realized_bps": 10.0},
            {"time": pd.Timestamp("2026-06-01", tz="UTC"), "symbol": "XAUUSD", "direction": "SHORT", "predicted_bps": -15.0, "realized_bps": -8.0},
        ]
    )

    rows = summarize_direction_edge(signals)

    by_direction = {row["direction"]: row for row in rows}
    assert by_direction["LONG"]["mean_edge_bps"] == 10.0
    assert by_direction["SHORT"]["mean_edge_bps"] == 8.0
    assert by_direction["SHORT"]["win_rate"] == 1.0


def test_trade_alignment_and_brk_capture(tmp_path):
    signals = pd.DataFrame(
        [
            {"time": pd.Timestamp("2026-06-01T00:00:00Z"), "symbol": "XAUUSD", "direction": "LONG", "predicted_bps": 12.0},
            {"time": pd.Timestamp("2026-06-01T00:30:00Z"), "symbol": "XAGUSD", "direction": "SHORT", "predicted_bps": -20.0},
        ]
    )
    trades = pd.DataFrame(
        [
            {"trade_time": pd.Timestamp("2026-06-01T00:05:00Z"), "symbol": "XAUUSD", "trade_direction": "LONG", "pnl_bps": 5.0, "reason": "BRK breakout"},
            {"trade_time": pd.Timestamp("2026-06-01T00:35:00Z"), "symbol": "XAGUSD", "trade_direction": "LONG", "pnl_bps": -3.0, "reason": "BRK fade"},
            {"trade_time": pd.Timestamp("2026-06-01T03:00:00Z"), "symbol": "XAUUSD", "trade_direction": "LONG", "pnl_bps": 1.0, "reason": "REV"},
        ]
    )

    aligned = align_trades_to_signals(signals, trades, pd.Timedelta(minutes=15))
    summary = summarize_trade_alignment(aligned)
    brk = summarize_brk_capture(aligned, "BRK")

    assert aligned["alignment"].tolist() == ["aligned", "opposed", "no_signal"]
    assert summary["aligned_trades"] == 1
    assert summary["opposed_trades"] == 1
    assert summary["no_signal_trades"] == 1
    assert brk["brk_trades"] == 2
    assert brk["matched_brk_trades"] == 2
    assert brk["aligned_brk_trades"] == 1
    assert brk["opposed_brk_trades"] == 1


def test_build_shadow_report_markdown_smoke(tmp_path):
    signals_path = tmp_path / "signals.csv"
    trades_path = tmp_path / "trades.csv"
    pd.DataFrame(
        [
            {"time": "2026-06-01T00:00:00Z", "symbol": "XAUUSD", "direction": "LONG", "predicted_bps": 12.0, "realized_bps": 10.0},
            {"time": "2026-06-01T00:00:00Z", "symbol": "XAUUSD", "direction": "SHORT", "predicted_bps": -15.0, "realized_bps": 10.0},
            {"time": "2026-06-01T00:30:00Z", "symbol": "XAGUSD", "direction": "SHORT", "predicted_bps": -20.0, "realized_bps": -8.0},
        ]
    ).to_csv(signals_path, index=False)
    pd.DataFrame(
        [
            {"entry_time": "2026-06-01T00:33:00Z", "symbol": "XAGUSD", "side": "SELL", "pnl_bps": 8.0, "reason": "BRK breakdown"},
        ]
    ).to_csv(trades_path, index=False)

    report = build_shadow_report(signals_path, trades_path, ShadowReportConfig())

    assert "# Kronos Shadow Report" in report
    assert "## 2. Conflict Policy Comparison" in report
    assert "## 3. Primary Policy Signal Edge" in report
    assert "## 5. BRK Capture" in report
    assert "strongest_abs_pred" in report


def test_load_live_trades_normalizes_aliases(tmp_path):
    path = tmp_path / "trades.csv"
    pd.DataFrame(
        [{"entry_time": "2026-06-01T00:00:00Z", "symbol": "xauusd", "side": "SELL", "profit": -2.5}]
    ).to_csv(path, index=False)

    trades = load_live_trades(path, ShadowReportConfig())

    assert trades["symbol"].tolist() == ["XAUUSD"]
    assert trades["trade_direction"].tolist() == ["SHORT"]
    assert trades["pnl_bps"].tolist() == [-2.5]
