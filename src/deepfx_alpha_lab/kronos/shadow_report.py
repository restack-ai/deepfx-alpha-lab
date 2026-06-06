"""Research-only Kronos shadow signal markdown reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from deepfx_alpha_lab.kronos.shadow_conflicts import (
    ConflictPolicy,
    ShadowConflictSummary,
    apply_shadow_conflict_policy,
)


@dataclass(frozen=True)
class ShadowReportConfig:
    """Configuration for offline Kronos shadow reports."""

    period: str = "daily"
    start: str | None = None
    end: str | None = None
    policies: tuple[ConflictPolicy, ...] = ("discard_conflict", "strongest_abs_pred", "net_pred")
    primary_policy: ConflictPolicy = "strongest_abs_pred"
    threshold_bps: float | None = 10.0
    match_tolerance_minutes: int = 15
    brk_prefix: str = "BRK"
    predicted_is_absolute: bool = False


def load_shadow_signals(path: Path, config: ShadowReportConfig) -> pd.DataFrame:
    """Load and normalize a Kronos shadow signal CSV export."""

    frame = pd.read_csv(path)
    frame = _rename_first_present(frame, ("ts", "timestamp", "time"), "time")
    frame = _rename_first_present(frame, ("side", "direction"), "direction")
    _require_columns(frame, {"time", "symbol", "direction", "predicted_bps"}, "signals")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["direction"] = frame["direction"].map(_normalize_direction)
    if frame["direction"].isna().any():
        raise ValueError("signals contain unsupported direction values")
    frame["predicted_bps"] = pd.to_numeric(frame["predicted_bps"], errors="raise")
    if config.predicted_is_absolute:
        frame["predicted_bps"] = frame["predicted_bps"].abs() * frame["direction"].map(_side_sign)
    frame = _filter_period(frame, config)
    if "realized_bps" in frame.columns:
        frame["realized_bps"] = pd.to_numeric(frame["realized_bps"], errors="raise")
    return frame.sort_values(["time", "symbol", "direction"]).reset_index(drop=True)


def load_live_trades(path: Path, config: ShadowReportConfig) -> pd.DataFrame:
    """Load and normalize a live trade CSV export."""

    frame = pd.read_csv(path)
    frame = _rename_first_present(frame, ("entry_time", "open_time", "ts", "timestamp", "time"), "trade_time")
    frame = _rename_first_present(frame, ("side", "direction"), "trade_direction")
    frame = _rename_first_present(frame, ("pnl_bps", "profit_bps", "pnl", "profit"), "pnl_bps")
    _require_columns(frame, {"trade_time", "symbol", "trade_direction", "pnl_bps"}, "trades")
    frame = frame.copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["trade_direction"] = frame["trade_direction"].map(_normalize_direction)
    if frame["trade_direction"].isna().any():
        raise ValueError("trades contain unsupported direction values")
    frame["pnl_bps"] = pd.to_numeric(frame["pnl_bps"], errors="raise")
    if "reason" not in frame.columns:
        frame["reason"] = ""
    frame = _filter_period(frame.rename(columns={"trade_time": "time"}), config).rename(columns={"time": "trade_time"})
    return frame.sort_values(["trade_time", "symbol"]).reset_index(drop=True)


def evaluate_conflict_policies(
    signals: pd.DataFrame, config: ShadowReportConfig
) -> dict[str, tuple[pd.DataFrame, ShadowConflictSummary]]:
    """Apply every requested conflict policy to normalized signal rows."""

    results: dict[str, tuple[pd.DataFrame, ShadowConflictSummary]] = {}
    for policy in config.policies:
        resolved, summary = apply_shadow_conflict_policy(
            signals,
            policy=policy,
            threshold_bps=config.threshold_bps if policy == "net_pred" else None,
        )
        results[policy] = (resolved, summary)
    return results


def add_signal_edge(signals: pd.DataFrame) -> pd.DataFrame:
    """Add direction-adjusted edge bps when realized_bps is present."""

    frame = signals.copy()
    frame["side_sign"] = frame["direction"].map(_side_sign)
    if "realized_bps" in frame.columns:
        frame["edge_bps"] = pd.to_numeric(frame["realized_bps"], errors="raise") * frame["side_sign"]
    return frame


def summarize_policy_result(resolved: pd.DataFrame, summary: ShadowConflictSummary) -> dict[str, object]:
    with_edge = add_signal_edge(resolved)
    metrics: dict[str, object] = {
        "input_rows": summary.input_rows,
        "groups": summary.groups,
        "output_rows": summary.output_rows,
        "conflict_groups": summary.conflict_groups,
        "same_direction_duplicate_groups": summary.same_direction_duplicate_groups,
        "discarded_groups": summary.discarded_groups,
        "conflict_discarded_groups": summary.conflict_discarded_groups,
        "zero_net_discarded_groups": summary.zero_net_discarded_groups,
        "threshold_discarded_groups": summary.threshold_discarded_groups,
        "discarded_rows": summary.discarded_rows,
        "retention_rate": _safe_ratio(summary.output_rows, summary.input_rows),
        "long_rows": int((with_edge.get("direction", pd.Series(dtype=str)) == "LONG").sum()),
        "short_rows": int((with_edge.get("direction", pd.Series(dtype=str)) == "SHORT").sum()),
    }
    if "edge_bps" in with_edge.columns and not with_edge.empty:
        metrics.update(
            {
                "mean_edge_bps": float(with_edge["edge_bps"].mean()),
                "median_edge_bps": float(with_edge["edge_bps"].median()),
                "win_rate": float((with_edge["edge_bps"] > 0).mean()),
            }
        )
    return metrics


def summarize_direction_edge(signals: pd.DataFrame) -> list[dict[str, object]]:
    with_edge = add_signal_edge(signals)
    rows: list[dict[str, object]] = []
    for (symbol, direction), group in with_edge.groupby(["symbol", "direction"], sort=True):
        row: dict[str, object] = {"symbol": symbol, "direction": direction, "count": int(len(group))}
        if "edge_bps" in group.columns:
            row.update(
                {
                    "mean_edge_bps": float(group["edge_bps"].mean()),
                    "median_edge_bps": float(group["edge_bps"].median()),
                    "win_rate": float((group["edge_bps"] > 0).mean()),
                }
            )
        rows.append(row)
    return rows


def align_trades_to_signals(signals: pd.DataFrame, trades: pd.DataFrame, tolerance: pd.Timedelta) -> pd.DataFrame:
    """Match each trade to the nearest same-symbol shadow signal within tolerance."""

    if trades.empty:
        return trades.copy().assign(alignment=pd.Series(dtype="object"), signal_time=pd.Series(dtype="datetime64[ns, UTC]"))
    signal_cols = ["time", "symbol", "direction", "predicted_bps"]
    available_signals = signals[signal_cols].sort_values("time") if not signals.empty else signals.iloc[0:0]
    matches: list[dict[str, object]] = []
    for trade in trades.itertuples(index=False):
        symbol_signals = available_signals[available_signals["symbol"] == trade.symbol]
        if symbol_signals.empty:
            matches.append(_trade_match_row(trade, None, "no_signal"))
            continue
        deltas = (symbol_signals["time"] - trade.trade_time).abs()
        nearest_idx = deltas.idxmin()
        if deltas.loc[nearest_idx] > tolerance:
            matches.append(_trade_match_row(trade, None, "no_signal"))
            continue
        signal = symbol_signals.loc[nearest_idx]
        alignment = "aligned" if signal["direction"] == trade.trade_direction else "opposed"
        matches.append(_trade_match_row(trade, signal, alignment))
    return pd.DataFrame(matches)


def summarize_trade_alignment(aligned: pd.DataFrame) -> dict[str, object]:
    if aligned.empty:
        return {"matched_trades": 0, "aligned_trades": 0, "opposed_trades": 0, "no_signal_trades": 0}
    out: dict[str, object] = {}
    out["matched_trades"] = int((aligned["alignment"] != "no_signal").sum())
    for name in ("aligned", "opposed", "no_signal"):
        group = aligned[aligned["alignment"] == name]
        out[f"{name}_trades"] = int(len(group))
        out[f"{name}_pnl_bps"] = float(group["pnl_bps"].sum()) if not group.empty else 0.0
        out[f"{name}_mean_pnl_bps"] = float(group["pnl_bps"].mean()) if not group.empty else 0.0
    return out


def summarize_brk_capture(aligned: pd.DataFrame, brk_prefix: str) -> dict[str, object]:
    if aligned.empty or "reason" not in aligned.columns:
        return {"brk_trades": 0, "matched_brk_trades": 0, "brk_capture_rate": 0.0}
    brk = aligned[aligned["reason"].fillna("").astype(str).str.startswith(brk_prefix)]
    matched = brk[brk["alignment"] != "no_signal"]
    out = {
        "brk_trades": int(len(brk)),
        "matched_brk_trades": int(len(matched)),
        "brk_capture_rate": _safe_ratio(len(matched), len(brk)),
        "aligned_brk_trades": int((brk["alignment"] == "aligned").sum()),
        "opposed_brk_trades": int((brk["alignment"] == "opposed").sum()),
        "missed_brk_trades": int((brk["alignment"] == "no_signal").sum()),
        "aligned_brk_pnl_bps": float(brk.loc[brk["alignment"] == "aligned", "pnl_bps"].sum()),
        "opposed_brk_pnl_bps": float(brk.loc[brk["alignment"] == "opposed", "pnl_bps"].sum()),
    }
    return out


def build_shadow_report(signals_path: Path, trades_path: Path | None, config: ShadowReportConfig) -> str:
    signals = load_shadow_signals(signals_path, config)
    results = evaluate_conflict_policies(signals, config)
    if config.primary_policy not in results:
        raise ValueError(f"primary_policy not evaluated: {config.primary_policy}")
    primary_signals, _ = results[config.primary_policy]
    trades = load_live_trades(trades_path, config) if trades_path is not None else pd.DataFrame()
    aligned = align_trades_to_signals(
        primary_signals,
        trades,
        tolerance=pd.Timedelta(minutes=config.match_tolerance_minutes),
    )
    payload = {
        "config": config,
        "signals_path": str(signals_path),
        "trades_path": str(trades_path) if trades_path else "",
        "raw_signals": signals,
        "policy_metrics": {policy: summarize_policy_result(*result) for policy, result in results.items()},
        "primary_policy": config.primary_policy,
        "primary_edge": summarize_direction_edge(primary_signals),
        "trade_alignment": summarize_trade_alignment(aligned),
        "brk_capture": summarize_brk_capture(aligned, config.brk_prefix),
        "has_realized_bps": "realized_bps" in signals.columns,
    }
    return render_markdown_report(payload)


def render_markdown_report(payload: dict[str, object]) -> str:
    config = payload["config"]
    assert isinstance(config, ShadowReportConfig)
    raw_signals = payload["raw_signals"]
    assert isinstance(raw_signals, pd.DataFrame)
    lines = [
        "# Kronos Shadow Report",
        "",
        f"- Period: {config.period}",
        f"- Window start: {config.start or 'input-min'}",
        f"- Window end: {config.end or 'input-max'}",
        f"- Signals CSV: {payload['signals_path']}",
        f"- Trades CSV: {payload['trades_path'] or 'not provided'}",
        f"- Primary policy: {payload['primary_policy']}",
        "",
        "## 1. Input Health",
        "",
        f"- Raw signal rows: {len(raw_signals)}",
        f"- Symbols: {', '.join(sorted(raw_signals['symbol'].unique())) if not raw_signals.empty else 'none'}",
        f"- Has realized_bps: {payload['has_realized_bps']}",
        "",
        "## 2. Conflict Policy Comparison",
        "",
    ]
    policy_metrics = payload["policy_metrics"]
    assert isinstance(policy_metrics, dict)
    for policy, metrics in policy_metrics.items():
        assert isinstance(metrics, dict)
        lines.extend(_render_metric_block(str(policy), metrics))
    lines.extend(["", "## 3. Primary Policy Signal Edge", ""])
    primary_edge = payload["primary_edge"]
    assert isinstance(primary_edge, list)
    if not primary_edge:
        lines.append("- No primary-policy signals kept.")
    for row in primary_edge:
        assert isinstance(row, dict)
        label = f"{row['symbol']} {row['direction']}"
        lines.extend(_render_metric_block(label, row))
    lines.extend(["", "## 4. Live Trade Alignment", ""])
    trade_alignment = payload["trade_alignment"]
    assert isinstance(trade_alignment, dict)
    lines.extend(_render_metric_block("alignment", trade_alignment))
    lines.extend(["", "## 5. BRK Capture", ""])
    brk_capture = payload["brk_capture"]
    assert isinstance(brk_capture, dict)
    lines.extend(_render_metric_block("brk", brk_capture))
    lines.extend(
        [
            "",
            "## 6. Notes and Limitations",
            "",
            "- Research-only report; no live trading action is taken.",
            "- Trade matching uses nearest same-symbol signal within the configured tolerance.",
            "- BRK capture MVP uses live trade reason prefix matching only.",
            "- Edge metrics require realized_bps in the signal CSV; otherwise counts/diagnostics only are shown.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_metric_block(title: str, metrics: dict[str, object]) -> list[str]:
    lines = [f"- {title}:"]
    for key, value in metrics.items():
        lines.append(f"  - {key}: {_format_value(value)}")
    return lines


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _trade_match_row(trade: object, signal: pd.Series | None, alignment: str) -> dict[str, object]:
    row = {
        "trade_time": trade.trade_time,
        "symbol": trade.symbol,
        "trade_direction": trade.trade_direction,
        "pnl_bps": float(trade.pnl_bps),
        "reason": getattr(trade, "reason", ""),
        "alignment": alignment,
    }
    if signal is None:
        row.update({"signal_time": pd.NaT, "signal_direction": "", "predicted_bps": float("nan")})
    else:
        row.update(
            {
                "signal_time": signal["time"],
                "signal_direction": signal["direction"],
                "predicted_bps": float(signal["predicted_bps"]),
            }
        )
    return row


def _rename_first_present(frame: pd.DataFrame, candidates: tuple[str, ...], target: str) -> pd.DataFrame:
    if target in frame.columns:
        return frame
    for candidate in candidates:
        if candidate in frame.columns:
            return frame.rename(columns={candidate: target})
    return frame


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} CSV missing required columns: {missing}")


def _filter_period(frame: pd.DataFrame, config: ShadowReportConfig) -> pd.DataFrame:
    out = frame
    if config.start:
        out = out[out["time"] >= pd.Timestamp(config.start, tz="UTC")]
    if config.end:
        out = out[out["time"] < pd.Timestamp(config.end, tz="UTC")]
    return out


def _normalize_direction(value: object) -> str | None:
    raw = str(value).strip().upper()
    if raw in {"LONG", "BUY", "1"}:
        return "LONG"
    if raw in {"SHORT", "SELL", "-1"}:
        return "SHORT"
    return None


def _side_sign(direction: object) -> int:
    normalized = _normalize_direction(direction)
    if normalized == "LONG":
        return 1
    if normalized == "SHORT":
        return -1
    raise ValueError(f"unsupported direction: {direction}")


def _safe_ratio(num: int | float, denom: int | float) -> float:
    return float(num / denom) if denom else 0.0
