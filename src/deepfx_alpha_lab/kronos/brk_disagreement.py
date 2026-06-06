"""BRK × Kronos disagreement research study.

This module is research-only. It joins live BRK entries to the latest PRIOR
same-symbol Kronos shadow signal and summarizes whether aligned/opposed/no
recent Kronos states explain live trade outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from deepfx_alpha_lab.kronos.shadow_conflicts import ConflictPolicy, apply_shadow_conflict_policy

Alignment = Literal["aligned", "opposed", "no_recent_kronos"]


@dataclass(frozen=True)
class BrkDisagreementConfig:
    """Configuration for BRK × Kronos disagreement studies."""

    windows_minutes: tuple[int, ...] = (15, 30, 60, 120, 180, 240)
    primary_policy: ConflictPolicy = "strongest_abs_pred"
    brk_prefix: str = "BRK"
    start: str | None = None
    end: str | None = None
    placebo_iterations: int = 500
    placebo_window_minutes: int = 120
    random_seed: int = 7


def load_kronos_signals(path: Path, config: BrkDisagreementConfig) -> pd.DataFrame:
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
    for col in ("horizon", "lookback"):
        if col not in frame.columns:
            frame[col] = np.nan
    for col in ("model", "tag"):
        if col not in frame.columns:
            frame[col] = ""
    frame = _filter_time(frame, "time", config.start, config.end)
    resolved, _summary = apply_shadow_conflict_policy(frame, policy=config.primary_policy)
    return resolved.sort_values(["symbol", "time"]).reset_index(drop=True)


def load_live_trades(path: Path, config: BrkDisagreementConfig) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = _rename_first_present(frame, ("entry_time", "open_time", "ts", "timestamp", "time"), "entry_time")
    frame = _rename_first_present(frame, ("side", "direction"), "side")
    frame = _rename_first_present(frame, ("pnl_bps", "profit_bps", "pnl", "profit"), "pnl")
    _require_columns(frame, {"entry_time", "symbol", "side", "pnl"}, "trades")
    frame = frame.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    if "exit_time" in frame.columns:
        frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["side"] = frame["side"].map(_normalize_direction)
    if frame["side"].isna().any():
        raise ValueError("trades contain unsupported side values")
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce")
    if "reason" not in frame.columns:
        frame["reason"] = ""
    if "family" not in frame.columns:
        frame["family"] = frame["reason"].fillna("").astype(str).str.extract(r"^([A-Za-z0-9_]+)", expand=False).fillna("")
    for col in ("entry_price", "exit_price", "stop_loss", "take_profit"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = _filter_time(frame, "entry_time", config.start, config.end)
    return frame.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def load_ohlcv(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = _rename_first_present(frame, ("ts", "timestamp", "time"), "time")
    _require_columns(frame, {"time", "symbol", "close"}, "ohlcv")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    for col in ("open", "high", "low", "close"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "open" not in frame.columns:
        frame["open"] = frame["close"]
    if "high" not in frame.columns:
        frame["high"] = frame[["open", "close"]].max(axis=1)
    if "low" not in frame.columns:
        frame["low"] = frame[["open", "close"]].min(axis=1)
    return add_ohlcv_features(frame.sort_values(["symbol", "time"]).reset_index(drop=True))


def filter_brk_trades(trades: pd.DataFrame, brk_prefix: str = "BRK") -> pd.DataFrame:
    reason = trades.get("reason", pd.Series(index=trades.index, dtype=object)).fillna("").astype(str)
    family = trades.get("family", pd.Series(index=trades.index, dtype=object)).fillna("").astype(str)
    mask = family.str.upper().str.startswith(brk_prefix.upper()) | reason.str.upper().str.startswith(brk_prefix.upper())
    return trades.loc[mask].copy().reset_index(drop=True)


def attach_latest_prior_signals(signals: pd.DataFrame, trades: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    signal_cols = ["time", "symbol", "direction", "predicted_bps", "horizon", "lookback", "model", "tag"]
    available = signals[[c for c in signal_cols if c in signals.columns]].copy() if not signals.empty else pd.DataFrame(columns=signal_cols)
    tolerance = pd.Timedelta(minutes=window_minutes)
    for trade in trades.itertuples(index=False):
        sym_sig = available[available["symbol"] == trade.symbol]
        sym_sig = sym_sig[(sym_sig["time"] <= trade.entry_time) & (sym_sig["time"] >= trade.entry_time - tolerance)]
        if sym_sig.empty:
            rows.append(_alignment_row(trade, None, "no_recent_kronos", window_minutes))
            continue
        signal = sym_sig.sort_values("time").iloc[-1]
        alignment: Alignment = "aligned" if signal["direction"] == trade.side else "opposed"
        rows.append(_alignment_row(trade, signal, alignment, window_minutes))
    return pd.DataFrame(rows)


def attach_trade_features(aligned: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    if aligned.empty:
        return aligned
    out = aligned.copy()
    out["r_multiple"] = _compute_r_multiple(out)
    if ohlcv.empty:
        for col in ("atr", "atr_pct", "bb_width", "bb_width_pct", "body_atr", "close_location", "compression_expansion", "risk_regime"):
            out[col] = np.nan if col != "risk_regime" else "unknown"
        out["atr_norm_return"] = np.nan
        return out

    feature_rows = []
    for trade in out.itertuples(index=False):
        bar = _last_bar(ohlcv, trade.symbol, trade.entry_time)
        features = {"atr": np.nan, "atr_pct": np.nan, "bb_width": np.nan, "bb_width_pct": np.nan, "body_atr": np.nan, "close_location": np.nan}
        if bar is not None:
            for col in features:
                features[col] = float(bar[col]) if pd.notna(bar[col]) else np.nan
        features["compression_expansion"] = bool(
            pd.notna(features["bb_width_pct"])
            and pd.notna(features["body_atr"])
            and features["bb_width_pct"] <= 0.35
            and features["body_atr"] >= 0.75
        )
        features["risk_regime"] = classify_cross_asset_regime(ohlcv, trade.entry_time)
        feature_rows.append(features)
    feat = pd.DataFrame(feature_rows, index=out.index)
    out = pd.concat([out, feat], axis=1)
    out["atr_norm_return"] = _compute_atr_norm_return(out)
    return out


def summarize_alignment(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_cols + ["trades", "pnl_sum", "pnl_mean", "pnl_median", "win_rate", "r_mean", "atr_norm_mean"])

    def _agg(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "trades": int(len(group)),
                "pnl_sum": float(group["pnl"].sum()),
                "pnl_mean": float(group["pnl"].mean()),
                "pnl_median": float(group["pnl"].median()),
                "win_rate": float((group["pnl"] > 0).mean()),
                "r_mean": float(group["r_multiple"].mean()) if "r_multiple" in group else np.nan,
                "atr_norm_mean": float(group["atr_norm_return"].mean()) if "atr_norm_return" in group else np.nan,
            }
        )

    return frame.groupby(group_cols, dropna=False, sort=True).apply(_agg, include_groups=False).reset_index()


def add_ohlcv_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _symbol, group in ohlcv.groupby("symbol", sort=False):
        g = group.sort_values("time").copy()
        prev_close = g["close"].shift(1)
        tr = pd.concat(
            [(g["high"] - g["low"]).abs(), (g["high"] - prev_close).abs(), (g["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        g["atr"] = tr.rolling(14, min_periods=3).mean()
        ma20 = g["close"].rolling(20, min_periods=5).mean()
        std20 = g["close"].rolling(20, min_periods=5).std()
        g["bb_width"] = (4.0 * std20 / ma20).replace([np.inf, -np.inf], np.nan)
        g["body_atr"] = ((g["close"] - g["open"]).abs() / g["atr"]).replace([np.inf, -np.inf], np.nan)
        range_ = (g["high"] - g["low"]).replace(0, np.nan)
        g["close_location"] = ((g["close"] - g["low"]) / range_).clip(0, 1)
        g["atr_pct"] = _rolling_percentile(g["atr"], 288)
        g["bb_width_pct"] = _rolling_percentile(g["bb_width"], 288)
        g["ret_1h"] = g["close"].pct_change(12)
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else ohlcv


def classify_cross_asset_regime(ohlcv: pd.DataFrame, when: pd.Timestamp) -> str:
    returns: dict[str, float] = {}
    for symbol in sorted(ohlcv["symbol"].unique()):
        bar = _last_bar(ohlcv, symbol, when)
        if bar is not None and pd.notna(bar.get("ret_1h", np.nan)):
            returns[symbol] = float(bar["ret_1h"])
    equity = [returns[s] for s in ("NAS100", "US30", "SPX500", "US500") if s in returns]
    metals = [returns[s] for s in ("XAUUSD", "XAGUSD") if s in returns]
    if not metals:
        return "unknown"
    metal_ret = float(np.nanmean(metals))
    if not equity:
        if metal_ret > 0.001:
            return "metals_up_partial"
        if metal_ret < -0.001:
            return "metals_down_partial"
        return "mixed_partial"
    equity_ret = float(np.nanmean(equity))
    if equity_ret < -0.001 and metal_ret > 0.001:
        return "classic_risk_off"
    if equity_ret < -0.001 and metal_ret < -0.001:
        return "liquidity_selloff"
    if equity_ret > 0.001 and metal_ret >= -0.001:
        return "risk_on"
    return "mixed"


def run_placebo_tests(signals: pd.DataFrame, brk_trades: pd.DataFrame, window_minutes: int, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observed = attach_latest_prior_signals(signals, brk_trades, window_minutes)
    obs_stats = _placebo_stat(observed)
    rows = [
        {"test": "observed", **obs_stats},
        {"test": "time_shift_plus_1d", **_placebo_stat(attach_latest_prior_signals(signals.assign(time=signals["time"] + pd.Timedelta(days=1)), brk_trades, window_minutes))},
        {"test": "time_shift_minus_1d", **_placebo_stat(attach_latest_prior_signals(signals.assign(time=signals["time"] - pd.Timedelta(days=1)), brk_trades, window_minutes))},
    ]
    null_opposed_means: list[float] = []
    for i in range(iterations):
        shuffled_dir = signals.copy()
        shuffled_dir["direction"] = _shuffle_within_groups(shuffled_dir, ["symbol", shuffled_dir["time"].dt.date], "direction", rng)
        stat = _placebo_stat(attach_latest_prior_signals(shuffled_dir, brk_trades, window_minutes))
        null_opposed_means.append(stat["opposed_pnl_mean"])
        if i == 0:
            rows.append({"test": "direction_shuffle_example", **stat})

        shuffled_asset = signals.copy()
        shuffled_asset["symbol"] = rng.permutation(shuffled_asset["symbol"].to_numpy())
        stat = _placebo_stat(attach_latest_prior_signals(shuffled_asset, brk_trades, window_minutes))
        if i == 0:
            rows.append({"test": "asset_shuffle_example", **stat})
    obs_opposed = obs_stats["opposed_pnl_mean"]
    if null_opposed_means and pd.notna(obs_opposed):
        p_value = float((np.array(null_opposed_means) >= obs_opposed).mean())
    else:
        p_value = np.nan
    rows.append({"test": "direction_shuffle_null", "opposed_pnl_mean": float(np.nanmean(null_opposed_means)) if null_opposed_means else np.nan, "opposed_pnl_mean_p_ge_observed": p_value})
    return pd.DataFrame(rows)


def build_disagreement_study(
    signals_path: Path,
    trades_path: Path,
    ohlcv_path: Path | None,
    config: BrkDisagreementConfig,
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame | str]:
    signals = load_kronos_signals(signals_path, config)
    trades = load_live_trades(trades_path, config)
    brk_trades = filter_brk_trades(trades, config.brk_prefix)
    ohlcv = load_ohlcv(ohlcv_path)

    aligned_frames = []
    for window in config.windows_minutes:
        aligned = attach_latest_prior_signals(signals, brk_trades, window)
        aligned = attach_trade_features(aligned, ohlcv)
        aligned_frames.append(aligned)
    aligned_all = pd.concat(aligned_frames, ignore_index=True) if aligned_frames else pd.DataFrame()

    window_summary = summarize_alignment(aligned_all, ["window_minutes", "alignment"])
    symbol_side_summary = summarize_alignment(aligned_all, ["window_minutes", "symbol", "side", "alignment"])
    vol_summary = summarize_alignment(aligned_all, ["window_minutes", "alignment", "compression_expansion"])
    regime_summary = summarize_alignment(aligned_all, ["window_minutes", "alignment", "risk_regime"])
    daily_summary = summarize_alignment(aligned_all.assign(entry_day=aligned_all["entry_time"].dt.date), ["window_minutes", "entry_day", "alignment"]) if not aligned_all.empty else pd.DataFrame()
    placebo = run_placebo_tests(signals, brk_trades, config.placebo_window_minutes, config.placebo_iterations, config.random_seed)

    report = render_disagreement_report(
        config=config,
        signals=signals,
        trades=trades,
        brk_trades=brk_trades,
        aligned_all=aligned_all,
        window_summary=window_summary,
        symbol_side_summary=symbol_side_summary,
        vol_summary=vol_summary,
        regime_summary=regime_summary,
        placebo=placebo,
    )
    outputs: dict[str, pd.DataFrame | str] = {
        "aligned_trades": aligned_all,
        "window_summary": window_summary,
        "symbol_side_summary": symbol_side_summary,
        "vol_summary": vol_summary,
        "regime_summary": regime_summary,
        "daily_summary": daily_summary,
        "placebo": placebo,
        "report": report,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, value in outputs.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / "report.md").write_text(report, encoding="utf-8")
    return outputs


def render_disagreement_report(**payload: object) -> str:
    config = payload["config"]
    assert isinstance(config, BrkDisagreementConfig)
    signals = payload["signals"]
    trades = payload["trades"]
    brk_trades = payload["brk_trades"]
    aligned_all = payload["aligned_all"]
    window_summary = payload["window_summary"]
    symbol_side_summary = payload["symbol_side_summary"]
    vol_summary = payload["vol_summary"]
    regime_summary = payload["regime_summary"]
    placebo = payload["placebo"]
    assert all(isinstance(x, pd.DataFrame) for x in [signals, trades, brk_trades, aligned_all, window_summary, symbol_side_summary, vol_summary, regime_summary, placebo])

    lines = [
        "# BRK × Kronos Disagreement Study",
        "",
        "Research-only. No live trading action is taken.",
        "",
        "## Inputs",
        "",
        f"- Start: {config.start or 'input-min'}",
        f"- End: {config.end or 'input-max'}",
        f"- Conflict policy: {config.primary_policy}",
        f"- Windows minutes: {', '.join(map(str, config.windows_minutes))}",
        f"- Signals: {len(signals)}",
        f"- Trades: {len(trades)}",
        f"- BRK trades: {len(brk_trades)}",
        "",
        "## 1. Latest PRIOR Kronos window sweep",
        "",
        *_render_frame(window_summary, max_rows=60),
        "",
        "## 2. Risk-normalized PnL",
        "",
        "- `r_mean` uses entry/stop/exit prices when available; otherwise NaN.",
        "- `atr_norm_mean` uses directional price move divided by M5 ATR when OHLCV has full OHLC columns.",
        "- Account PnL units are the `pnl` column from the trade export, not true bps unless the source says so.",
        "",
        "## 3. Metals direction asymmetry / symbol-side split",
        "",
        *_render_frame(symbol_side_summary, max_rows=80),
        "",
        "## 4. Volatility compression → expansion split",
        "",
        *_render_frame(vol_summary, max_rows=80),
        "",
        "## 5. Cross-asset / partial risk-regime split",
        "",
        *_render_frame(regime_summary, max_rows=80),
        "",
        "## 6. Placebo / leakage checks",
        "",
        *_render_frame(placebo, max_rows=80),
        "",
        "## Notes",
        "",
        "- Signal join is PRIOR-only: signal_time <= entry_time.",
        "- The effect is a hypothesis until repeated walk-forward windows confirm it.",
        "- With small weekly BRK samples, counts matter more than pretty averages.",
    ]
    return "\n".join(lines)


def _alignment_row(trade: object, signal: pd.Series | None, alignment: Alignment, window_minutes: int) -> dict[str, object]:
    row = trade._asdict() if hasattr(trade, "_asdict") else dict(trade)
    row["window_minutes"] = window_minutes
    row["alignment"] = alignment
    if signal is None:
        row.update({"signal_time": pd.NaT, "signal_direction": "", "signal_predicted_bps": np.nan, "signal_age_minutes": np.nan})
    else:
        row.update(
            {
                "signal_time": signal["time"],
                "signal_direction": signal["direction"],
                "signal_predicted_bps": float(signal["predicted_bps"]),
                "signal_age_minutes": float((row["entry_time"] - signal["time"]).total_seconds() / 60.0),
            }
        )
    return row


def _compute_r_multiple(frame: pd.DataFrame) -> pd.Series:
    required = {"entry_price", "exit_price", "stop_loss", "side"}
    if not required <= set(frame.columns):
        return pd.Series(np.nan, index=frame.index)
    risk = (frame["entry_price"] - frame["stop_loss"]).abs().replace(0, np.nan)
    signed_move = np.where(frame["side"] == "LONG", frame["exit_price"] - frame["entry_price"], frame["entry_price"] - frame["exit_price"])
    return pd.Series(signed_move / risk, index=frame.index).replace([np.inf, -np.inf], np.nan)


def _compute_atr_norm_return(frame: pd.DataFrame) -> pd.Series:
    required = {"entry_price", "exit_price", "side", "atr"}
    if not required <= set(frame.columns):
        return pd.Series(np.nan, index=frame.index)
    signed_move = np.where(frame["side"] == "LONG", frame["exit_price"] - frame["entry_price"], frame["entry_price"] - frame["exit_price"])
    return pd.Series(signed_move / frame["atr"].replace(0, np.nan), index=frame.index).replace([np.inf, -np.inf], np.nan)


def _last_bar(ohlcv: pd.DataFrame, symbol: str, when: pd.Timestamp) -> pd.Series | None:
    rows = ohlcv[(ohlcv["symbol"] == symbol) & (ohlcv["time"] <= when)]
    if rows.empty:
        return None
    return rows.iloc[-1]


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def pct(values: np.ndarray) -> float:
        current = values[-1]
        if np.isnan(current):
            return np.nan
        valid = values[~np.isnan(values)]
        if len(valid) <= 1:
            return np.nan
        return float((valid <= current).mean())

    return series.rolling(window, min_periods=max(20, window // 5)).apply(pct, raw=True)


def _placebo_stat(aligned: pd.DataFrame) -> dict[str, float | int]:
    out: dict[str, float | int] = {"trades": int(len(aligned))}
    for name in ("aligned", "opposed", "no_recent_kronos"):
        group = aligned[aligned["alignment"] == name] if not aligned.empty else pd.DataFrame()
        out[f"{name}_trades"] = int(len(group))
        out[f"{name}_pnl_sum"] = float(group["pnl"].sum()) if not group.empty else 0.0
        out[f"{name}_pnl_mean"] = float(group["pnl"].mean()) if not group.empty else np.nan
    return out


def _shuffle_within_groups(frame: pd.DataFrame, keys: list[object], column: str, rng: np.random.Generator) -> pd.Series:
    helper = pd.DataFrame({"value": frame[column].to_numpy()}, index=frame.index)
    for i, key in enumerate(keys):
        helper[f"key_{i}"] = key if isinstance(key, pd.Series) else frame[key].to_numpy()
    out = helper["value"].copy()
    for _, idx in helper.groupby([f"key_{i}" for i in range(len(keys))]).groups.items():
        idx_list = list(idx)
        out.loc[idx_list] = rng.permutation(out.loc[idx_list].to_numpy())
    return out


def _render_frame(frame: pd.DataFrame, max_rows: int = 30) -> list[str]:
    if frame.empty:
        return ["- No rows."]
    shown = frame.head(max_rows)
    lines = []
    for row in shown.to_dict(orient="records"):
        label_parts = []
        for key in ("window_minutes", "alignment", "symbol", "side", "compression_expansion", "risk_regime", "test"):
            if key in row:
                label_parts.append(f"{key}={row[key]}")
        label = ", ".join(label_parts) or "row"
        lines.append(f"- {label}:")
        for key, value in row.items():
            if key in {"window_minutes", "alignment", "symbol", "side", "compression_expansion", "risk_regime", "test"}:
                continue
            lines.append(f"  - {key}: {_format_value(value)}")
    if len(frame) > max_rows:
        lines.append(f"- ... truncated {len(frame) - max_rows} rows")
    return lines


def _format_value(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


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


def _filter_time(frame: pd.DataFrame, column: str, start: str | None, end: str | None) -> pd.DataFrame:
    out = frame
    if start:
        out = out[out[column] >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out[column] < pd.Timestamp(end, tz="UTC")]
    return out


def _normalize_direction(value: object) -> str | None:
    raw = str(value).strip().upper()
    if raw in {"LONG", "BUY", "1"}:
        return "LONG"
    if raw in {"SHORT", "SELL", "-1"}:
        return "SHORT"
    return None
