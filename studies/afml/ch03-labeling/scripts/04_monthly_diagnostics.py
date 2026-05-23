#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfx_alpha_lab.data.clickhouse import load_ohlcv
from deepfx_alpha_lab.labeling import (
    add_vertical_barrier,
    get_bins,
    get_daily_vol,
    get_events,
    symmetric_cusum_filter,
)

SCRIPT_DIR = Path(__file__).resolve().parent
META_SCRIPT = SCRIPT_DIR / "03_ma_crossover_meta_labeling.py"
spec = importlib.util.spec_from_file_location("ma_crossover_meta_labeling", META_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load helper script: {META_SCRIPT}")
meta_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta_helpers)

compute_metrics = meta_helpers.compute_metrics
make_features = meta_helpers.make_features
make_primary_signal = meta_helpers.make_primary_signal
split_time_ordered = meta_helpers.split_time_ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monthly AFML Ch03 diagnostics.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--lookahead-days", type=float, default=1.0)
    parser.add_argument("--vol-span", type=int, default=100)
    parser.add_argument("--min-ret", type=float, default=0.0)
    parser.add_argument("--min-daily-bars", type=int, default=60)
    parser.add_argument("--primary", choices=["ema", "bb-reversion"], default="ema")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--bb-window", type=int, default=20)
    parser.add_argument("--bb-num-std", type=float, default=2.0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "afml" / "ch03",
    )
    return parser.parse_args()


def distribution(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def ratio(series: pd.Series, value: object) -> float | None:
    if len(series) == 0:
        return None
    return float((series == value).mean())


def filter_sparse_days(frame: pd.DataFrame, min_daily_bars: int) -> pd.DataFrame:
    if min_daily_bars <= 0 or frame.empty:
        return frame
    session = pd.Series(frame.index.normalize(), index=frame.index)
    counts = session.map(session.value_counts())
    return frame.loc[counts >= min_daily_bars]


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    starts = pd.date_range(start=start, end=end, freq="MS")
    return [ts for ts in starts if ts < end]


def first_existing(mapping: dict[str, int], key: str) -> int:
    return int(mapping.get(key, 0))


def run_triple_barrier_month(
    close_context: pd.Series,
    daily_vol_context: pd.Series,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    close_month = close_context.loc[(close_context.index >= month_start) & (close_context.index < month_end)]
    daily_vol_month = daily_vol_context.loc[(daily_vol_context.index >= month_start) & (daily_vol_context.index < month_end)]
    if close_month.empty or daily_vol_month.empty:
        return pd.DataFrame(), {"error": "empty close or daily volatility"}

    threshold = float(daily_vol_month.mean())
    t_events = symmetric_cusum_filter(close_month, threshold=threshold)
    t1 = add_vertical_barrier(t_events, close_context, num_days=args.lookahead_days)
    events = get_events(
        close=close_context,
        t_events=t_events,
        pt_sl=[1.0, 1.0],
        trgt=daily_vol_context,
        min_ret=args.min_ret,
        t1=t1,
    )
    bins = get_bins(close_context, events)
    labels = events.join(bins[["ret", "bin"]], how="inner")
    labels.index.name = "t0"
    holding = ((bins["t1"] - bins.index).dt.total_seconds() / 60.0) if not bins.empty else pd.Series(dtype=float)

    label_dist = distribution(labels["bin"]) if not labels.empty else {}
    barrier_dist = distribution(labels["type"]) if not labels.empty else {}
    summary = {
        "daily_vol_mean": threshold,
        "cusum_events": int(len(t_events)),
        "labeled_events_31": int(len(labels)),
        "label_neg_31": first_existing(label_dist, "-1"),
        "label_pos_31": first_existing(label_dist, "1"),
        "pt_31": first_existing(barrier_dist, "pt"),
        "sl_31": first_existing(barrier_dist, "sl"),
        "t1_31": first_existing(barrier_dist, "t1"),
        "pos_rate_31": ratio(labels["bin"], 1) if not labels.empty else None,
        "t1_rate_31": ratio(labels["type"], "t1") if not labels.empty else None,
        "median_holding_minutes_31": float(holding.median()) if not holding.empty else None,
        "avg_holding_minutes_31": float(holding.mean()) if not holding.empty else None,
    }
    return labels, summary


def run_meta_month(
    bars_context: pd.DataFrame,
    close_context: pd.Series,
    daily_vol_context: pd.Series,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    close_month = close_context.loc[(close_context.index >= month_start) & (close_context.index < month_end)]
    daily_vol_month = daily_vol_context.loc[(daily_vol_context.index >= month_start) & (daily_vol_context.index < month_end)]
    if close_month.empty or daily_vol_month.empty:
        return pd.DataFrame(), {"error": "empty close or daily volatility"}

    threshold = float(daily_vol_month.mean())
    t_events = symmetric_cusum_filter(close_month, threshold=threshold)
    t1 = add_vertical_barrier(t_events, close_context, num_days=args.lookahead_days)
    primary = make_primary_signal(
        close_context,
        primary=args.primary,
        fast=args.fast,
        slow=args.slow,
        bb_window=args.bb_window,
        bb_num_std=args.bb_num_std,
    )
    side = primary["side"].reindex(t_events).dropna()
    events = get_events(
        close=close_context,
        t_events=pd.DatetimeIndex(side.index),
        pt_sl=[1.0, 2.0],
        trgt=daily_vol_context,
        min_ret=args.min_ret,
        t1=t1,
        side=side,
    )
    bins = get_bins(close_context, events)
    labels = events.join(bins[["ret", "bin"]], how="inner")
    labels.index.name = "t0"
    if labels.empty:
        return labels, {"error": "empty meta labels"}

    features = make_features(bars_context, daily_vol_context, primary)
    feature_cols = list(features.columns)
    dataset = features.reindex(labels.index).join(labels[["t1", "trgt", "type", "ret", "bin"]])
    dataset = dataset.dropna(subset=feature_cols + ["bin"]).copy()
    dataset["bin"] = dataset["bin"].astype(int)
    label_dist = distribution(dataset["bin"]) if not dataset.empty else {}
    barrier_dist = distribution(dataset["type"]) if not dataset.empty else {}
    side_dist = distribution(dataset["side"]) if not dataset.empty else {}

    summary: dict[str, object] = {
        "meta_events_34": int(len(labels)),
        "dataset_rows_34": int(len(dataset)),
        "meta_label_0_34": first_existing(label_dist, "0"),
        "meta_label_1_34": first_existing(label_dist, "1"),
        "meta_pos_rate_34": ratio(dataset["bin"], 1) if not dataset.empty else None,
        "side_short_34": first_existing(side_dist, "-1"),
        "side_long_34": first_existing(side_dist, "1"),
        "pt_34": first_existing(barrier_dist, "pt"),
        "sl_34": first_existing(barrier_dist, "sl"),
        "t1_34": first_existing(barrier_dist, "t1"),
    }

    if len(dataset) < 50 or dataset["bin"].nunique() < 2:
        summary["model_error_34"] = "not enough rows or classes"
        return dataset, summary

    train, test = split_time_ordered(dataset, args.train_frac)
    if train["bin"].nunique() < 2 or test["bin"].nunique() < 2:
        summary["model_error_34"] = "train or test has fewer than two classes"
        return dataset, summary

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(train[feature_cols], train["bin"])
    pred = model.predict(test[feature_cols])
    proba = model.predict_proba(test[feature_cols])[:, list(model.classes_).index(1)] if 1 in model.classes_ else None
    majority_class = int(train["bin"].mode().iloc[0])
    majority_pred = np.full(len(test), majority_class)
    metrics = compute_metrics(test["bin"], pred, proba)
    majority = compute_metrics(test["bin"], majority_pred, None)

    summary.update(
        {
            "train_rows_34": int(len(train)),
            "test_rows_34": int(len(test)),
            "rf_accuracy_34": metrics.get("accuracy"),
            "rf_f1_34": metrics.get("f1"),
            "rf_roc_auc_34": metrics.get("roc_auc"),
            "majority_accuracy_34": majority.get("accuracy"),
            "accuracy_edge_34": float(metrics["accuracy"] - majority["accuracy"]),
        }
    )
    return dataset, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    load_start = start - pd.Timedelta(days=args.warmup_days)
    load_end = end + pd.Timedelta(days=args.lookahead_days)

    bars = load_ohlcv(args.symbol, args.timeframe, start=str(load_start), end=str(load_end))
    if bars.empty:
        raise RuntimeError(f"No bars found for {args.symbol} {args.timeframe}")

    bars = bars.loc[(bars["close"] > 0) & (bars["high"] >= bars["low"])]
    bars = filter_sparse_days(bars, args.min_daily_bars)
    close = bars["close"].dropna()
    daily_vol = get_daily_vol(close, span=args.vol_span).dropna()
    if close.empty or daily_vol.empty:
        raise RuntimeError("No usable close or daily volatility after filtering")

    rows: list[dict[str, object]] = []
    for month_start in month_starts(start, end):
        month_end = min(month_start + pd.DateOffset(months=1), end)
        month_key = month_start.strftime("%Y-%m")
        bars_context = bars.loc[
            (bars.index >= month_start - pd.Timedelta(days=args.warmup_days))
            & (bars.index < month_end + pd.Timedelta(days=args.lookahead_days))
        ]
        close_context = bars_context["close"].dropna()
        daily_vol_context = daily_vol.reindex(close_context.index, method="ffill").dropna()
        rows_in_month = int(((bars.index >= month_start) & (bars.index < month_end)).sum())

        _, tb_summary = run_triple_barrier_month(close_context, daily_vol_context, month_start, month_end, args)
        _, meta_summary = run_meta_month(bars_context, close_context, daily_vol_context, month_start, month_end, args)
        rows.append(
            {
                "month": month_key,
                "start": str(month_start),
                "end": str(month_end),
                "rows": rows_in_month,
                **tb_summary,
                **meta_summary,
            }
        )

    monthly = pd.DataFrame(rows)
    primary_suffix = "" if args.primary == "ema" else f"_{args.primary.replace('-', '_')}"
    stem = f"monthly_diagnostics_{args.symbol.lower()}_{args.timeframe.lower()}{primary_suffix}_{start:%Y%m}_{(end - pd.Timedelta(days=1)):%Y%m}"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    monthly.to_csv(csv_path, index=False)
    summary = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start": args.start,
        "end": args.end,
        "warmup_days": args.warmup_days,
        "lookahead_days": args.lookahead_days,
        "primary": args.primary,
        "bb_window": args.bb_window if args.primary == "bb-reversion" else None,
        "bb_num_std": args.bb_num_std if args.primary == "bb-reversion" else None,
        "months": monthly.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    display_cols = [
        "month",
        "rows",
        "labeled_events_31",
        "pos_rate_31",
        "t1_rate_31",
        "median_holding_minutes_31",
        "dataset_rows_34",
        "meta_pos_rate_34",
        "rf_accuracy_34",
        "majority_accuracy_34",
        "accuracy_edge_34",
        "rf_roc_auc_34",
    ]
    print(monthly[display_cols].to_string(index=False))
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()
