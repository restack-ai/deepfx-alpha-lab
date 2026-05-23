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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run purged walk-forward validation for AFML Exercise 3.4.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--lookahead-days", type=float, default=1.0)
    parser.add_argument("--primary", choices=["ema", "bb-reversion"], default="ema")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--bb-window", type=int, default=20)
    parser.add_argument("--bb-num-std", type=float, default=2.0)
    parser.add_argument("--pt", type=float, default=1.0)
    parser.add_argument("--sl", type=float, default=2.0)
    parser.add_argument("--vol-span", type=int, default=100)
    parser.add_argument("--min-ret", type=float, default=0.0)
    parser.add_argument("--min-daily-bars", type=int, default=60)
    parser.add_argument("--n-estimators", type=int, default=300)
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


def filter_sparse_days(frame: pd.DataFrame, min_daily_bars: int) -> pd.DataFrame:
    if min_daily_bars <= 0 or frame.empty:
        return frame
    session = pd.Series(frame.index.normalize(), index=frame.index)
    counts = session.map(session.value_counts())
    return frame.loc[counts >= min_daily_bars]


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    starts = pd.date_range(start=start, end=end, freq="MS")
    return [ts for ts in starts if ts < end]


def build_meta_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
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

    close_eval = close.loc[(close.index >= start) & (close.index < end)]
    threshold = daily_vol.reindex(close_eval.index, method="ffill").dropna()
    t_events = symmetric_cusum_filter(close_eval.reindex(threshold.index), threshold=threshold)
    t1 = add_vertical_barrier(t_events, close, num_days=args.lookahead_days)

    primary = make_primary_signal(
        close,
        primary=args.primary,
        fast=args.fast,
        slow=args.slow,
        bb_window=args.bb_window,
        bb_num_std=args.bb_num_std,
    )
    side = primary["side"].reindex(t_events).dropna()
    events = get_events(
        close=close,
        t_events=pd.DatetimeIndex(side.index),
        pt_sl=[args.pt, args.sl],
        trgt=daily_vol,
        min_ret=args.min_ret,
        t1=t1,
        side=side,
    )
    bins = get_bins(close, events)
    labels = events.join(bins[["ret", "bin"]], how="inner")
    labels.index.name = "t0"

    features = make_features(bars, daily_vol, primary)
    feature_cols = list(features.columns)
    dataset = features.reindex(labels.index).join(labels[["t1", "trgt", "type", "ret", "bin"]])
    dataset = dataset.loc[(dataset.index >= start) & (dataset.index < end)]
    dataset = dataset.dropna(subset=feature_cols + ["t1", "bin"]).copy()
    dataset["bin"] = dataset["bin"].astype(int)
    dataset = dataset.sort_index()

    metadata = {
        "raw_rows_after_dedup": int(len(bars)),
        "dataset_rows": int(len(dataset)),
        "cusum_events": int(len(t_events)),
        "meta_label_events": int(len(labels)),
        "label_distribution": distribution(dataset["bin"]),
        "barrier_distribution": distribution(dataset["type"]),
        "side_distribution": distribution(dataset["side"]),
    }
    return dataset, feature_cols, metadata


def run_fold(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    train_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    raw_train = dataset.loc[(dataset.index >= train_start) & (dataset.index < test_start)]
    train = raw_train.loc[raw_train["t1"] < test_start]
    test = dataset.loc[(dataset.index >= test_start) & (dataset.index < test_end)]

    fold: dict[str, object] = {
        "fold": f"{train_start:%Y-%m}_to_{(test_start - pd.Timedelta(days=1)):%Y-%m}_test_{test_start:%Y-%m}",
        "train_start": str(train_start),
        "train_end": str(test_start),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "raw_train_rows": int(len(raw_train)),
        "purged_train_rows": int(len(train)),
        "purged_rows": int(len(raw_train) - len(train)),
        "test_rows": int(len(test)),
        "train_label_distribution": distribution(train["bin"]) if not train.empty else {},
        "test_label_distribution": distribution(test["bin"]) if not test.empty else {},
    }
    empty_predictions = pd.DataFrame()
    if len(train) < 50 or len(test) < 20 or train["bin"].nunique() < 2 or test["bin"].nunique() < 2:
        fold["error"] = "not enough rows or classes"
        return fold, empty_predictions

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
    fold.update(
        {
            "rf_accuracy": metrics.get("accuracy"),
            "rf_precision": metrics.get("precision"),
            "rf_recall": metrics.get("recall"),
            "rf_f1": metrics.get("f1"),
            "rf_roc_auc": metrics.get("roc_auc"),
            "majority_accuracy": majority.get("accuracy"),
            "majority_f1": majority.get("f1"),
            "accuracy_edge": float(metrics["accuracy"] - majority["accuracy"]),
        }
    )

    predictions = test[["t1", "trgt", "side", "type", "ret", "bin"]].copy()
    predictions["fold"] = fold["fold"]
    predictions["pred"] = pred
    if proba is not None:
        predictions["proba_1"] = proba
    return fold, predictions


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    dataset, feature_cols, metadata = build_meta_dataset(args)
    months = month_starts(start, end)
    if len(months) < 2:
        raise RuntimeError("Need at least two months for walk-forward validation")

    folds: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for test_start in months[1:]:
        test_end = min(test_start + pd.DateOffset(months=1), end)
        fold, fold_predictions = run_fold(dataset, feature_cols, start, test_start, test_end, args)
        folds.append(fold)
        if not fold_predictions.empty:
            predictions.append(fold_predictions)

    folds_frame = pd.DataFrame(folds)
    predictions_frame = pd.concat(predictions).sort_index() if predictions else pd.DataFrame()
    valid = folds_frame.dropna(subset=["rf_accuracy", "majority_accuracy"], how="any")

    aggregate = {
        "folds": int(len(folds_frame)),
        "valid_folds": int(len(valid)),
        "mean_rf_accuracy": float(valid["rf_accuracy"].mean()) if not valid.empty else None,
        "mean_majority_accuracy": float(valid["majority_accuracy"].mean()) if not valid.empty else None,
        "mean_accuracy_edge": float(valid["accuracy_edge"].mean()) if not valid.empty else None,
        "mean_rf_roc_auc": float(valid["rf_roc_auc"].mean()) if "rf_roc_auc" in valid and not valid.empty else None,
        "positive_edge_folds": int((valid["accuracy_edge"] > 0).sum()) if not valid.empty else 0,
    }

    primary_suffix = "" if args.primary == "ema" else f"_{args.primary.replace('-', '_')}"
    stem = f"walk_forward_{args.symbol.lower()}_{args.timeframe.lower()}{primary_suffix}_{start:%Y%m}_{(end - pd.Timedelta(days=1)):%Y%m}"
    folds_path = args.output_dir / f"{stem}_folds.csv"
    predictions_path = args.output_dir / f"{stem}_predictions.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"

    folds_frame.to_csv(folds_path, index=False)
    if not predictions_frame.empty:
        predictions_frame.to_csv(predictions_path)
    else:
        predictions_path.write_text("")

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
        "purge_rule": "training rows with t1 >= test_start are excluded",
        "feature_columns": feature_cols,
        "model": {
            "name": "RandomForestClassifier",
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "class_weight": "balanced_subsample",
            "random_state": int(args.random_state),
        },
        "dataset": metadata,
        "aggregate": aggregate,
        "folds": folds,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    display_cols = [
        "fold",
        "purged_train_rows",
        "purged_rows",
        "test_rows",
        "rf_accuracy",
        "majority_accuracy",
        "accuracy_edge",
        "rf_roc_auc",
    ]
    print(folds_frame[display_cols].to_string(index=False))
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"folds: {folds_path}")
    print(f"predictions: {predictions_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
