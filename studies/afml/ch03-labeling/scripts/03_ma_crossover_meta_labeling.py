#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AFML Exercise 3.4 on XAUUSD M1 bars.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--primary", choices=["ema", "bb-reversion"], default="ema")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--bb-window", type=int, default=20)
    parser.add_argument("--bb-num-std", type=float, default=2.0)
    parser.add_argument("--pt", type=float, default=1.0)
    parser.add_argument("--sl", type=float, default=2.0)
    parser.add_argument("--num-days", type=float, default=1.0)
    parser.add_argument("--vol-span", type=int, default=100)
    parser.add_argument("--min-ret", type=float, default=0.0)
    parser.add_argument("--min-daily-bars", type=int, default=60)
    parser.add_argument("--train-frac", type=float, default=0.70)
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


def make_primary_side(close: pd.Series, fast: int, slow: int) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    side = pd.Series(np.where(ema_fast > ema_slow, 1, -1), index=close.index, name="side")
    return pd.DataFrame(
        {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "side": side,
            "ema_diff": ema_fast / ema_slow - 1.0,
        }
    )


def make_bb_reversion_side(close: pd.Series, window: int, num_std: float) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    side = pd.Series(np.nan, index=close.index, name="side")
    side.loc[close < lower] = 1
    side.loc[close > upper] = -1
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "side": side,
            "bb_z": (close - mid) / std,
            "bb_width": (upper - lower) / close,
            "bb_dist_to_lower": close / lower - 1.0,
            "bb_dist_to_upper": close / upper - 1.0,
        }
    ).replace([np.inf, -np.inf], np.nan)


def make_primary_signal(
    close: pd.Series,
    *,
    primary: str,
    fast: int,
    slow: int,
    bb_window: int,
    bb_num_std: float,
) -> pd.DataFrame:
    if primary == "ema":
        return make_primary_side(close, fast=fast, slow=slow)
    if primary == "bb-reversion":
        return make_bb_reversion_side(close, window=bb_window, num_std=bb_num_std)
    raise ValueError(f"Unsupported primary signal: {primary}")


def make_features(bars: pd.DataFrame, daily_vol: pd.Series, primary: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    log_ret = np.log(close).diff()

    features = pd.DataFrame(index=bars.index)
    features["side"] = primary["side"]
    for col in ["ema_diff", "bb_z", "bb_width", "bb_dist_to_lower", "bb_dist_to_upper"]:
        if col in primary:
            features[col] = primary[col]
    features["ret_1"] = close.pct_change(1)
    features["ret_5"] = close.pct_change(5)
    features["ret_15"] = close.pct_change(15)
    features["ret_60"] = close.pct_change(60)
    features["vol_30"] = log_ret.rolling(30).std()
    features["vol_120"] = log_ret.rolling(120).std()
    features["daily_vol"] = daily_vol.reindex(features.index, method="ffill")
    features["range_pct"] = (bars["high"] - bars["low"]) / close
    features["body_pct"] = (bars["close"] - bars["open"]) / bars["open"]
    features["tick_volume_log"] = np.log1p(bars["tick_volume"])
    tick_volume_mean = features["tick_volume_log"].rolling(120).mean()
    tick_volume_std = features["tick_volume_log"].rolling(120).std()
    features["tick_volume_z_120"] = (features["tick_volume_log"] - tick_volume_mean) / tick_volume_std.replace(0, np.nan)
    features["tick_volume_z_120"] = features["tick_volume_z_120"].fillna(0.0)
    minute_of_day = features.index.hour * 60 + features.index.minute
    features["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    features["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    features["day_of_week"] = features.index.dayofweek
    return features.replace([np.inf, -np.inf], np.nan)


def split_time_ordered(dataset: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_frac < 1:
        raise ValueError("--train-frac must be between 0 and 1")
    split = int(len(dataset) * train_frac)
    if split == 0 or split == len(dataset):
        raise RuntimeError("Not enough rows for a train/test split")
    return dataset.iloc[:split], dataset.iloc[split:]


def compute_metrics(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray | None) -> dict[str, object]:
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    if proba is not None and y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bars = load_ohlcv(args.symbol, args.timeframe, start=args.start, end=args.end)
    if bars.empty:
        raise RuntimeError(f"No bars found for {args.symbol} {args.timeframe}")

    bars = bars.loc[(bars["close"] > 0) & (bars["high"] >= bars["low"])]
    filtered_bars = filter_sparse_days(bars, args.min_daily_bars)
    close = filtered_bars["close"].dropna()
    if close.empty:
        raise RuntimeError("No usable close prices after filtering")

    daily_vol = get_daily_vol(close, span=args.vol_span).dropna()
    if daily_vol.empty:
        raise RuntimeError("Daily volatility is empty; widen the date range or lower filters")

    threshold = float(daily_vol.mean())
    t_events = symmetric_cusum_filter(close, threshold=threshold)
    t1 = add_vertical_barrier(t_events, close, num_days=args.num_days)
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

    features = make_features(filtered_bars, daily_vol, primary)
    feature_cols = list(features.columns)
    dataset = features.reindex(labels.index).join(labels[["t1", "trgt", "type", "ret", "bin"]])
    dataset = dataset.dropna(subset=feature_cols + ["bin"]).copy()
    dataset["bin"] = dataset["bin"].astype(int)
    if dataset["bin"].nunique() < 2:
        raise RuntimeError("Meta-label target has fewer than two classes")

    train, test = split_time_ordered(dataset, args.train_frac)
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

    predictions = test[["t1", "trgt", "side", "type", "ret", "bin"]].copy()
    predictions["pred"] = pred
    if proba is not None:
        predictions["proba_1"] = proba

    feature_importance = dict(
        sorted(
            zip(feature_cols, [float(v) for v in model.feature_importances_]),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    summary = {
        "exercise": "3.4",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start": str(filtered_bars.index.min()),
        "end": str(filtered_bars.index.max()),
        "raw_rows_after_dedup": int(len(bars)),
        "rows_after_sparse_day_filter": int(len(filtered_bars)),
        "daily_vol_mean": float(daily_vol.mean()),
        "cusum_threshold": threshold,
        "cusum_events": int(len(t_events)),
        "meta_label_events": int(len(labels)),
        "dataset_rows": int(len(dataset)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "fast_ema": int(args.fast),
        "slow_ema": int(args.slow),
        "primary": args.primary,
        "bb_window": int(args.bb_window) if args.primary == "bb-reversion" else None,
        "bb_num_std": float(args.bb_num_std) if args.primary == "bb-reversion" else None,
        "pt_sl": [float(args.pt), float(args.sl)],
        "num_days": float(args.num_days),
        "label_distribution": distribution(dataset["bin"]),
        "train_label_distribution": distribution(train["bin"]),
        "test_label_distribution": distribution(test["bin"]),
        "side_distribution": distribution(dataset["side"]),
        "barrier_distribution": distribution(dataset["type"]),
        "model": {
            "name": "RandomForestClassifier",
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "class_weight": "balanced_subsample",
            "random_state": int(args.random_state),
        },
        "metrics": compute_metrics(test["bin"], pred, proba),
        "majority_baseline_metrics": compute_metrics(test["bin"], majority_pred, None),
        "feature_importance": feature_importance,
    }

    primary_suffix = "" if args.primary == "ema" else f"_{args.primary.replace('-', '_')}"
    stem = f"exercise_3_4_{args.symbol.lower()}_{args.timeframe.lower()}{primary_suffix}"
    labels_path = args.output_dir / f"{stem}_meta_labels.csv"
    dataset_path = args.output_dir / f"{stem}_dataset.csv"
    predictions_path = args.output_dir / f"{stem}_predictions.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"

    labels.to_csv(labels_path)
    dataset.to_csv(dataset_path)
    predictions.to_csv(predictions_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"labels: {labels_path}")
    print(f"dataset: {dataset_path}")
    print(f"predictions: {predictions_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
