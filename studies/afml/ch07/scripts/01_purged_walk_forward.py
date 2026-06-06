#!/usr/bin/env python3
"""AFML Ch07 purged walk-forward validation on Ch03 meta-label data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from deepfx_alpha_lab.validation.purged import PurgedFold, PurgedWalkForwardSplit
from deepfx_alpha_lab.validation.weighted import WeightMode, add_interval_uniqueness_weights

DEFAULT_FEATURES = (
    "side",
    "ema_diff",
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "vol_30",
    "vol_120",
    "daily_vol",
    "range_pct",
    "body_pct",
    "tick_volume_log",
    "tick_volume_z_120",
    "time_sin",
    "time_cos",
    "day_of_week",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/afml/ch03/exercise_3_4_xauusd_m1_dataset.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/afml/ch07/purged_xauusd_m1_202601_202605"))
    parser.add_argument("--start", default="2026-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-06-01T00:00:00Z")
    parser.add_argument("--embargo-days", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument(
        "--weight-mode",
        choices=["none", "symbol", "portfolio"],
        default="none",
        help="Ch04 uniqueness sample weights for the train fold. Use portfolio for single-symbol datasets without a symbol column.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset, start=args.start, end=args.end)
    feature_cols = [col for col in DEFAULT_FEATURES if col in dataset.columns]
    if not feature_cols:
        raise ValueError("no feature columns found")

    naive_folds = make_naive_walk_forward_folds(dataset)
    purged_folds = list(
        PurgedWalkForwardSplit(
            min_train_periods=1,
            embargo=pd.Timedelta(days=args.embargo_days),
        ).split(dataset)
    )
    naive_metrics = evaluate_folds(
        dataset,
        naive_folds,
        feature_cols=feature_cols,
        model_label="naive_expanding",
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    purged_metrics = evaluate_folds(
        dataset,
        purged_folds,
        feature_cols=feature_cols,
        model_label="purged_expanding",
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    metric_frames = [naive_metrics, purged_metrics]
    if args.weight_mode != "none":
        weight_mode = cast(WeightMode, args.weight_mode)
        weighted_purged_metrics = evaluate_folds(
            dataset,
            purged_folds,
            feature_cols=feature_cols,
            model_label=f"purged_expanding_weighted_{args.weight_mode}",
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            weight_mode=weight_mode,
        )
        metric_frames.append(weighted_purged_metrics)
    fold_metrics = pd.concat(metric_frames, ignore_index=True)
    summary = summarize_fold_metrics(fold_metrics)
    report = render_report(args=args, dataset=dataset, feature_cols=feature_cols, fold_metrics=fold_metrics, summary=summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.out_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"wrote {args.out_dir / 'report.md'}")
    print(report)
    return 0


def load_dataset(path: Path, *, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted({"t0", "t1", "bin"} - set(frame.columns))
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    frame = frame.copy()
    frame["t0"] = pd.to_datetime(frame["t0"], utc=True)
    frame["t1"] = pd.to_datetime(frame["t1"], utc=True)
    frame = frame[(frame["t0"] >= pd.Timestamp(start)) & (frame["t0"] < pd.Timestamp(end))].copy()
    frame = frame.sort_values("t0").reset_index(drop=True)
    frame["bin"] = pd.to_numeric(frame["bin"], errors="raise").astype(int)
    return frame


def make_naive_walk_forward_folds(events: pd.DataFrame) -> list[PurgedFold]:
    periods = events["t0"].dt.tz_convert(None).dt.to_period("M")
    unique_periods = sorted(periods.unique())
    folds: list[PurgedFold] = []
    for idx, period in enumerate(unique_periods):
        if idx < 1:
            continue
        train_indices = events.index[periods < period].to_numpy(dtype=int)
        test_indices = events.index[periods == period].to_numpy(dtype=int)
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue
        folds.append(
            PurgedFold(
                label=f"test_{period}",
                train_indices=train_indices,
                test_indices=test_indices,
                diagnostics={
                    "fold": f"test_{period}",
                    "candidate_train_rows": int(len(train_indices)),
                    "train_rows": int(len(train_indices)),
                    "test_rows": int(len(test_indices)),
                    "purged_overlap_rows": 0,
                    "embargo_rows": 0,
                },
            )
        )
    return folds


def evaluate_folds(
    dataset: pd.DataFrame,
    folds: list[PurgedFold],
    *,
    feature_cols: list[str],
    model_label: str,
    n_estimators: int,
    random_state: int,
    weight_mode: WeightMode = "none",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_idx, fold in enumerate(folds):
        train = dataset.loc[fold.train_indices]
        test = dataset.loc[fold.test_indices]
        if train.empty or test.empty:
            continue
        x_train = train[feature_cols].fillna(0.0)
        y_train = train["bin"].astype(int)
        x_test = test[feature_cols].fillna(0.0)
        y_test = test["bin"].astype(int)
        train_weighted = add_interval_uniqueness_weights(train, mode=weight_mode) if weight_mode != "none" else None
        sample_weight = train_weighted["sample_weight"].astype(float).to_numpy() if train_weighted is not None else None
        effective_train_rows = float(sample_weight.sum()) if sample_weight is not None else float(len(train))
        if sample_weight is not None:
            weighted_positive_rate = float(np.average(y_train, weights=sample_weight))
            majority = int(weighted_positive_rate >= 0.5)
        else:
            majority = int(y_train.mean() >= 0.5)
        majority_pred = np.full(len(y_test), majority)
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state + fold_idx, class_weight="balanced_subsample")
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        rf.fit(x_train, y_train, **fit_kwargs)
        pred = rf.predict(x_test)
        proba = rf.predict_proba(x_test)[:, 1] if len(rf.classes_) > 1 else np.full(len(y_test), float(rf.classes_[0]))
        row = {
            "model": model_label,
            "fold": fold.label,
            "train_rows": int(len(train)),
            "effective_train_rows": effective_train_rows,
            "mean_train_sample_weight": float(effective_train_rows / len(train)) if len(train) else np.nan,
            "test_rows": int(len(test)),
            "candidate_train_rows": int(fold.diagnostics.get("candidate_train_rows", len(train))),
            "purged_overlap_rows": int(fold.diagnostics.get("purged_overlap_rows", 0)),
            "embargo_rows": int(fold.diagnostics.get("embargo_rows", 0)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
            "majority_accuracy": float(accuracy_score(y_test, majority_pred)),
            "rf_accuracy": float(accuracy_score(y_test, pred)),
            "rf_precision": float(precision_score(y_test, pred, zero_division=0)),
            "rf_recall": float(recall_score(y_test, pred, zero_division=0)),
            "rf_f1": float(f1_score(y_test, pred, zero_division=0)),
            "rf_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else np.nan,
        }
        row["rf_accuracy_edge_vs_majority"] = row["rf_accuracy"] - row["majority_accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    metric_cols = [
        "train_rows",
        "effective_train_rows",
        "mean_train_sample_weight",
        "test_rows",
        "candidate_train_rows",
        "purged_overlap_rows",
        "embargo_rows",
        "majority_accuracy",
        "rf_accuracy",
        "rf_accuracy_edge_vs_majority",
        "rf_precision",
        "rf_recall",
        "rf_f1",
        "rf_auc",
    ]
    rows = []
    for model, group in fold_metrics.groupby("model", sort=True):
        row: dict[str, object] = {"model": model, "folds": int(len(group))}
        for col in metric_cols:
            row[f"mean_{col}"] = float(group[col].mean())
            row[f"sum_{col}"] = float(group[col].sum()) if col.endswith("rows") else np.nan
        row["positive_edge_folds"] = int((group["rf_accuracy_edge_vs_majority"] > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def render_report(*, args: argparse.Namespace, dataset: pd.DataFrame, feature_cols: list[str], fold_metrics: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "# AFML Ch07 Purged Cross-Validation Study",
        "",
        "Research-only. No live trading action is taken.",
        "",
        "## Inputs",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Window: `{args.start}` to `{args.end}`",
        f"- Rows: {len(dataset)}",
        f"- Features: {', '.join(feature_cols)}",
        f"- Embargo days: {args.embargo_days}",
        f"- Ch04 train sample-weight mode: `{args.weight_mode}`",
        "",
        "## Summary",
        "",
        *_render_frame(summary, max_rows=20),
        "",
        "## Fold metrics",
        "",
        *_render_frame(fold_metrics, max_rows=40),
        "",
        "## Notes",
        "",
        "- Naive expanding walk-forward trains on all prior months.",
        "- Purged expanding walk-forward removes prior train events whose label intervals overlap the held-out test month.",
        "- In prior-only expanding validation, post-test embargo usually removes no rows because future rows are not train candidates.",
        "- Compare purged vs naive to quantify label-interval leakage around month boundaries.",
    ]
    return "\n".join(lines)


def _render_frame(frame: pd.DataFrame, max_rows: int = 30) -> list[str]:
    if frame.empty:
        return ["- No rows."]
    lines: list[str] = []
    for row in frame.head(max_rows).to_dict(orient="records"):
        label = str(row.get("model", row.get("fold", "row")))
        if "fold" in row and "model" in row:
            label = f"{row['model']} / {row['fold']}"
        lines.append(f"- {label}:")
        for key, value in row.items():
            if key in {"model", "fold"}:
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


if __name__ == "__main__":
    raise SystemExit(main())
