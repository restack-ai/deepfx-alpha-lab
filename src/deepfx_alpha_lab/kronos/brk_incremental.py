"""Incremental explanatory-power tests for BRK-only vs BRK+Kronos features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BrkIncrementalConfig:
    """Configuration for BRK incremental explanatory-power tests."""

    window_minutes: int = 120
    target: str = "pnl"
    min_train_group_size: int = 2


def load_aligned_trades(path: Path, config: BrkIncrementalConfig) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"entry_time", "symbol", "side", "pnl", "window_minutes", "alignment"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"aligned trades CSV missing required columns: {missing}")
    frame = frame.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["entry_day"] = frame["entry_time"].dt.date.astype(str)
    frame = frame[frame["window_minutes"] == config.window_minutes].copy()
    frame[config.target] = pd.to_numeric(frame[config.target], errors="raise")
    return add_feature_buckets(frame).sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def add_feature_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("bb_width_pct", "atr_pct", "body_atr", "signal_age_minutes", "signal_predicted_bps"):
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["bb_width_bucket"] = _bucket_percentile(out["bb_width_pct"])
    out["atr_bucket"] = _bucket_percentile(out["atr_pct"])
    out["body_atr_bucket"] = _bucket_by_edges(out["body_atr"].abs(), edges=(0.25, 0.75, 1.5), labels=("tiny", "small", "medium", "large"))
    out["signal_age_bucket"] = _bucket_by_edges(out["signal_age_minutes"], edges=(15, 60, 120), labels=("fresh", "recent", "stale", "none"))
    out["pred_abs_bucket"] = _bucket_by_edges(out["signal_predicted_bps"].abs(), edges=(15, 30, 60), labels=("weak", "medium", "strong", "very_strong"))
    for column in ("risk_regime", "alignment", "signal_direction", "symbol", "side"):
        if column not in out.columns:
            out[column] = "unknown"
        out[column] = out[column].fillna("unknown").astype(str)
    return out


def evaluate_feature_sets(frame: pd.DataFrame, target: str = "pnl", min_train_group_size: int = 2) -> pd.DataFrame:
    """Compare BRK-only and BRK+Kronos feature sets via leave-one-day-out group means."""

    feature_sets: dict[str, list[str]] = {
        "global_mean": [],
        "brk_symbol_side": ["symbol", "side"],
        "brk_regime": ["symbol", "side", "risk_regime"],
        "brk_vol": ["symbol", "side", "risk_regime", "bb_width_bucket", "atr_bucket", "body_atr_bucket"],
        "brk_kronos_alignment": ["symbol", "side", "risk_regime", "alignment"],
        "brk_kronos_full": [
            "symbol",
            "side",
            "risk_regime",
            "bb_width_bucket",
            "atr_bucket",
            "body_atr_bucket",
            "alignment",
            "signal_age_bucket",
            "pred_abs_bucket",
        ],
    }
    rows = []
    for name, features in feature_sets.items():
        predictions = leave_one_day_out_group_mean(frame, features, target=target, min_train_group_size=min_train_group_size)
        rows.append(summarize_predictions(name, features, predictions, target=target))
    return pd.DataFrame(rows).sort_values(["mae", "rmse", "feature_set"]).reset_index(drop=True)


def leave_one_day_out_group_mean(
    frame: pd.DataFrame,
    features: list[str],
    target: str = "pnl",
    min_train_group_size: int = 2,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy().assign(prediction=pd.Series(dtype=float), fallback_level=pd.Series(dtype=str))
    rows = []
    days = sorted(frame["entry_day"].unique())
    for day in days:
        train = frame[frame["entry_day"] != day]
        test = frame[frame["entry_day"] == day]
        for _, row in test.iterrows():
            prediction, fallback_level = _predict_group_mean(train, row, features, target, min_train_group_size)
            out = row.to_dict()
            out["prediction"] = prediction
            out["fallback_level"] = fallback_level
            rows.append(out)
    return pd.DataFrame(rows)


def summarize_predictions(name: str, features: list[str], predictions: pd.DataFrame, target: str = "pnl") -> dict[str, object]:
    if predictions.empty:
        return {
            "feature_set": name,
            "features": ",".join(features) or "<none>",
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_error": np.nan,
            "sign_accuracy": np.nan,
            "positive_precision": np.nan,
            "fallback_global_rate": np.nan,
        }
    y = predictions[target].astype(float)
    pred = predictions["prediction"].astype(float)
    error = pred - y
    positive_pred = pred > 0
    positive_actual = y > 0
    return {
        "feature_set": name,
        "features": ",".join(features) or "<none>",
        "n": int(len(predictions)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_error": float(error.mean()),
        "sign_accuracy": float((positive_pred == positive_actual).mean()),
        "positive_precision": float((positive_actual[positive_pred]).mean()) if positive_pred.any() else np.nan,
        "predicted_positive_rate": float(positive_pred.mean()),
        "actual_positive_rate": float(positive_actual.mean()),
        "fallback_global_rate": float((predictions["fallback_level"] == "global").mean()),
    }


def summarize_kronos_lift(frame: pd.DataFrame, target: str = "pnl") -> pd.DataFrame:
    """Bucket-level descriptive lift from adding Kronos alignment."""

    rows = []
    base_cols = ["symbol", "side", "risk_regime"]
    for keys, label in [
        (base_cols, "brk_regime"),
        (base_cols + ["alignment"], "brk_regime_alignment"),
        (base_cols + ["alignment", "signal_age_bucket"], "brk_regime_alignment_age"),
    ]:
        for group_key, group in frame.groupby(keys, dropna=False, sort=True):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = {"bucket_family": label, "n": int(len(group)), "pnl_sum": float(group[target].sum()), "pnl_mean": float(group[target].mean()), "win_rate": float((group[target] > 0).mean())}
            for key, value in zip(keys, group_key, strict=True):
                row[key] = value
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["bucket_family", "pnl_sum"], ascending=[True, False]).reset_index(drop=True)


def build_incremental_study(aligned_path: Path, config: BrkIncrementalConfig, out_dir: Path | None = None) -> dict[str, pd.DataFrame | str]:
    frame = load_aligned_trades(aligned_path, config)
    feature_eval = evaluate_feature_sets(frame, target=config.target, min_train_group_size=config.min_train_group_size)
    bucket_lift = summarize_kronos_lift(frame, target=config.target)
    predictions = []
    for feature_set, features in {
        "brk_regime": ["symbol", "side", "risk_regime"],
        "brk_kronos_alignment": ["symbol", "side", "risk_regime", "alignment"],
    }.items():
        pred = leave_one_day_out_group_mean(frame, features, target=config.target, min_train_group_size=config.min_train_group_size)
        pred["feature_set"] = feature_set
        predictions.append(pred)
    prediction_rows = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    report = render_incremental_report(config, frame, feature_eval, bucket_lift)
    outputs: dict[str, pd.DataFrame | str] = {
        "feature_eval": feature_eval,
        "bucket_lift": bucket_lift,
        "prediction_rows": prediction_rows,
        "report": report,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, value in outputs.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / "report.md").write_text(report, encoding="utf-8")
    return outputs


def render_incremental_report(config: BrkIncrementalConfig, frame: pd.DataFrame, feature_eval: pd.DataFrame, bucket_lift: pd.DataFrame) -> str:
    lines = [
        "# BRK-only vs BRK+Kronos Incremental Study",
        "",
        "Research-only. No live trading action is taken.",
        "",
        "## Inputs",
        "",
        f"- Window minutes: {config.window_minutes}",
        f"- Target: `{config.target}`",
        f"- Rows: {len(frame)}",
        f"- Days: {frame['entry_day'].nunique() if not frame.empty else 0}",
        "",
        "## 1. Leave-one-day-out feature-set comparison",
        "",
        *_render_frame(feature_eval, max_rows=20),
        "",
        "## 2. Bucket-level Kronos lift diagnostics",
        "",
        *_render_frame(bucket_lift, max_rows=60),
        "",
        "## Notes",
        "",
        "- The evaluation predicts each held-out day using group means from the other days.",
        "- If a group is too sparse, prediction backs off to progressively simpler prefixes, then global mean.",
        "- Lower MAE/RMSE is better; higher sign accuracy is better.",
        "- With one week of data, this is a stability check, not a production model.",
    ]
    return "\n".join(lines)


def _predict_group_mean(train: pd.DataFrame, row: pd.Series, features: list[str], target: str, min_train_group_size: int) -> tuple[float, str]:
    if train.empty:
        return 0.0, "empty"
    for width in range(len(features), 0, -1):
        subset = features[:width]
        mask = pd.Series(True, index=train.index)
        for feature in subset:
            mask &= train[feature].astype(str) == str(row[feature])
        group = train[mask]
        if len(group) >= min_train_group_size:
            return float(group[target].mean()), "+".join(subset)
    return float(train[target].mean()), "global"


def _bucket_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    labels = pd.Series("unknown", index=series.index, dtype=object)
    labels[(values >= 0.0) & (values < 1 / 3)] = "low"
    labels[(values >= 1 / 3) & (values < 2 / 3)] = "mid"
    labels[values >= 2 / 3] = "high"
    return labels


def _bucket_by_edges(series: pd.Series, edges: tuple[float, ...], labels: tuple[str, ...]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(labels[-1], index=series.index, dtype=object)
    out[values.isna()] = "unknown"
    lower = -np.inf
    for edge, label in zip(edges, labels, strict=False):
        out[(values >= lower) & (values < edge)] = label
        lower = edge
    out[values >= lower] = labels[-1]
    return out


def _render_frame(frame: pd.DataFrame, max_rows: int = 30) -> list[str]:
    if frame.empty:
        return ["- No rows."]
    lines: list[str] = []
    for row in frame.head(max_rows).to_dict(orient="records"):
        title = row.get("feature_set") or row.get("bucket_family") or "row"
        lines.append(f"- {title}:")
        for key, value in row.items():
            if key in {"feature_set", "bucket_family"}:
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
