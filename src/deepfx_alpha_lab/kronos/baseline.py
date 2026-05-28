from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_statistical_embeddings(x: np.ndarray, *, feature_columns: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build frozen statistical embeddings from fixed windows.

    This is the MVP fallback before wiring an actual frozen Kronos encoder. It keeps
    the same classifier contract: [n_events, embedding_dim] -> label target.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape [n_events, lookback, n_features]")
    if x.shape[2] != len(feature_columns):
        raise ValueError("feature_columns length must match x.shape[2]")
    means = x.mean(axis=1)
    stds = x.std(axis=1)
    lasts = x[:, -1, :]
    slopes = x[:, -1, :] - x[:, 0, :]
    embeddings = np.concatenate([means, stds, lasts, slopes], axis=1).astype("float32")
    columns: list[str] = []
    for column in feature_columns:
        columns.extend([f"{column}_mean", f"{column}_std", f"{column}_last", f"{column}_slope"])
    # np.concatenate groups by statistic, so reorder to per-feature column order.
    grouped = []
    for feature_idx in range(len(feature_columns)):
        grouped.extend(
            [
                embeddings[:, feature_idx],
                embeddings[:, len(feature_columns) + feature_idx],
                embeddings[:, 2 * len(feature_columns) + feature_idx],
                embeddings[:, 3 * len(feature_columns) + feature_idx],
            ]
        )
    return np.stack(grouped, axis=1).astype("float32"), columns


def time_ordered_split(event_times: pd.DatetimeIndex, *, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    event_values = event_times.to_numpy()
    unique_times = np.sort(pd.unique(event_values))
    split = int(len(unique_times) * train_frac)
    if split <= 0 or split >= len(unique_times):
        raise ValueError("Not enough events for train/test split")
    cutoff = unique_times[split]
    train_idx = np.flatnonzero(event_values < cutoff)
    test_idx = np.flatnonzero(event_values >= cutoff)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Not enough events for train/test split")
    return train_idx, test_idx


def _metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def evaluate_baseline_classifiers(
    embeddings: np.ndarray,
    y: np.ndarray,
    event_times: pd.DatetimeIndex,
    *,
    train_frac: float = 0.7,
    random_state: int = 42,
) -> dict[str, object]:
    train_idx, test_idx = time_ordered_split(event_times, train_frac=train_frac)
    x_train, x_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    results: dict[str, object] = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_distribution": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()},
        "test_distribution": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()},
        "models": {},
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results["models"][name] = _metrics(y_test, pred)
    majority_acc = results["models"]["majority"]["accuracy"]
    for metrics in results["models"].values():
        metrics["accuracy_edge_vs_majority"] = float(metrics["accuracy"] - majority_acc)
    return results


def load_npz_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str], np.ndarray | None, list[str] | None, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    x = data["x"]
    y_type = data["y_type"]
    event_times = pd.to_datetime(data["event_times"].astype("int64"))
    feature_columns = [str(item) for item in data["feature_columns"].tolist()]
    kronos_x = data["kronos_x"] if "kronos_x" in data.files else None
    kronos_columns = [str(item) for item in data["kronos_columns"].tolist()] if "kronos_columns" in data.files else None
    window_times = data["window_times"] if "window_times" in data.files else None
    return x, y_type, pd.DatetimeIndex(event_times), feature_columns, kronos_x, kronos_columns, window_times


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
