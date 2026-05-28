from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_TYPE_TO_ID = {"pt": 0, "sl": 1, "t1": 2, "ambiguous": 3}
ID_TO_LABEL_TYPE = {value: key for key, value in LABEL_TYPE_TO_ID.items()}
DEFAULT_FEATURE_COLUMNS = ["open", "high", "low", "close", "tick_volume", "return_1"]


def concatenate_kronos_label_datasets(items: list[tuple[str, KronosLabelDataset]]) -> KronosLabelDataset:
    """Concatenate per-symbol Kronos datasets and keep symbol metadata."""
    non_empty = [(symbol, dataset) for symbol, dataset in items if len(dataset.y_type) > 0]
    if not non_empty:
        return KronosLabelDataset(
            x=np.empty((0, 0, 0), dtype="float32"),
            y_type=np.array([], dtype="int64"),
            y_bin=np.array([], dtype="int64"),
            y_ret=np.array([], dtype="float32"),
            event_times=pd.DatetimeIndex([]),
            feature_columns=[],
            metadata=pd.DataFrame(),
            kronos_x=np.empty((0, 0, 0), dtype="float32"),
            kronos_columns=["open", "high", "low", "close", "volume", "amount"],
            window_times=np.empty((0, 0), dtype="int64"),
        )

    feature_columns = non_empty[0][1].feature_columns
    kronos_columns = non_empty[0][1].kronos_columns
    for symbol, dataset in non_empty:
        if dataset.feature_columns != feature_columns:
            raise ValueError(f"feature column mismatch for {symbol}")
        if dataset.kronos_columns != kronos_columns:
            raise ValueError(f"kronos column mismatch for {symbol}")

    x = np.concatenate([dataset.x for _, dataset in non_empty], axis=0).astype("float32")
    kronos_x = np.concatenate([dataset.kronos_x for _, dataset in non_empty], axis=0).astype("float32")
    window_times = np.concatenate([dataset.window_times for _, dataset in non_empty], axis=0).astype("int64")
    y_type = np.concatenate([dataset.y_type for _, dataset in non_empty]).astype("int64")
    y_bin = np.concatenate([dataset.y_bin for _, dataset in non_empty]).astype("int64")
    y_ret = np.concatenate([dataset.y_ret for _, dataset in non_empty]).astype("float32")
    event_times = pd.DatetimeIndex([ts for _, dataset in non_empty for ts in dataset.event_times])
    metadata_frames: list[pd.DataFrame] = []
    for symbol, dataset in non_empty:
        metadata = dataset.metadata.copy()
        metadata["symbol"] = symbol
        metadata_frames.append(metadata)
    metadata_all = pd.concat(metadata_frames)
    # Keep a deterministic temporal order across symbols. For duplicate timestamps,
    # numpy's stable sort preserves the input symbol order.
    order = np.argsort(event_times.to_numpy(), kind="stable")
    return KronosLabelDataset(
        x=x[order],
        y_type=y_type[order],
        y_bin=y_bin[order],
        y_ret=y_ret[order],
        event_times=pd.DatetimeIndex(event_times.to_numpy()[order]),
        feature_columns=list(feature_columns),
        metadata=metadata_all.iloc[order],
        kronos_x=kronos_x[order],
        kronos_columns=list(kronos_columns),
        window_times=window_times[order],
    )


@dataclass(frozen=True)
class KronosLabelDataset:
    x: np.ndarray
    y_type: np.ndarray
    y_bin: np.ndarray
    y_ret: np.ndarray
    event_times: pd.DatetimeIndex
    feature_columns: list[str]
    metadata: pd.DataFrame
    kronos_x: np.ndarray
    kronos_columns: list[str]
    window_times: np.ndarray

    def save(self, output_dir: Path, stem: str) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / f"{stem}.npz"
        metadata_path = output_dir / f"{stem}_metadata.csv"
        np.savez_compressed(
            npz_path,
            x=self.x,
            y_type=self.y_type,
            y_bin=self.y_bin,
            y_ret=self.y_ret,
            event_times=self.event_times.astype("datetime64[ns]").astype("int64"),
            feature_columns=np.array(self.feature_columns, dtype=object),
            kronos_x=self.kronos_x,
            kronos_columns=np.array(self.kronos_columns, dtype=object),
            window_times=self.window_times,
        )
        self.metadata.to_csv(metadata_path)
        return {"npz": npz_path, "metadata": metadata_path}


def _prepare_features(event_bars: pd.DataFrame) -> pd.DataFrame:
    bars = event_bars.sort_index().copy()
    required = {"open", "high", "low", "close", "tick_volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"event_bars missing required columns: {sorted(missing)}")
    features = bars[["open", "high", "low", "close", "tick_volume"]].astype("float64")
    features["return_1"] = features["close"].pct_change().fillna(0.0)
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def _prepare_kronos_features(event_bars: pd.DataFrame) -> pd.DataFrame:
    bars = event_bars.sort_index().copy()
    required = {"open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"event_bars missing required columns: {sorted(missing)}")
    volume_candidate = pd.to_numeric(bars["volume"], errors="coerce") if "volume" in bars else None
    if volume_candidate is not None and volume_candidate.fillna(0.0).abs().sum() > 0:
        volume = volume_candidate
    elif "tick_volume" in bars:
        volume = pd.to_numeric(bars["tick_volume"], errors="coerce")
    else:
        volume = pd.Series(0.0, index=bars.index)

    kronos = bars[["open", "high", "low", "close"]].astype("float64")
    kronos["volume"] = volume.astype("float64")
    kronos["amount"] = kronos["volume"] * kronos["close"]
    return kronos.replace([np.inf, -np.inf], np.nan).dropna()


def _normalize_window(window: pd.DataFrame) -> np.ndarray:
    values = window.to_numpy(dtype="float64")
    close_anchor = float(window["close"].iloc[-1])
    volume_anchor = float(max(window["tick_volume"].median(), 1.0))
    out = values.copy()
    # OHLC columns become relative to the event-bar close. The final close is 0.
    out[:, 0:4] = out[:, 0:4] / close_anchor - 1.0
    # Volume is robustly scaled without needing future information.
    out[:, 4] = np.log1p(out[:, 4]) / np.log1p(volume_anchor) - 1.0
    # return_1 is already stationary; keep as-is.
    return out.astype("float32")


def build_kronos_label_dataset(
    event_bars: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    lookback: int,
    feature_columns: list[str] | None = None,
) -> KronosLabelDataset:
    """Build fixed lookback windows and multiclass triple-barrier targets.

    The MVP target is `type`: `pt`, `sl`, or `t1` from execution-aware labels.
    Windows are normalized using information available at the event timestamp only.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    features = _prepare_features(event_bars)
    kronos_features = _prepare_kronos_features(event_bars)
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    missing = set(feature_columns).difference(features.columns)
    if missing:
        raise ValueError(f"feature_columns missing from prepared features: {sorted(missing)}")
    features = features[feature_columns]
    kronos_columns = ["open", "high", "low", "close", "volume", "amount"]

    labels = labels.sort_index().copy()
    rows: list[np.ndarray] = []
    kronos_rows: list[np.ndarray] = []
    window_time_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    event_times: list[pd.Timestamp] = []
    y_type: list[int] = []
    y_bin: list[int] = []
    y_ret: list[float] = []

    for event_time, label in labels.iterrows():
        if event_time not in features.index:
            continue
        if event_time not in kronos_features.index:
            continue
        end_pos = features.index.get_loc(event_time)
        kronos_end_pos = kronos_features.index.get_loc(event_time)
        if isinstance(end_pos, (slice, np.ndarray)) or isinstance(kronos_end_pos, (slice, np.ndarray)):
            continue
        start_pos = int(end_pos) - lookback + 1
        kronos_start_pos = int(kronos_end_pos) - lookback + 1
        if start_pos < 0 or kronos_start_pos < 0:
            continue
        label_type = str(label["type"])
        if label_type not in LABEL_TYPE_TO_ID:
            continue
        window = features.iloc[start_pos : int(end_pos) + 1]
        kronos_window = kronos_features.iloc[kronos_start_pos : int(kronos_end_pos) + 1]
        if len(window) != lookback or len(kronos_window) != lookback:
            continue

        label_type_id = LABEL_TYPE_TO_ID[label_type]
        rows.append(_normalize_window(window))
        kronos_rows.append(kronos_window[kronos_columns].to_numpy(dtype="float32"))
        window_time_rows.append(kronos_window.index.astype("datetime64[ns]").astype("int64").to_numpy())
        event_times.append(pd.Timestamp(event_time))
        y_type.append(label_type_id)
        y_bin.append(int(label.get("bin", 0)))
        y_ret.append(float(label.get("ret", 0.0)))
        metadata_rows.append(
            {
                "t0": pd.Timestamp(event_time),
                "t1": label.get("t1"),
                "label_type": label_type,
                "label_type_id": label_type_id,
                "bin": int(label.get("bin", 0)),
                "ret": float(label.get("ret", 0.0)),
                "trgt": float(label.get("trgt", np.nan)),
            }
        )

    x = np.stack(rows).astype("float32") if rows else np.empty((0, lookback, len(feature_columns)), dtype="float32")
    kronos_x = (
        np.stack(kronos_rows).astype("float32")
        if kronos_rows
        else np.empty((0, lookback, len(kronos_columns)), dtype="float32")
    )
    window_times = (
        np.stack(window_time_rows).astype("int64") if window_time_rows else np.empty((0, lookback), dtype="int64")
    )
    metadata = pd.DataFrame(metadata_rows)
    if not metadata.empty:
        metadata = metadata.set_index("t0").sort_index()
    return KronosLabelDataset(
        x=x,
        y_type=np.array(y_type, dtype="int64"),
        y_bin=np.array(y_bin, dtype="int64"),
        y_ret=np.array(y_ret, dtype="float32"),
        event_times=pd.DatetimeIndex(event_times),
        feature_columns=list(feature_columns),
        metadata=metadata,
        kronos_x=kronos_x,
        kronos_columns=kronos_columns,
        window_times=window_times,
    )
