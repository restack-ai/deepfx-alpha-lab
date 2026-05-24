import numpy as np
import pandas as pd

from deepfx_alpha_lab.kronos import build_kronos_label_dataset, concatenate_kronos_label_datasets


def test_build_kronos_label_dataset_creates_fixed_windows_and_multiclass_targets():
    index = pd.date_range("2026-01-01 00:00", periods=6, freq="15min")
    event_bars = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "tick_volume": [10, 11, 12, 13, 14, 15],
        },
        index=index,
    )
    labels = pd.DataFrame(
        {
            "t1": [index[4], index[5]],
            "type": ["pt", "sl"],
            "ret": [0.005, -0.005],
            "bin": [1, -1],
            "trgt": [0.01, 0.01],
        },
        index=pd.DatetimeIndex([index[3], index[4]]),
    )

    dataset = build_kronos_label_dataset(event_bars, labels, lookback=4)

    assert dataset.x.shape == (2, 4, 6)
    assert dataset.y_type.tolist() == [0, 1]
    assert dataset.y_bin.tolist() == [1, -1]
    assert dataset.event_times.tolist() == [index[3], index[4]]
    assert dataset.feature_columns == ["open", "high", "low", "close", "tick_volume", "return_1"]
    np.testing.assert_allclose(dataset.x[0, -1, 3], 0.0)
    assert dataset.metadata.loc[index[3], "label_type_id"] == 0


def test_build_kronos_label_dataset_skips_events_without_enough_history():
    index = pd.date_range("2026-01-01 00:00", periods=4, freq="15min")
    event_bars = pd.DataFrame(
        {
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100.5, 101.5, 102.5, 103.5],
            "tick_volume": [10, 11, 12, 13],
        },
        index=index,
    )
    labels = pd.DataFrame(
        {
            "t1": [index[2], index[3]],
            "type": ["pt", "t1"],
            "ret": [0.005, 0.0],
            "bin": [1, 0],
            "trgt": [0.01, 0.01],
        },
        index=pd.DatetimeIndex([index[1], index[3]]),
    )

    dataset = build_kronos_label_dataset(event_bars, labels, lookback=3)

    assert dataset.x.shape == (1, 3, 6)
    assert dataset.event_times.tolist() == [index[3]]
    assert dataset.y_type.tolist() == [2]


def test_concatenate_kronos_label_datasets_keeps_symbol_metadata():
    index = pd.date_range("2026-01-01 00:00", periods=5, freq="15min")
    base_bars = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "tick_volume": [10, 11, 12, 13, 14],
        },
        index=index,
    )
    labels_a = pd.DataFrame(
        {"t1": [index[4]], "type": ["pt"], "ret": [0.005], "bin": [1], "trgt": [0.01]},
        index=pd.DatetimeIndex([index[3]]),
    )
    labels_b = pd.DataFrame(
        {"t1": [index[4]], "type": ["sl"], "ret": [-0.005], "bin": [-1], "trgt": [0.01]},
        index=pd.DatetimeIndex([index[4]]),
    )
    ds_a = build_kronos_label_dataset(base_bars, labels_a, lookback=3)
    ds_b = build_kronos_label_dataset(base_bars, labels_b, lookback=3)

    combined = concatenate_kronos_label_datasets([("XAUUSD", ds_a), ("NAS100", ds_b)])

    assert combined.x.shape == (2, 3, 6)
    assert combined.y_type.tolist() == [0, 1]
    assert combined.metadata["symbol"].tolist() == ["XAUUSD", "NAS100"]
    assert combined.metadata.index.tolist() == [index[3], index[4]]
