import pandas as pd

from deepfx_alpha_lab.labeling import add_vertical_barrier, get_bins, get_events, resolve_ohlc_events, symmetric_cusum_filter


def test_symmetric_cusum_filter_samples_large_moves():
    index = pd.date_range("2026-01-01", periods=5, freq="min")
    close = pd.Series([100.0, 100.1, 101.5, 101.4, 99.0], index=index)

    events = symmetric_cusum_filter(close, threshold=0.01)

    assert list(events) == [index[2], index[4]]


def test_triple_barrier_labels_first_profit_taking_touch():
    index = pd.date_range("2026-01-01", periods=5, freq="min")
    close = pd.Series([100.0, 100.2, 101.2, 100.8, 100.7], index=index)
    t_events = pd.DatetimeIndex([index[0]])
    t1 = pd.Series(index[-1], index=t_events)
    trgt = pd.Series(0.01, index=t_events)

    events = get_events(close, t_events, [1, 1], trgt, min_ret=0.0, t1=t1)
    bins = get_bins(close, events)

    assert events.loc[index[0], "type"] == "pt"
    assert bins.loc[index[0], "bin"] == 1


def test_vertical_barrier_can_be_labeled_zero():
    index = pd.date_range("2026-01-01", periods=5, freq="min")
    close = pd.Series([100.0, 100.1, 100.2, 100.3, 100.4], index=index)
    t_events = pd.DatetimeIndex([index[0]])
    t1 = pd.Series(index[-1], index=t_events)
    trgt = pd.Series(0.10, index=t_events)

    events = get_events(close, t_events, [1, 1], trgt, min_ret=0.0, t1=t1)
    bins = get_bins(close, events, zero_on_vertical=True)

    assert events.loc[index[0], "type"] == "t1"
    assert bins.loc[index[0], "bin"] == 0


def test_meta_labeling_uses_side_adjusted_returns():
    index = pd.date_range("2026-01-01", periods=4, freq="min")
    close = pd.Series([100.0, 99.5, 99.0, 98.8], index=index)
    t_events = pd.DatetimeIndex([index[0]])
    t1 = pd.Series(index[-1], index=t_events)
    trgt = pd.Series(0.005, index=t_events)
    side = pd.Series(-1, index=t_events)

    events = get_events(close, t_events, [1, 2], trgt, min_ret=0.0, t1=t1, side=side)
    bins = get_bins(close, events)

    assert events.loc[index[0], "type"] == "pt"
    assert bins.loc[index[0], "ret"] > 0
    assert bins.loc[index[0], "bin"] == 1


def test_add_vertical_barrier_uses_first_index_after_horizon():
    index = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 12:00", "2026-01-02 00:01"])
    close = pd.Series([1.0, 1.1, 1.2], index=index)
    t_events = pd.DatetimeIndex([index[0]])

    barriers = add_vertical_barrier(t_events, close, num_days=1)

    assert barriers.loc[index[0]] == index[2]


def test_add_vertical_barrier_accepts_fractional_days_on_second_resolution_index():
    index = pd.date_range("2026-05-01 00:00:00", periods=4, freq="h").astype("datetime64[s]")
    close = pd.Series([1.0, 1.1, 1.2, 1.3], index=index)
    t_events = pd.DatetimeIndex([index[0]])

    barriers = add_vertical_barrier(t_events, close, num_days=0.0416666667)

    assert barriers.loc[index[0]] == index[2]


def test_ohlc_events_use_intrabar_high_for_profit_taking():
    index = pd.date_range("2026-01-01", periods=3, freq="min")
    path_bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.4],
            "high": [100.0, 101.2, 100.5],
            "low": [100.0, 99.8, 100.1],
            "close": [100.0, 100.4, 100.2],
        },
        index=index,
    )
    entry_close = pd.Series([100.0], index=pd.DatetimeIndex([index[0]]))
    t_events = pd.DatetimeIndex([index[0]])
    t1 = pd.Series(index[-1], index=t_events)
    trgt = pd.Series(0.01, index=t_events)

    labels = resolve_ohlc_events(entry_close, path_bars, t_events, t1, trgt, pt=1.0, sl=1.0, min_ret=0.0)

    assert labels.loc[index[0], "type"] == "pt"
    assert labels.loc[index[0], "t1"] == index[1]
    assert labels.loc[index[0], "bin"] == 1
    assert labels.loc[index[0], "ret"] == 0.01


def test_ohlc_events_ignore_entry_bar_high_low_before_trade_can_exist():
    index = pd.date_range("2026-01-01", periods=3, freq="min")
    path_bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [101.5, 100.2, 100.2],
            "low": [98.5, 99.8, 99.8],
            "close": [100.0, 100.1, 100.1],
        },
        index=index,
    )
    entry_close = pd.Series([100.0], index=pd.DatetimeIndex([index[0]]))
    t_events = pd.DatetimeIndex([index[0]])
    t1 = pd.Series(index[-1], index=t_events)
    trgt = pd.Series(0.01, index=t_events)

    labels = resolve_ohlc_events(entry_close, path_bars, t_events, t1, trgt, pt=1.0, sl=1.0, min_ret=0.0)

    assert labels.loc[index[0], "type"] == "t1"
    assert labels.loc[index[0], "t1"] == index[-1]
    assert labels.loc[index[0], "bin"] == 1


def test_ohlc_events_use_conservative_sl_first_for_same_bar_ambiguity():
    index = pd.date_range("2026-01-01", periods=2, freq="min")
    path_bars = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [100.0, 101.5],
            "low": [100.0, 98.5],
            "close": [100.0, 100.2],
        },
        index=index,
    )
    entry_close = pd.Series([100.0], index=pd.DatetimeIndex([index[0]]))
    t_events = pd.DatetimeIndex([index[0]])
    t1 = pd.Series(index[-1], index=t_events)
    trgt = pd.Series(0.01, index=t_events)

    labels = resolve_ohlc_events(entry_close, path_bars, t_events, t1, trgt, pt=1.0, sl=1.0, min_ret=0.0)

    assert labels.loc[index[0], "type"] == "sl"
    assert bool(labels.loc[index[0], "ambiguous"]) is True
    assert labels.loc[index[0], "bin"] == -1
    assert labels.loc[index[0], "ret"] == -0.01
