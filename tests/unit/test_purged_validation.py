from __future__ import annotations

import pandas as pd

from deepfx_alpha_lab.validation.purged import PurgedWalkForwardSplit, apply_purge_embargo, interval_overlaps


def test_interval_overlaps_detects_half_open_overlap() -> None:
    starts = pd.to_datetime(["2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z", "2026-01-01T12:00:00Z"], utc=True)
    ends = pd.to_datetime(["2026-01-01T10:30:00Z", "2026-01-01T11:30:00Z", "2026-01-01T12:30:00Z"], utc=True)
    test_starts = pd.to_datetime(["2026-01-01T10:15:00Z", "2026-01-01T11:30:00Z"], utc=True)
    test_ends = pd.to_datetime(["2026-01-01T10:45:00Z", "2026-01-01T12:00:00Z"], utc=True)

    overlaps = interval_overlaps(starts, ends, test_starts, test_ends)

    assert overlaps.tolist() == [True, False, False]


def test_apply_purge_embargo_removes_overlaps_and_embargo_rows() -> None:
    events = pd.DataFrame(
        {
            "t0": pd.to_datetime(
                [
                    "2026-01-01T09:00:00Z",  # keep
                    "2026-01-01T10:15:00Z",  # purge overlap
                    "2026-01-01T11:05:00Z",  # embargo
                    "2026-01-01T12:00:00Z",  # keep
                ],
                utc=True,
            ),
            "t1": pd.to_datetime(
                [
                    "2026-01-01T09:30:00Z",
                    "2026-01-01T10:45:00Z",
                    "2026-01-01T11:10:00Z",
                    "2026-01-01T12:30:00Z",
                ],
                utc=True,
            ),
        }
    )
    test = pd.DataFrame(
        {
            "t0": pd.to_datetime(["2026-01-01T10:00:00Z"], utc=True),
            "t1": pd.to_datetime(["2026-01-01T11:00:00Z"], utc=True),
        }
    )

    keep_mask, diagnostics = apply_purge_embargo(events, test, embargo=pd.Timedelta(minutes=30))

    assert keep_mask.tolist() == [True, False, False, True]
    assert diagnostics["purged_overlap_rows"] == 1
    assert diagnostics["embargo_rows"] == 1


def test_purged_walk_forward_split_uses_prior_months_and_purges() -> None:
    events = pd.DataFrame(
        {
            "t0": pd.to_datetime(
                [
                    "2026-01-31T23:50:00Z",  # train candidate overlaps Feb test
                    "2026-01-15T10:00:00Z",  # train keep
                    "2026-02-01T00:05:00Z",  # test Feb
                    "2026-03-01T00:05:00Z",  # test Mar
                ],
                utc=True,
            ),
            "t1": pd.to_datetime(
                [
                    "2026-02-01T00:10:00Z",
                    "2026-01-15T10:20:00Z",
                    "2026-02-01T00:15:00Z",
                    "2026-03-01T00:20:00Z",
                ],
                utc=True,
            ),
            "bin": [1, 0, 1, 0],
        }
    )
    splitter = PurgedWalkForwardSplit(test_freq="MS", min_train_periods=1, embargo=pd.Timedelta(minutes=0))

    folds = list(splitter.split(events))

    feb = [fold for fold in folds if fold.label == "test_2026-02"][0]
    assert feb.test_indices.tolist() == [2]
    assert feb.train_indices.tolist() == [1]
    assert feb.diagnostics["purged_overlap_rows"] == 1
