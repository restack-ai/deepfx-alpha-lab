# AFML Ch04 + Ch07 Weighted Purged BB-Reversion Study

Research-only. No live trading action is taken.

## Purpose

Combine two validation guards before promoting any `double-b-v2` idea:

- **Ch04 sample uniqueness**: overlapping event intervals should not count as fully independent training evidence.
- **Ch07 purged validation**: train intervals overlapping the held-out test month are removed before fitting.

The weighted arm fits the RandomForest with per-fold `sample_weight = interval uniqueness`, computed only on the training fold. Test metrics remain unweighted out-of-sample metrics.

## Inputs

Window: `2026-01-01T00:00:00Z` to `2026-06-01T00:00:00Z`

Datasets:

- `data/processed/afml/ch03/exercise_3_4_xauusd_m1_bb_reversion_dataset.csv`
- `data/processed/afml/ch03/exercise_3_4_xauusd_m5_bb_reversion_dataset.csv`

Command pattern:

```bash
uv run --extra dev --extra ml python studies/afml/ch07/scripts/01_purged_walk_forward.py \
  --dataset <bb_reversion_dataset.csv> \
  --out-dir data/processed/afml/ch07/<run_name> \
  --start 2026-01-01T00:00:00Z \
  --end 2026-06-01T00:00:00Z \
  --embargo-days 1 \
  --weight-mode portfolio
```

## M1 BB-reversion result

Rows: `2131`

- `naive_expanding`:
  - folds: 4
  - mean majority accuracy: 0.6784
  - mean RF accuracy: 0.6707
  - RF edge vs majority: -0.0077
  - mean RF AUC: 0.5436
  - positive-edge folds: 1 / 4
- `purged_expanding`:
  - purged overlap rows total: 2
  - mean majority accuracy: 0.6784
  - mean RF accuracy: 0.6679
  - RF edge vs majority: -0.0105
  - mean RF AUC: 0.5467
  - positive-edge folds: 1 / 4
- `purged_expanding_weighted_portfolio`:
  - mean train rows: 1252.0
  - mean effective train rows: 545.4615
  - mean train sample weight: 0.4552
  - mean majority accuracy: 0.6784
  - mean RF accuracy: 0.6596
  - RF edge vs majority: -0.0188
  - mean RF AUC: 0.5445
  - positive-edge folds: 0 / 4

## M5 BB-reversion result

Rows: `441`

- `naive_expanding`:
  - folds: 4
  - mean majority accuracy: 0.6271
  - mean RF accuracy: 0.5823
  - RF edge vs majority: -0.0448
  - mean RF AUC: 0.5487
  - positive-edge folds: 1 / 4
- `purged_expanding`:
  - purged overlap rows total: 2
  - mean majority accuracy: 0.6271
  - mean RF accuracy: 0.5872
  - RF edge vs majority: -0.0399
  - mean RF AUC: 0.5566
  - positive-edge folds: 1 / 4
- `purged_expanding_weighted_portfolio`:
  - mean train rows: 258.5
  - mean effective train rows: 129.0795
  - mean train sample weight: 0.5207
  - mean majority accuracy: 0.6271
  - mean RF accuracy: 0.5675
  - RF edge vs majority: -0.0596
  - mean RF AUC: 0.5426
  - positive-edge folds: 0 / 4

## Interpretation

The BB-reversion primary signal still does not clear the bar once Ch04 and Ch07 are combined.

Important signal:

```text
M1 effective train ratio: ~45.5%
M5 effective train ratio: ~52.1%
```

That means raw BB event counts materially overstate independent evidence. After uniqueness weighting, the RF loses its only positive-edge fold in both M1 and M5 runs.

This does **not** prove Bollinger-style logic is useless. It says the current Ch03 BB-reversion meta-label dataset is not a validated `double-b-v2` edge candidate by the current safety rails.

## double-b-v2 implication

Do not promote the current BB-reversion RF meta-labeler directly.

A better `double-b-v2` research direction is:

1. Treat BB as a **setup detector**, not the final signal.
2. Require a second confirmation layer, e.g. volatility contraction/expansion, BRK continuation context, or Kronos disagreement/confirmation feature.
3. Keep Ch04 uniqueness + Ch07 purged validation as the default gate.
4. Prefer economic metrics after the next iteration: directional realized return bps, win/loss asymmetry, turnover, and simple cost haircut; accuracy alone is too blunt.

## Reusable artifact

`studies/afml/ch07/scripts/01_purged_walk_forward.py` now supports:

```text
--weight-mode none|symbol|portfolio
```

When enabled, only the training fold is weighted; test data remains untouched.
