# AFML Chapter 7 Cross-Validation Mini Project

This mini project applies AFML Chapter 7 validation ideas to DeepFX research datasets.

Chapter 7's core warning:

```text
Financial labels span time intervals.
If train label intervals overlap test label intervals, validation leaks information.
```

The first pass implements:

- purged expanding walk-forward validation;
- interval-overlap purging using `t0/t1` labels;
- optional post-test embargo;
- comparison against naive expanding walk-forward validation.

## Study 01: Ch03 XAUUSD M1 meta-label data

Input:

```text
data/processed/afml/ch03/exercise_3_4_xauusd_m1_dataset.csv
```

Run:

```bash
uv run --extra dev --extra ml python studies/afml/ch07/scripts/01_purged_walk_forward.py \
  --dataset data/processed/afml/ch03/exercise_3_4_xauusd_m1_dataset.csv \
  --out-dir data/processed/afml/ch07/purged_xauusd_m1_202601_202605 \
  --start 2026-01-01T00:00:00Z \
  --end 2026-06-01T00:00:00Z \
  --embargo-days 1
```

Default outputs:

```text
data/processed/afml/ch07/purged_xauusd_m1_202601_202605/fold_metrics.csv
data/processed/afml/ch07/purged_xauusd_m1_202601_202605/summary.csv
data/processed/afml/ch07/purged_xauusd_m1_202601_202605/report.md
```

## Latest result: 2026-01 to 2026-05

Dataset:

```text
rows: 4520
features: side, ema_diff, returns, volatility, range/body, tick-volume, time features
folds: Feb, Mar, Apr, May held out one month at a time
embargo: 1 day
```

### Naive expanding walk-forward

```text
folds: 4
mean_train_rows: 2802.75
mean_test_rows: 875.00
mean_majority_accuracy: 0.6451
mean_rf_accuracy: 0.6146
mean_rf_accuracy_edge_vs_majority: -0.0305
mean_rf_auc: 0.5271
positive_edge_folds: 0 / 4
```

### Purged expanding walk-forward

```text
folds: 4
mean_candidate_train_rows: 2802.75
mean_train_rows_after_purge: 2801.25
purged_overlap_rows_total: 6
embargo_rows_total: 0
mean_majority_accuracy: 0.6451
mean_rf_accuracy: 0.6186
mean_rf_accuracy_edge_vs_majority: -0.0265
mean_rf_auc: 0.5265
positive_edge_folds: 0 / 4
```

## Interpretation

Purging found only a small amount of label-interval leakage in the existing monthly expanding setup:

```text
6 train rows removed across 4 folds
```

So for this particular Ch03 monthly walk-forward setup:

```text
month-boundary label leakage is present but not the main reason the model fails.
```

The RandomForest still does not beat the majority baseline:

```text
purged RF mean accuracy: 0.6186
majority mean accuracy: 0.6451
edge: -0.0265
positive-edge folds: 0 / 4
```

This reinforces the earlier Ch03 conclusion: the EMA primary + RF meta-labeling baseline is not a validated edge.

## Notes

- Prior-only expanding validation means post-test embargo usually removes no rows because future rows are not train candidates.
- Embargo becomes more important for non-prior-only schemes such as combinatorial purged CV or k-fold-style splits.
- The main reusable artifact is `src/deepfx_alpha_lab/validation/purged.py`.

## Next steps

1. Apply the same purged split to M5 and BB-reversion Ch03 variants.
2. Add sample uniqueness weights from Ch04 into purged validation.
3. Implement combinatorial purged CV when sample size supports it.
4. Apply Ch07 split to BRK/Kronos rolling-week experiments once multi-week live trade exports are available.
