# AFML Ch07 Purged Validation Report — XAUUSD M1 Ch03 Dataset, 2026-01 to 2026-05

Research-only report. No live trading action is implied.

## Question

The Ch03 meta-labeling experiments used monthly walk-forward validation and rejected the EMA/BB + RandomForest baseline. Ch07 asks whether the validation itself has label-interval leakage:

> Do train labels overlap held-out test labels around month boundaries?

If they do, train/test separation is contaminated because each label represents an interval from `t0` to `t1`, not a point event.

## Scope

Dataset:

```text
data/processed/afml/ch03/exercise_3_4_xauusd_m1_dataset.csv
```

Date range:

```text
2026-01-01 UTC <= t0 < 2026-06-01 UTC
```

Rows:

```text
4520
```

Held-out folds:

```text
2026-02
2026-03
2026-04
2026-05
```

Model:

```text
RandomForestClassifier, 200 estimators, balanced_subsample
```

Features:

```text
side, ema_diff, ret_1, ret_5, ret_15, ret_60,
vol_30, vol_120, daily_vol,
range_pct, body_pct,
tick_volume_log, tick_volume_z_120,
time_sin, time_cos, day_of_week
```

## Method

Two validation modes were compared.

### Naive expanding walk-forward

For each test month:

```text
train = all prior months
test = current month
```

No interval purging.

### Purged expanding walk-forward

For each test month:

```text
candidate_train = all prior months
remove train rows where [train_t0, train_t1) overlaps any [test_t0, test_t1)
apply 1-day post-test embargo
```

Because this is prior-only expanding validation, post-test embargo usually removes no rows; future months are not train candidates.

## Summary

### Naive expanding

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

### Purged expanding

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

## Fold details

### February 2026

```text
train rows: 1020
purged rows: 0
test rows: 1323
majority accuracy: 0.6576
RF accuracy: 0.5971
RF edge: -0.0605
AUC: 0.4765
```

### March 2026

```text
train rows: 2343
purged rows: 0
test rows: 1418
majority accuracy: 0.6566
RF accuracy: 0.6417
RF edge: -0.0148
AUC: 0.5319
```

### April 2026

```text
train rows: 3761
purged rows: 0
test rows: 326
majority accuracy: 0.6196
RF accuracy: 0.5982
RF edge: -0.0215
AUC: 0.5983
```

### May 2026

```text
candidate train rows: 4087
train rows after purge: 4081
purged rows: 6
test rows: 433
majority accuracy: 0.6467
RF accuracy naive: 0.6212
RF accuracy purged: 0.6374
RF edge purged: -0.0092
AUC purged: 0.4992
```

May is the only fold where month-boundary purging removed rows.

## Interpretation

The important result is not that purged RF slightly improved from 0.6146 to 0.6186. That movement is too small to treat as meaningful.

The important result is:

```text
Only 6 rows were purged across 4 monthly folds.
```

So for the current Ch03 monthly expanding protocol:

```text
label-interval leakage exists, but it is not the main reason the model fails.
```

The RandomForest still fails the basic test:

```text
RF does not beat majority baseline in any fold.
```

This supports the earlier Ch03 verdict:

> EMA primary + RandomForest meta-labeling remains a rejected baseline, even under purged validation.

## Verdict

Verdict: **Ch07 validation implemented; Ch03 baseline still rejected**.

What validated:

- Purged walk-forward infrastructure works on interval labels.
- The existing monthly expanding Ch03 setup has low boundary leakage.
- Purging does not rescue the model.

What failed:

- RF accuracy remains below majority baseline.
- Positive-edge folds remain 0 / 4.
- AUC remains weak around 0.526.

## Next steps

1. Run the same Ch07 script on:
   - M5 EMA meta-label dataset
   - M1 BB-reversion dataset
   - M5 BB-reversion dataset
2. Add Ch04 sample uniqueness weights to evaluation/training.
3. Build combinatorial purged CV for non-prior-only evaluation.
4. Reuse this split machinery for future BRK/Kronos rolling-week validation.
