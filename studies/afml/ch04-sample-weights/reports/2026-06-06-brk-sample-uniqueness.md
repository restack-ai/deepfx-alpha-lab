# AFML Ch04 BRK Sample Uniqueness Report — 2026-05-31 to 2026-06-06

Research-only report. No live trading action is implied.

## Question

The earlier BRK × Kronos work showed a tempting pattern:

```text
BRK trades opposed to the latest prior Kronos shadow produced most of the weekly PnL.
```

But before treating that as meaningful evidence, AFML Chapter 4 asks a more basic question:

> Are those 15 opposed BRK trades really 15 independent samples, or are they overlapping slices of the same market move?

This report applies sample uniqueness weighting to the BRK trade intervals.

## Method

For each BRK trade:

```text
t0 = entry_time
t1 = exit_time
```

Concurrency is computed over the active intervals. Each trade receives:

```text
uniqueness = duration-weighted average of 1 / concurrency
```

Then:

```text
effective_n = sum(uniqueness)
weighted_pnl = pnl * uniqueness
```

Two concurrency pools were tested:

1. `symbol` — only same-symbol overlapping trades reduce uniqueness.
2. `portfolio` — all BRK trades share one concurrency pool, so cross-symbol overlap also reduces uniqueness.

Input rows:

- Source: `data/processed/brk_kronos_disagreement/20260531_20260606/study/aligned_trades.csv`
- Window: 120-minute BRK/Kronos alignment rows
- BRK trades: 21

## Same-symbol concurrency result

When pooling separately by symbol:

```text
raw_n: 21
effective_n: 21.0000
effective/raw ratio: 1.0000
mean_uniqueness: 1.0000
mean_concurrency: 1.0000
```

Interpretation:

> Same-symbol overlapping intervals did not inflate this week's BRK count.

So the XAUUSD/XAGUSD BRK observations are not simply the same symbol holding multiple overlapping positions at once.

## Portfolio concurrency result

When pooling all BRK trades portfolio-wide:

```text
raw_n: 21
effective_n: 18.4137
effective/raw ratio: 0.8768
mean_uniqueness: 0.8768
mean_concurrency: 1.2463
```

Interpretation:

> Cross-symbol overlap compresses the weekly sample by about 12.3%.

This is meaningful, but not catastrophic. The week is not one giant duplicated event; it is more like 21 raw trades representing about 18.4 independent interval-weighted observations.

## Kronos alignment after uniqueness weighting

Portfolio-wide pool:

```text
opposed:
  raw_n: 15
  effective_n: 12.7702
  mean_uniqueness: 0.8513
  raw_pnl_sum: +82.84
  weighted_pnl_sum: +74.76
  raw_pnl_mean: +5.52
  weighted_pnl_mean: +5.85
  weighted_win_rate: 73.35%

no_recent_kronos:
  raw_n: 5
  effective_n: 4.6435
  mean_uniqueness: 0.9287
  raw_pnl_sum: +1.12
  weighted_pnl_sum: +1.09
  weighted_pnl_mean: +0.23

aligned:
  raw_n: 1
  effective_n: 1.0000
  raw_pnl_sum: -1.17
  weighted_pnl_sum: -1.17
```

The opposed bucket still dominates after uniqueness weighting:

```text
opposed raw pnl:      +82.84
opposed weighted pnl: +74.76
```

But this should be read carefully. Chapter 4 says the bucket was not just a raw-count artifact, but it does **not** prove Kronos adds incremental signal.

The previous incremental test still matters:

```text
BRK symbol+side baseline beat BRK+Kronos features in leave-one-day-out testing.
```

## Symbol-side details

Portfolio-wide pool:

```text
XAUUSD SHORT + opposed:
  raw_n: 9
  effective_n: 7.3205
  mean_uniqueness: 0.8134
  raw_pnl_sum: +56.29
  weighted_pnl_sum: +53.27
  weighted_pnl_mean: +7.28

XAGUSD SHORT + opposed:
  raw_n: 6
  effective_n: 5.4497
  mean_uniqueness: 0.9083
  raw_pnl_sum: +26.55
  weighted_pnl_sum: +21.49
  weighted_pnl_mean: +3.94
```

XAUUSD had more overlap compression than XAGUSD, but both remain positive after weighting.

## Verdict

Verdict: **sample-count sanity check passes, alpha claim still unvalidated**.

What Chapter 4 supports:

- The BRK opposed bucket is not merely 15 duplicate same-symbol overlapping trades.
- Portfolio overlap reduces the opposed bucket from 15 raw trades to 12.77 effective trades.
- The BRK metals SHORT result remains positive after uniqueness weighting.

What Chapter 4 does not support:

- It does not rescue Kronos as an alpha source.
- It does not overturn the placebo failure.
- It does not overturn the incremental-test result where BRK-only features beat BRK+Kronos features.

Best current interpretation:

> BRK metals SHORT was the real weekly regime. Kronos disagreement remained associated with that regime, and sample uniqueness weighting says the observation was not purely duplicate counting. But Kronos still has not shown incremental predictive value.

## Next step

The next AFML-style experiment should add uniqueness weights into model/evaluation work:

1. Add `uniqueness` to BRK incremental study rows.
2. Compare unweighted vs uniqueness-weighted BRK-only and BRK+Kronos models.
3. Move from one week to rolling weekly validation.
4. Combine with Chapter 7-style purged/embargoed validation before trusting any model score.
