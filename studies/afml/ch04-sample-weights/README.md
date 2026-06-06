# AFML Chapter 4 Sample Weights Mini Project

This mini project applies AFML Chapter 4 sample weighting ideas to DeepFX trade/event intervals.

The first pass focuses on **sample uniqueness** for live BRK trades:

```text
entry_time / exit_time intervals
-> interval concurrency
-> average uniqueness = duration-weighted mean(1 / concurrency)
-> effective sample size = sum(uniqueness)
-> raw vs uniqueness-weighted PnL summaries
```

## Why this matters

A raw count of trades can overstate evidence when trades overlap in time or share the same market regime. If 15 trades occur during one sell wave, they may not represent 15 independent observations.

Chapter 4 gives us a way to ask:

```text
How many independent samples do these trades really represent?
```

## Study 01: BRK trade interval uniqueness

Script:

```bash
uv run --extra dev python studies/afml/ch04-sample-weights/scripts/01_trade_uniqueness.py \
  --events data/processed/brk_kronos_disagreement/20260531_20260606/study/aligned_trades.csv \
  --out-dir data/processed/afml/ch04/brk_uniqueness_portfolio_20260531_20260606 \
  --window-minutes 120 \
  --family BRK \
  --pool portfolio
```

For same-symbol concurrency only:

```bash
uv run --extra dev python studies/afml/ch04-sample-weights/scripts/01_trade_uniqueness.py \
  --events data/processed/brk_kronos_disagreement/20260531_20260606/study/aligned_trades.csv \
  --out-dir data/processed/afml/ch04/brk_uniqueness_symbol_20260531_20260606 \
  --window-minutes 120 \
  --family BRK \
  --pool symbol
```

Default outputs:

```text
data/processed/afml/ch04/<study>/weighted_events.csv
data/processed/afml/ch04/<study>/summary_family.csv
data/processed/afml/ch04/<study>/summary_family__alignment.csv
data/processed/afml/ch04/<study>/summary_symbol__side.csv
data/processed/afml/ch04/<study>/summary_symbol__side__alignment.csv
data/processed/afml/ch04/<study>/summary_risk_regime__alignment.csv
data/processed/afml/ch04/<study>/report.md
```

## Latest result: 2026-05-31 to 2026-06-06 BRK trades

Input:

- Source: `data/processed/brk_kronos_disagreement/20260531_20260606/study/aligned_trades.csv`
- Window: 120-minute BRK/Kronos alignment rows
- Raw BRK trades: 21

### Same-symbol pool

When concurrency is computed separately per symbol:

```text
raw_n: 21
effective_n: 21.0000
effective/raw ratio: 1.0000
mean_uniqueness: 1.0000
mean_concurrency: 1.0000
```

Interpretation:

```text
Within the same symbol, BRK trades did not materially overlap.
The raw trade count is not inflated by same-symbol overlapping intervals.
```

### Portfolio pool

When all BRK trades share one portfolio-wide concurrency pool:

```text
raw_n: 21
effective_n: 18.4137
effective/raw ratio: 0.8768
mean_uniqueness: 0.8768
mean_concurrency: 1.2463
```

Interpretation:

```text
Portfolio-wide overlap compresses the sample by about 12.3%.
This is meaningful but not severe.
The week was not just one massively duplicated trade cluster.
```

### Alignment split under portfolio pooling

```text
opposed:
  raw_n: 15
  effective_n: 12.7702
  mean_uniqueness: 0.8513
  raw_pnl_sum: +82.84
  weighted_pnl_sum: +74.76
  raw_pnl_mean: +5.52
  weighted_pnl_mean: +5.85

no_recent_kronos:
  raw_n: 5
  effective_n: 4.6435
  mean_uniqueness: 0.9287
  raw_pnl_sum: +1.12
  weighted_pnl_sum: +1.09

aligned:
  raw_n: 1
  effective_n: 1.0000
  raw_pnl_sum: -1.17
```

Interpretation:

```text
The opposed bucket shrinks from 15 raw trades to 12.77 effective trades.
The Kronos-opposed descriptive pattern survives sample uniqueness weighting,
but Chapter 4 does not solve the earlier placebo/incremental-test failures.
```

### Symbol-side split under portfolio pooling

```text
XAUUSD SHORT + opposed:
  raw_n: 9
  effective_n: 7.3205
  raw_pnl_sum: +56.29
  weighted_pnl_sum: +53.27
  weighted_pnl_mean: +7.28

XAGUSD SHORT + opposed:
  raw_n: 6
  effective_n: 5.4497
  raw_pnl_sum: +26.55
  weighted_pnl_sum: +21.49
  weighted_pnl_mean: +3.94
```

Interpretation:

```text
XAUUSD opposed had more portfolio-overlap compression than XAGUSD,
but both remain positive after uniqueness weighting.
```

## Verdict

Chapter 4 uniqueness weighting gives a useful sanity check:

```text
The BRK/Kronos opposed result was not purely an artifact of many same-symbol overlapping trades.
Portfolio-wide overlap reduces effective N, but only moderately.
```

However:

```text
This does not validate Kronos as an alpha source.
The Ch03 follow-up still stands: Kronos features did not beat BRK-only baselines,
and placebo checks did not confirm the observed opposed bucket.
```

Best current interpretation:

```text
BRK metals SHORT worked during this week.
Kronos disagreement remained descriptively associated with that move,
but not yet proven as incremental signal.
```

## Next steps

1. Extend Ch04 weighting to multi-week rolling exports.
2. Add uniqueness-weighted metrics to the BRK-only vs BRK+Kronos incremental study.
3. For model training, pass uniqueness weights as `sample_weight` and compare against unweighted models under purged/embargoed validation.
4. Later Ch04 work: sequential bootstrap for ensemble training.
