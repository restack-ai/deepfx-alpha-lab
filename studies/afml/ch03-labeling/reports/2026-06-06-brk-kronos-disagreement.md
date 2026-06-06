# BRK × Kronos Disagreement Study — 2026-05-31 to 2026-06-06

Research-only report. No live trading action is implied.

## Scope

This study evaluates whether live `BRK` entries performed differently when the latest **prior** same-symbol Kronos shadow signal disagreed with the trade direction.

Analysis window:

- Start: `2026-05-31 00:00 UTC`
- End: `2026-06-06 12:00 UTC`

Inputs:

- Live trades: 25
- BRK trades: 21
- Kronos shadow signals: 566 raw, 556 after `strongest_abs_pred` conflict policy
- M5 OHLCV rows: 5,575
- Symbols represented in the OHLCV feature set: metals plus index proxies such as `NAS100` and `US30`

Study artifacts:

- Script: `studies/afml/ch03-labeling/scripts/12_brk_kronos_disagreement_study.py`
- Library: `src/deepfx_alpha_lab/kronos/brk_disagreement.py`
- Generated report: `data/processed/brk_kronos_disagreement/20260531_20260606/study/report.md`
- Generated summaries: `data/processed/brk_kronos_disagreement/20260531_20260606/study/*.csv`

## Hypothesis

The useful hypothesis is **not** "invert Kronos".

A safer framing:

> BRK entries that occur shortly after an opposite Kronos shadow signal may be capturing breakout continuation after a failed counter-flow forecast.

In practical terms, the interesting pattern is:

```text
Kronos LONG shadow -> BRK SHORT entry -> continuation lower
```

This would make Kronos disagreement a possible *failed bounce / trapped liquidity* marker, not a standalone trading signal.

## Method

For each BRK entry:

1. Use only same-symbol Kronos shadow signals.
2. Require `signal_time <= entry_time`; no future signal is allowed.
3. Attach the latest prior signal within each lookback window:
   - 15m
   - 30m
   - 60m
   - 120m
   - 180m
   - 240m
4. Bucket the trade as:
   - `aligned`: Kronos direction equals BRK trade side
   - `opposed`: Kronos direction differs from BRK trade side
   - `no_recent_kronos`: no eligible prior Kronos shadow
5. Summarize:
   - account PnL from the trade export (`pnl` column)
   - R multiple where entry/exit/stop data is available
   - ATR-normalized return using M5 OHLCV
   - symbol/side split
   - volatility compression/expansion flag
   - cross-asset risk regime
   - placebo/leakage checks

## 1. Latest-prior Kronos window sweep

The 60m, 120m, 180m, and 240m windows converged to the same split:

```text
opposed:
  trades: 15
  pnl_sum: +82.84
  pnl_mean: +5.52
  pnl_median: +5.55
  win_rate: 73.3%
  r_mean: +0.217
  atr_norm_mean: +0.594

no_recent_kronos:
  trades: 5
  pnl_sum: +1.12
  pnl_mean: +0.224
  pnl_median: +0.24
  win_rate: 80.0%
  r_mean: +0.226
  atr_norm_mean: +0.584

aligned:
  trades: 1
  pnl_sum: -1.17
  pnl_mean: -1.17
  win_rate: 0.0%
  r_mean: -0.062
  atr_norm_mean: -0.133
```

Surface read:

- Opposed BRK trades captured most of the account PnL.
- Aligned evidence is too thin because there was only one aligned BRK trade.
- `no_recent_kronos` had similar R/ATR-normalized averages, so the account-PnL gap may be partly trade clustering or sizing/instrument effects rather than pure Kronos explanatory power.

## 2. Risk-normalized PnL

Risk-normalized metrics are less dramatic than account PnL.

At the stable 60m+ windows:

```text
opposed:
  pnl_mean: +5.52
  r_mean: +0.217
  atr_norm_mean: +0.594

no_recent_kronos:
  pnl_mean: +0.224
  r_mean: +0.226
  atr_norm_mean: +0.584

aligned:
  pnl_mean: -1.17
  r_mean: -0.062
  atr_norm_mean: -0.133
```

Interpretation:

- Opposed trades carried the large account-PnL contribution.
- But R and ATR-normalized averages are close between `opposed` and `no_recent_kronos`.
- This weakens the claim that Kronos disagreement itself improves trade quality.
- It strengthens the need for a baseline-vs-incremental test: BRK-only features first, then add Kronos features.

## 3. Metals direction asymmetry

The strongest visible pattern is in metals SHORT continuation.

At 60m+ windows:

```text
XAUUSD SHORT + opposed:
  trades: 9
  pnl_sum: +56.29
  pnl_mean: +6.25
  win_rate: 66.7%
  r_mean: +0.272
  atr_norm_mean: +0.581

XAGUSD SHORT + opposed:
  trades: 6
  pnl_sum: +26.55
  pnl_mean: +4.43
  win_rate: 83.3%
  r_mean: +0.135
  atr_norm_mean: +0.613
```

This is the most interesting financial-engineering lead:

> In this week, Kronos often emitted a LONG shadow shortly before profitable BRK SHORT continuation trades in XAUUSD and XAGUSD.

A plausible interpretation is failed counter-flow:

```text
Kronos detects bounce/reversion pressure,
but BRK confirms that sellers overwhelm it,
so the failed bounce becomes continuation fuel.
```

Still, this is a hypothesis, not an edge.

## 4. Volatility compression → expansion

The first boolean compression-expansion rule was too strict:

```text
bb_width_pct <= 0.35 and body_atr >= 0.75
```

All BRK rows landed in:

```text
compression_expansion = False
```

Conclusion:

- The current binary flag is not useful enough.
- The next iteration should replace it with quantile buckets:
  - `bb_width_pct`: low / mid / high
  - `atr_pct`: low / mid / high
  - `body_atr`: low / mid / high
  - possibly previous-window compression versus current-window expansion, rather than same-bar only

## 5. Cross-asset / risk-regime split

At 60m+ windows, opposed BRK trades split mostly into liquidity selloff and mixed regimes:

```text
opposed + liquidity_selloff:
  trades: 5
  pnl_sum: +33.63
  pnl_mean: +6.73
  win_rate: 80.0%
  r_mean: +0.289
  atr_norm_mean: +0.662

opposed + mixed:
  trades: 9
  pnl_sum: +48.31
  pnl_mean: +5.37
  win_rate: 66.7%
  r_mean: +0.199
  atr_norm_mean: +0.614

opposed + risk_on:
  trades: 1
  pnl_sum: +0.90
```

This fits the qualitative observation that the week had broad sell pressure in metals/index flows. The pattern looks less like classic gold safe-haven behavior and more like a **liquidity selloff / mixed sell regime** where BRK SHORT continuation was rewarded.

## 6. Placebo and leakage checks

The placebo checks are the main reason not to overclaim.

Observed at 120m:

```text
observed opposed:
  trades: 15
  pnl_sum: +82.84
  pnl_mean: +5.52
```

Direction-shuffle null:

```text
direction_shuffle_null opposed_pnl_mean: +5.88
p_ge_observed: 0.602
```

Time shifts:

```text
time_shift_plus_1d opposed_pnl_mean: +2.88
time_shift_minus_1d opposed_pnl_mean: +9.43
```

Interpretation:

- The observed opposed mean does not beat the direction-shuffle null.
- A minus-one-day time shift can look even better than the observed alignment.
- Therefore the study does **not** validate Kronos disagreement as a standalone edge.

## Verdict

Verdict: **PARTIAL, not validated as an edge yet**.

What worked:

- The reproducible study pipeline exists.
- The prior-only BRK × Kronos join works across 15m to 240m windows.
- The week shows a clear concentration of profit in `BRK SHORT + opposed Kronos`, especially XAUUSD and XAGUSD.

What did not validate:

- Placebo tests do not support calling this a Kronos-disagreement edge.
- The compression-expansion feature is currently too blunt.
- The sample is one week and only 21 BRK trades.

Best current interpretation:

> This week probably contained a BRK SHORT regime edge. Kronos LONG disagreement may be a failed-bounce marker, but we have not proven that it adds explanatory power beyond BRK symbol/side/regime features.

## Next experiment

Run an incremental explanatory-power test:

1. Build a BRK-only baseline:
   - symbol
   - side
   - family/reason-derived BRK features
   - risk regime
   - volatility buckets
   - day/time features
2. Add Kronos features:
   - has prior Kronos
   - aligned/opposed
   - signal age
   - predicted_bps
   - direction
3. Compare baseline vs baseline+Kronos on:
   - mean out-of-sample PnL by bucket
   - win-rate lift
   - simple classifier/logistic metrics if the sample permits
   - walk-forward or leave-one-day-out stability
4. Keep the bar high: Kronos features must improve over BRK-only baselines and survive placebo checks before being treated as an edge.
