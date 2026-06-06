# BRK-only vs BRK+Kronos Incremental Study — 2026-05-31 to 2026-06-06

Research-only report. No live trading action is implied.

## Scope

This follow-up tests whether Kronos disagreement adds explanatory power beyond BRK-only context features.

The prior BRK × Kronos disagreement study found that many profitable BRK trades were `opposed` to the latest prior Kronos shadow. This study asks the harder question:

> Does adding Kronos features improve prediction/explanation compared with BRK-only features?

Analysis input:

- Source rows: `data/processed/brk_kronos_disagreement/20260531_20260606/study/aligned_trades.csv`
- Window: 120 minutes
- Rows: 21 BRK trades
- Days: 4
- Target: trade export `pnl`

Study artifacts:

- Script: `studies/afml/ch03-labeling/scripts/13_brk_kronos_incremental_study.py`
- Library: `src/deepfx_alpha_lab/kronos/brk_incremental.py`
- Generated report: `data/processed/brk_kronos_disagreement/20260531_20260606/incremental/report.md`
- Generated CSVs:
  - `feature_eval.csv`
  - `bucket_lift.csv`
  - `prediction_rows.csv`

## Method

Because this is only one week and 21 BRK trades, this study avoids pretending to train a robust ML model.

Instead it uses leave-one-day-out group-mean prediction:

1. Hold out one trading day.
2. Use other days as training data.
3. Predict held-out trade PnL using grouped historical means.
4. If a group is too sparse, back off to progressively simpler prefixes, then global mean.
5. Compare feature sets using:
   - MAE
   - RMSE
   - mean error
   - sign accuracy
   - positive precision
   - fallback-to-global rate

Feature sets compared:

```text
global_mean:
  <none>

brk_symbol_side:
  symbol, side

brk_regime:
  symbol, side, risk_regime

brk_vol:
  symbol, side, risk_regime, bb_width_bucket, atr_bucket, body_atr_bucket

brk_kronos_alignment:
  symbol, side, risk_regime, alignment

brk_kronos_full:
  symbol, side, risk_regime,
  bb_width_bucket, atr_bucket, body_atr_bucket,
  alignment, signal_age_bucket, pred_abs_bucket
```

## Result: feature-set comparison

Lower MAE/RMSE is better.

```text
brk_symbol_side:
  MAE: 13.190
  RMSE: 17.807
  sign_accuracy: 19.0%

brk_kronos_alignment:
  MAE: 13.852
  RMSE: 18.849
  sign_accuracy: 19.0%

brk_regime:
  MAE: 13.852
  RMSE: 18.849
  sign_accuracy: 19.0%

global_mean:
  MAE: 14.325
  RMSE: 18.085
  sign_accuracy: 19.0%

brk_kronos_full:
  MAE: 14.392
  RMSE: 19.581
  sign_accuracy: 19.0%

brk_vol:
  MAE: 14.392
  RMSE: 19.581
  sign_accuracy: 19.0%
```

Best MAE came from the simplest BRK-only model:

```text
symbol + side
```

Adding Kronos alignment did **not** improve leave-one-day-out performance.

In fact:

```text
brk_symbol_side MAE:       13.190
brk_kronos_alignment MAE:  13.852
brk_kronos_full MAE:       14.392
```

So Kronos features made the held-out-day error slightly worse in this sample.

## Bucket-level diagnostics

The descriptive buckets still show the original interesting pattern:

```text
XAUUSD SHORT + mixed + opposed:
  n: 5
  pnl_sum: +28.21
  pnl_mean: +5.64
  win_rate: 60.0%

XAUUSD SHORT + liquidity_selloff + opposed:
  n: 4
  pnl_sum: +28.08
  pnl_mean: +7.02
  win_rate: 75.0%

XAGUSD SHORT + mixed + opposed:
  n: 4
  pnl_sum: +20.10
  pnl_mean: +5.03
  win_rate: 75.0%
```

But those buckets mostly restate the fact that **BRK metals SHORT worked this week**.

The strongest bucket by account PnL was:

```text
XAGUSD SHORT + mixed + opposed + recent signal:
  n: 2
  pnl_sum: +41.75
  pnl_mean: +20.875
  win_rate: 100.0%
```

However, this is only two trades. It is a lead for monitoring, not a validated feature.

There was also a conflicting fresh opposed XAGUSD bucket:

```text
XAGUSD SHORT + mixed + opposed + fresh signal:
  n: 2
  pnl_sum: -21.65
  pnl_mean: -10.825
  win_rate: 50.0%
```

So even within the attractive XAGUSD opposed theme, signal age may matter — but the sample is too small to separate it cleanly.

## Interpretation

This study weakens the Kronos-disagreement edge claim.

The current best interpretation is:

> The week contained a BRK metals SHORT opportunity. Kronos disagreement highlighted many of those trades descriptively, but did not improve leave-one-day-out explanatory performance beyond BRK-only symbol/side context.

This is exactly the kind of result worth catching early. It prevents us from turning a seductive chart pattern into a false feature.

## Verdict

Verdict: **Kronos incremental edge not validated**.

What validated:

- The incremental test pipeline is now reusable.
- BRK-only vs BRK+Kronos feature comparisons can be run from the generated `aligned_trades.csv`.
- Descriptive opposed buckets remain worth monitoring, especially metals SHORT.

What failed:

- `BRK+Kronos alignment` did not beat `BRK symbol+side` in leave-one-day-out MAE or RMSE.
- `BRK+Kronos full` performed worse than simpler feature sets.
- Sign accuracy was poor and unchanged across feature sets.

What this means:

- Do not add a live rule that blindly rewards Kronos disagreement.
- Treat Kronos disagreement as a monitoring/research feature only.
- The next useful work is expanding the evaluation window and testing whether the same pattern repeats out-of-sample.

## Next recommended experiment

Move from one-week discovery to rolling validation:

1. Export multiple weeks of BRK trades, Kronos shadows, and M5 OHLCV.
2. Run the same disagreement and incremental scripts per week.
3. Aggregate by week:
   - BRK-only baseline MAE/RMSE
   - BRK+Kronos MAE/RMSE
   - opposed bucket PnL
   - placebo p-values
4. Require Kronos features to improve over BRK-only baselines on multiple held-out weeks before considering live usage.

The minimum next bar:

```text
Kronos features must beat BRK-only symbol+side/regime baselines
on at least two independent weekly windows,
and direction-shuffle placebo must not explain the lift.
```
