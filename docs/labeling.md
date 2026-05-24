# Lecture Notes: Labeling Methods in AFML Chapter 3

## Document Map

This document is the **methodology and design reference** for AFML Chapter 3 labeling.

The executable study code and latest experiment outputs live here:

```text
studies/afml/ch03-labeling/
```

Key files:

```text
studies/afml/ch03-labeling/README.md
studies/afml/ch03-labeling/scripts/
src/deepfx_alpha_lab/labeling/
data/processed/afml/ch03/
```

Use this split intentionally:

```text
docs/labeling.md
  -> Concepts, formulas, design rules, interpretation guidance

studies/afml/ch03-labeling/README.md
  -> Reproducible commands, latest results, diagnostics, rejected baselines
```

Current study status:

```text
Done:
- AFML 3.1 triple-barrier baseline on XAUUSD M1/M5
- AFML 3.3 vertical-barrier-zero variant
- AFML 3.4 EMA crossover meta-labeling baseline
- BB mean-reversion primary-signal check
- monthly diagnostics
- purged walk-forward validation
- triple-barrier parameter sweeps
- M5 event + M1 close-path barrier sweep
- M5 event + M1 OHLC execution-aware barrier sweep
- M15/H1 event-timeframe comparison with M1 OHLC execution-aware path

Current interpretation:
- Label geometry is usable as a research foundation.
- EMA/BB primary + RandomForest meta-labeling is rejected as an edge candidate.
- M15 event + M1 OHLC path is the strongest near-term Kronos target candidate for the H1/M15 discretionary workflow.
- H1 event + M1 OHLC path is clean but sample-starved on the current dataset, so use it as a comparison/regime target until more history is available.
```

Recommended next step:

```text
Build MVP 1: Kronos Triple Barrier Labeler on top of M15 event + M1 OHLC execution-aware labels.

Preferred first target:
- event timeframe: M15
- path timeframe: M1 OHLC
- pt/sl: [0.5, 0.5]
- vertical barrier: 8h or 1d
- ambiguity policy: sl_first

Reason:
This keeps the label close to the actual H1/M15 trading workflow while retaining roughly 2x more events than H1 labels in the current sample.
```

---

## 1. Three Labeling Methods

### 1.1 Fixed-Time Horizon Labeling

**Concept**

Fixed-time horizon labeling assigns a label to an observation by looking at the return after a fixed number of bars or a fixed time interval.

It asks:

> “After $h$ bars, did the price go up enough, down enough, or stay within a neutral zone?”

**Return definition**

$$
r_{t_{i,0}, t_{i,0}+h} = \frac{P_{t_{i,0}+h}}{P_{t_{i,0}}} - 1
$$

**Label definition**

$$
y_i =
\begin{cases}
-1, & \text{if } r_{t_{i,0}, t_{i,0}+h} < -\tau \\
0, & \text{if } |r_{t_{i,0}, t_{i,0}+h}| \le \tau \\
1, & \text{if } r_{t_{i,0}, t_{i,0}+h} > \tau
\end{cases}
$$

where:

| Symbol | Meaning                 |
| ------ | ----------------------- |
| $X_i$  | feature observation     |
| $y_i$  | label assigned to $X_i$ |
| $h$    | fixed future horizon    |
| $\tau$ | fixed return threshold  |
| $P_t$  | price at time $t$       |

**Problem**

The method is simple, but it has three major weaknesses:

| Weakness          | Explanation                                                       |
| ----------------- | ----------------------------------------------------------------- |
| Fixed threshold   | Same $\tau$ is used in both low- and high-volatility regimes      |
| Fixed horizon     | It ignores what happens before the horizon expires                |
| Path independence | It does not know whether the trade would have hit stop-loss first |

AFML criticizes this method because fixed thresholds on time bars often ignore volatility regimes and the actual price path. 

---

### 1.2 Triple-Barrier Labeling

**Concept**

Triple-barrier labeling assigns a label based on which of three barriers is touched first:

1. Upper horizontal barrier: profit-taking
2. Lower horizontal barrier: stop-loss
3. Vertical barrier: maximum holding period

It asks:

> “Which exit condition happened first?”

**Barrier setup**

For an event at $t_0$, define:

$$
\text{Upper barrier} = P_{t_0}(1 + pt \cdot \sigma_{t_0})
$$

$$
\text{Lower barrier} = P_{t_0}(1 - sl \cdot \sigma_{t_0})
$$

$$
\text{Vertical barrier} = t_0 + h
$$

where:

| Symbol         | Meaning                    |
| -------------- | -------------------------- |
| $pt$             | profit-taking multiplier   |
| $sl$             | stop-loss multiplier       |
| $\sigma_{t_0}$   | target volatility at $t_0$ |
| $h$              | maximum holding period     |

**Label definition**

$$
y_i =
\begin{cases}
1, & \text{if the upper barrier is touched first} \\
-1, & \text{if the lower barrier is touched first} \\
\text{sign}(r_{t_0,t_1}) \text{ or } 0, & \text{if the vertical barrier is touched first}
\end{cases}
$$

**Key property**

Triple-barrier labeling is **path-dependent**.

It does not only look at the final return. It checks the full price path from entry to exit.

AFML defines the method as labeling an observation according to the first of three barriers touched: profit-taking, stop-loss, or expiration. The horizontal barriers are dynamic functions of estimated volatility. 

---

### 1.3 Meta-Labeling

**Concept**

Meta-labeling is used when a primary model already decides the side of the trade.

The primary model answers:

> “Long or short?”

The meta model answers:

> “Should we take this trade or skip it?”

So meta-labeling separates:

| Model         | Responsibility                             |
| ------------- | ------------------------------------------ |
| Primary model | Predicts side: $-1$ or $1$                 |
| Meta model    | Predicts action: trade or pass, $0$ or $1$ |

**Meta-label definition**

Let the primary model produce a side:

$$
s_i \in \{-1, 1\}
$$

The realized return is adjusted by the side:

$$
r_i^{meta} = s_i \cdot r_i
$$

Then the meta-label is:

$$
y_i^{meta} =
\begin{cases}
1, & \text{if the primary signal was profitable} \\
0, & \text{otherwise}
\end{cases}
$$

**Interpretation**

A meta-label of `1` means:

> “The primary model’s signal was worth taking.”

A meta-label of `0` means:

> “The primary model’s signal should have been filtered out.”

AFML describes meta-labeling as building a secondary ML model that learns how to use a primary exogenous model; when side information is provided, the horizontal barriers no longer need to be symmetric, and the secondary model predicts whether to trade or not. 

---

## 2. Comparison Table

| Method             |                      Predicts |           Label Space | Path-Dependent? | Main Use                         |
| ------------------ | ----------------------------: | --------------------: | --------------: | -------------------------------- |
| Fixed-Time Horizon | Direction after fixed horizon |            $(-1, 0, 1)$ |              No | Basic classification             |
| Triple-Barrier     |  First touched exit condition | $(-1, 0, 1)$ or $(-1, 1)$ |             Yes | Realistic trade outcome labeling |
| Meta-Labeling      |              Whether to trade |                $(0, 1)$ |     Usually yes | Filtering primary model signals  |

---

# Practical Guidelines for AFML Exercises 3.1, 3.3, and 3.4

## Exercise 3.1: Basic Triple-Barrier Labeling Pipeline

### Objective

Build the full labeling pipeline:

```text
Dollar bars
→ CUSUM event sampling
→ Daily volatility target
→ Vertical barrier
→ Triple-barrier method
→ Final labels
```

AFML Exercise 3.1 asks you to form dollar bars, apply a symmetric CUSUM filter using daily return standard deviation as threshold, create a vertical barrier with `numDays = 1`, apply triple-barrier with `ptSl = [1,1]`, and then run `getBins`. 

---

### Step 1. Prepare price bars

Original AFML uses E-mini S&P 500 futures dollar bars.

For a first implementation, you can simplify:

```text
Input:
- timestamp
- open
- high
- low
- close
- volume
```

Recommended practical choices:

| Data maturity | Recommended bar                    |
| ------------- | ---------------------------------- |
| Beginner      | 1-minute time bars                 |
| Intermediate  | tick bars                          |
| AFML-style    | dollar bars                        |
| FX/Gold CFD   | tick/quote bars or time bars first |

For XAUUSD, start with 1-minute bars first. Dollar bars are less reliable if real traded volume is unavailable.

---

### Step 2. Compute daily volatility target

AFML uses daily volatility as a dynamic target. Conceptually:

$$
\sigma_t = EWMA(\text{daily returns})
$$

Use this as:

```python
trgt = daily_vol
```

This becomes the unit width for the horizontal barriers.

---

### Step 3. Apply CUSUM filter

The CUSUM filter selects event timestamps.

Conceptually:

```text
Do not label every bar.
Label only meaningful price movement events.
```

Input:

```python
close
threshold = daily_vol.mean() or time-varying daily_vol
```

Output:

```python
t_events
```

Each timestamp in `t_events` becomes a candidate event for triple-barrier labeling.

AFML describes CUSUM event sampling as a way to sample bars only when a full run of sufficient magnitude occurs, instead of triggering repeatedly around a noisy threshold. 

---

### Step 4. Create vertical barriers

For each event timestamp $t_0$, define:

```text
t1 = t0 + 1 day
```

In code-level terms:

```python
t1 = add_vertical_barrier(t_events, close, num_days=1)
```

This creates the expiration time for every event.

---

### Step 5. Apply triple-barrier

Use:

```python
ptSl = [1, 1]
```

Meaning:

```text
profit-taking = 1 × target volatility
stop-loss     = 1 × target volatility
```

This is symmetric.

Expected output:

```text
event start time
first touched barrier time
target volatility
label
```

---

### Step 6. Generate labels with getBins

Expected label space:

```text
-1: lower barrier touched first
 1: upper barrier touched first
 0: optional, if vertical barrier is treated as neutral
```

At this stage, first check:

```text
label distribution
number of events
average holding period
percentage of vertical-barrier exits
```

---

## Exercise 3.3: Modify getBins for Vertical Barrier = 0

### Objective

Change the labeling rule so that if the vertical barrier is touched first, the label becomes `0`.

AFML notes that when the vertical barrier is touched first, one can either use the sign of the return or assign `0`; the book leaves this adjustment as an exercise. 

---

### Default behavior

Typical `getBins` behavior:

```text
If vertical barrier is touched first:
    label = sign(return)
```

Example:

```text
No profit-taking hit
No stop-loss hit
Time expires with +0.2% return

label = 1
```

---

### Modified behavior

Exercise 3.3 behavior:

```text
If vertical barrier is touched first:
    label = 0
```

Example:

```text
No profit-taking hit
No stop-loss hit
Time expires with +0.2% return

label = 0
```

---

### Why this matters

The two interpretations are different.

| Vertical barrier treatment | Meaning                                               |
| -------------------------- | ----------------------------------------------------- |
| `sign(return)`             | Small profit/loss still counts as directional outcome |
| `0`                        | No decisive move occurred before expiration           |

For ML, the second version is often cleaner when you want the model to learn only strong directional outcomes.

---

### What to compare

Run both versions:

```text
Version A: vertical barrier → sign(return)
Version B: vertical barrier → 0
```

Compare:

| Metric                   | Question                               |
| ------------------------ | -------------------------------------- |
| Label distribution       | Did class imbalance increase?          |
| Number of 0 labels       | Did neutral events dominate?           |
| Model accuracy           | Did it improve artificially?           |
| Precision / Recall       | Did directional labels become cleaner? |
| Average return per label | Are labels economically meaningful?    |

For XAUUSD, this experiment is especially useful because many events may expire without a strong directional move.

---

## Exercise 3.4: Trend-Following Strategy + Meta-Labeling

### Objective

Build a primary trend-following model, then train a secondary ML model to decide whether to trade or not.

AFML Exercise 3.4 asks you to build a trend-following strategy such as moving-average crossover, derive meta-labels with `ptSl = [1,2]` and `numDays = 1`, and train a random forest to decide whether to trade. The primary model decides the side, while the secondary model decides trade/pass. 

---

### Step 1. Build a primary trend-following rule

Example: moving-average crossover.

$$
side_t =
\begin{cases}
1, & \text{if } MA_{fast,t} > MA_{slow,t} \\
-1, & \text{if } MA_{fast,t} < MA_{slow,t}
\end{cases}
$$

Example settings:

```text
Fast MA = 20 bars
Slow MA = 50 bars
```

For XAUUSD 1-minute bars, you can start with:

```text
EMA(20) / EMA(50)
EMA(10) / EMA(30)
```

---

### Step 2. Align primary side with event timestamps

The primary side should only be evaluated at event timestamps:

```python
side = side.reindex(t_events).dropna()
```

This gives:

```text
event timestamp → long or short
```

---

### Step 3. Build meta-labeling events

Use:

```python
ptSl = [1, 2]
t1 = vertical barrier with numDays = 1
trgt = daily volatility
side = primary model side
```

Interpretation:

```text
profit-taking = 1 × volatility
stop-loss     = 2 × volatility
```

Because side is already known, the triple-barrier method can distinguish profit-taking from stop-loss.

---

### Step 4. Generate meta-labels

The label space becomes:

```text
1: take the trade
0: skip the trade
```

Conceptually:

$$
y_i^{meta} =
\begin{cases}
1, & \text{if the primary signal led to a profitable outcome} \\
0, & \text{otherwise}
\end{cases}
$$

The secondary model is not predicting long or short.
It is predicting whether the primary model’s long/short signal is worth taking.

---

### Step 5. Prepare features for the meta model

Start simple.

Recommended feature set:

| Feature                 | Meaning                     |
| ----------------------- | --------------------------- |
| volatility              | current realized volatility |
| MA spread               | $MA_{fast} / MA_{slow} - 1$ |
| MA slope                | trend strength              |
| return over last N bars | recent momentum             |
| rolling z-score         | distance from recent mean   |
| CUSUM event direction   | event shock direction       |
| hour/session            | London / NY / Asia session  |

Example:

```text
X_meta = [
    volatility,
    fast_ma / slow_ma - 1,
    fast_ma_slope,
    recent_return_5,
    recent_return_20,
    rolling_zscore,
    session_feature
]
```

---

### Step 6. Train a Random Forest

Target:

```python
y = meta_labels["bin"]  # 0 or 1
```

Model:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=3~6,
    class_weight="balanced",
    random_state=42
)
```

Use a time-aware split:

```text
Train: older period
Test: newer period
```

Avoid random shuffle, because financial time series are ordered.

---

### Step 7. Evaluate primary vs secondary model

Primary-only baseline:

```text
Take every primary model signal.
```

Secondary model:

```text
Take only signals where meta model predicts 1.
```

Compare:

| Metric                   | Meaning                                           |
| ------------------------ | ------------------------------------------------- |
| Accuracy                 | Overall correctness                               |
| Precision                | Among selected trades, how many were good?        |
| Recall                   | Among good opportunities, how many were captured? |
| F1-score                 | Balance between precision and recall              |
| Trade count              | How aggressively the meta model filters           |
| Average return per trade | Economic quality                                  |
| Max drawdown             | Risk impact                                       |

Expected pattern:

```text
Primary model:
- higher recall
- lower precision
- more trades

Meta model:
- lower trade count
- higher precision
- possibly better F1
```

This matches the purpose of meta-labeling: filter out false positives from the primary model.

---

# Suggested Mini Project Structure

```text
notebooks/
  01_prepare_bars.ipynb
  02_daily_vol_and_cusum.ipynb
  03_triple_barrier_labels.ipynb
  04_vertical_barrier_zero_test.ipynb
  05_meta_labeling_ma_crossover.ipynb

src/
  bars.py
  volatility.py
  sampling.py
  labeling.py
  features.py
  models.py
  metrics.py

data/
  raw/
  processed/

reports/
  label_distribution.md
  meta_labeling_results.md
```

---

# Minimal Checklist

## For 3.1

```text
[ ] Prepare close price series
[ ] Compute daily volatility
[ ] Apply CUSUM filter
[ ] Add vertical barrier
[ ] Apply triple-barrier with ptSl = [1,1]
[ ] Generate labels with getBins
[ ] Inspect label distribution
```

## For 3.3

```text
[ ] Detect whether first touched barrier is vertical
[ ] Assign label = 0 for vertical barrier exits
[ ] Compare with sign(return) version
[ ] Check impact on label distribution and model metrics
```

## For 3.4

```text
[ ] Build MA crossover primary side
[ ] Align side with CUSUM events
[ ] Generate meta-labels with ptSl = [1,2]
[ ] Build meta features
[ ] Train Random Forest
[ ] Compare primary-only vs meta-filtered performance
```
