# DeepFX Alpha Lab

AI-driven FX research, validation, and alpha discovery framework.

`deepfx-alpha-lab` is a research-oriented repository for validating AI-based FX signals, especially Kronos shadow signals, under realistic trading assumptions.

The goal is not simply to ask:

> “Does Kronos predict the market correctly?”

Instead, the real research question is:

> “Under which market conditions, holding windows, cost structures, and risk constraints does Kronos provide a repeatable edge that can be used as a practical trade filter?”

---

## 1. Project Purpose

DeepFX Alpha Lab focuses on building a disciplined research loop for AI-assisted FX trading.

The repository is designed to help answer questions such as:

- Did the signal reach take-profit before stop-loss?
- Did the signal survive spread, slippage, and transaction costs?
- Does the edge remain stable across sessions, symbols, and volatility regimes?
- Is the observed performance statistically meaningful?
- Is the strategy overfitted to a specific window or parameter set?
- Can a meta-model decide when to trade or skip a Kronos signal?

---

## 2. Core Philosophy

Most trading research fails not because the model is weak, but because the validation process is contaminated.

This repository prioritizes:

1. Realistic labeling
2. Proper financial cross-validation
3. Backtest overfitting prevention
4. Cost-aware evaluation
5. Regime/session-based performance analysis
6. Meta-labeling for trade/no-trade filtering

In other words:

```text
Signal
→ Label
→ Validate
→ Penalize overfitting
→ Meta-filter
→ Deploy
````

---

## 3. Key Concepts

### 3.1 Triple Barrier Labeling

Instead of labeling a signal by simple next-candle direction, DeepFX Alpha Lab uses the Triple Barrier Method.

For each signal timestamp `t0`, three barriers are defined:

* Upper barrier: take-profit
* Lower barrier: stop-loss
* Vertical barrier: maximum holding time

The final label is determined by which barrier is reached first.

Example labels:

| Label | Meaning                          |
| ----- | -------------------------------- |
| `+1`  | Take-profit reached first        |
| `-1`  | Stop-loss reached first          |
| `0`   | Time-out or inconclusive outcome |

This is more realistic than simple direction prediction because it reflects the actual trading structure of TP, SL, and holding window.

---

### 3.2 Meta-labeling

Kronos can be treated as a primary signal generator.

Example:

```text
Kronos says LONG
```

A secondary model then decides:

```text
Should we actually take this LONG trade?
```

The meta-labeling model focuses on trade filtering:

| Primary Model               | Meta Model                  |
| --------------------------- | --------------------------- |
| Predicts direction          | Decides trade/no-trade      |
| Generates LONG/SHORT signal | Filters low-quality signals |
| Captures market opinion     | Controls execution quality  |

Candidate meta-model features:

* Kronos confidence
* Realized volatility
* Spread proxy
* Trading session: Asia / London / New York
* Recent trend or reversal strength
* Agreement with existing strategy signal
* Recent loss state
* News-like spike condition
* Volatility bucket
* Regime classification

---

### 3.3 Purged Cross-Validation and Embargo

Financial time series data is highly vulnerable to leakage.

DeepFX Alpha Lab uses finance-specific validation methods such as:

* Purged K-Fold Cross-Validation
* Embargo
* Walk-forward validation
* Combinatorial Purged Cross-Validation

These methods help prevent future information from leaking into the training set.

---

### 3.4 Deflated Sharpe Ratio

A good Sharpe ratio can be misleading when many strategies, thresholds, symbols, or time windows are tested.

The Deflated Sharpe Ratio helps answer:

> “Is this Sharpe ratio still meaningful after accounting for selection bias, non-normality, and multiple testing?”

This is important because the best-looking backtest may simply be the result of overfitting.

---

### 3.5 Probability of Backtest Overfitting

When many parameter combinations are tested, the selected best strategy often fails out-of-sample.

DeepFX Alpha Lab uses the Probability of Backtest Overfitting concept to estimate whether an apparent edge is robust or just a lucky artifact.

---

## 4. Initial MVP

The first MVP focuses on Kronos shadow signal validation.

### MVP Pipeline

```text
Kronos shadow signal
→ Triple barrier label
→ Session / regime / volatility bucket analysis
→ Purged walk-forward validation
→ DSR / PBO overfitting penalty
→ Meta-label model
→ Trade / no-trade decision
```

### MVP Tasks

1. Collect Kronos shadow signals
2. Attach triple-barrier labels to each signal
3. Analyze performance by:

   * Symbol
   * Session
   * Volatility bucket
   * Market regime
   * Holding window
   * Signal confidence
4. Compare:

   * Kronos-only signal
   * Existing strategy-only signal
   * Kronos + existing strategy agreement
   * Kronos disagreement filter
5. Apply purged walk-forward validation
6. Evaluate DSR and PBO
7. Train a simple meta-labeling model for trade/no-trade filtering

---

## 5. Suggested Repository Structure

```text
deepfx-alpha-lab/
├── README.md
├── docs/
│   ├── research-notes.md
│   ├── labeling.md
│   ├── validation.md
│   ├── backtesting-protocol.md
│   └── references.md
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 001_triple_barrier_labeling.ipynb
│   ├── 002_kronos_shadow_analysis.ipynb
│   ├── 003_purged_walk_forward_validation.ipynb
│   └── 004_meta_labeling_experiment.ipynb
├── src/
│   └── deepfx_alpha_lab/
│       ├── __init__.py
│       ├── labeling/
│       │   ├── triple_barrier.py
│       │   └── meta_labeling.py
│       ├── validation/
│       │   ├── purged_kfold.py
│       │   ├── embargo.py
│       │   └── walk_forward.py
│       ├── metrics/
│       │   ├── sharpe.py
│       │   ├── deflated_sharpe.py
│       │   └── pbo.py
│       ├── backtesting/
│       │   ├── cost_model.py
│       │   ├── slippage.py
│       │   └── evaluator.py
│       ├── kronos/
│       │   ├── signal_loader.py
│       │   └── shadow_log.py
│       └── research/
│           ├── regime.py
│           ├── session.py
│           └── volatility.py
├── tests/
│   ├── test_triple_barrier.py
│   ├── test_purged_kfold.py
│   └── test_metrics.py
├── scripts/
│   ├── label_kronos_signals.py
│   ├── run_validation.py
│   └── generate_report.py
└── pyproject.toml
```

---

## 6. Research References

### Must-read

1. Marcos López de Prado
   *Advances in Financial Machine Learning*

   Key topics:

   * Triple Barrier Labeling
   * Meta-labeling
   * Purged K-Fold Cross-Validation
   * Embargo
   * Sequential Bootstrap
   * Fractional Differentiation

2. Bailey & López de Prado
   *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*

3. Bailey, Borwein, López de Prado, Zhu
   *The Probability of Backtest Overfitting*

4. Bailey & López de Prado
   *A Backtesting Protocol in the Era of Machine Learning*

---

## 7. Useful Open Source References

These projects are useful as design references, but their licenses and production suitability should be reviewed carefully before direct integration.

### mlfinlab

* Repository: [https://github.com/hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab)
* Useful for:

  * Labeling
  * Sampling
  * Cross-validation
  * Backtest statistics

### qlib

* Repository: [https://github.com/microsoft/qlib](https://github.com/microsoft/qlib)
* Useful for:

  * Quant research workflow
  * Model prediction pipeline
  * Backtesting architecture

### vectorbt

* Repository: [https://github.com/polakowo/vectorbt](https://github.com/polakowo/vectorbt)
* Useful for:

  * Fast vectorized backtesting
  * Parameter sweep experiments

### backtesting.py

* Repository: [https://github.com/kernc/backtesting.py](https://github.com/kernc/backtesting.py)
* Useful for:

  * Simple strategy testing
  * Lightweight research prototypes

### backtrader

* Repository: [https://github.com/mementum/backtrader](https://github.com/mementum/backtrader)
* Useful for:

  * Event-driven backtesting concepts
  * Classic strategy research patterns

### zipline

* Repository: [https://github.com/quantopian/zipline](https://github.com/quantopian/zipline)
* Useful for:

  * Event-driven research design
  * Historical backtesting architecture

### FinRL

* Repository: [https://github.com/AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)
* Useful for:

  * Future reinforcement learning experiments
  * Action policy optimization

---

## 8. Weekend Research Plan

### 1-hour track

Goal: decide how to label Kronos shadow logs.

* Understand Triple Barrier Method
* Understand Meta-labeling
* Understand Purged K-Fold and Embargo

### 3-hour track

Goal: understand how to avoid statistical self-deception.

* Read AFML chapters on labeling and cross-validation
* Skim Deflated Sharpe Ratio paper
* Review mlfinlab labeling and validation APIs

### 1-day track

Goal: build the first DeepFX research loop.

* Design triple-barrier labeler
* Attach labels to Kronos shadow signals
* Analyze performance by session, symbol, and volatility bucket
* Define purged split criteria
* Compare Kronos-only vs strategy-agreement filters
* Prepare first validation report

---

## 9. Current Research Question

DeepFX Alpha Lab starts from this question:

> Does Kronos provide a repeatable and cost-adjusted trading edge under specific market regimes, sessions, and holding windows?

Not:

> Does Kronos simply predict the next price direction?

This distinction matters because:

* The direction may be correct, but stop-loss may be hit first.
* The trade may reach take-profit, but spread and slippage may erase the edge.
* The signal may work only in a specific session.
* The signal may work only in a specific volatility regime.
* The best threshold may be overfitted.
* A high Sharpe ratio may be the result of multiple testing.

---

## 10. Roadmap

### Phase 1 — Labeling

* [ ] Define signal schema
* [ ] Implement triple-barrier labeling
* [ ] Add TP / SL / holding-window configuration
* [ ] Generate labeled Kronos shadow dataset

### Phase 2 — Exploratory Analysis

* [ ] Analyze win rate by session
* [ ] Analyze expectancy by volatility bucket
* [ ] Analyze signal quality by confidence score
* [ ] Compare symbol-level performance
* [ ] Identify unstable regimes

### Phase 3 — Validation

* [ ] Implement purged K-Fold validation
* [ ] Implement embargo logic
* [ ] Implement walk-forward validation
* [ ] Add Deflated Sharpe Ratio
* [ ] Add Probability of Backtest Overfitting analysis

### Phase 4 — Meta-labeling

* [ ] Define trade/no-trade target
* [ ] Build baseline meta-labeling dataset
* [ ] Train simple classifier
* [ ] Compare raw Kronos vs meta-filtered Kronos
* [ ] Evaluate out-of-sample robustness

### Phase 5 — Reporting

* [ ] Generate research report
* [ ] Summarize edge by regime/session/symbol
* [ ] Export experiment results
* [ ] Document reproducibility rules

---

## 11. Disclaimer

This repository is for research and engineering purposes only.

It does not provide financial advice, investment recommendations, or guaranteed trading strategies.
All results must be interpreted with caution and validated under realistic market assumptions, including spread, slippage, latency, transaction costs, and execution risk.
