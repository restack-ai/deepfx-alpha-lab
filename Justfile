set dotenv-load := true
set shell := ["bash", "-uc"]

python := env_var_or_default("PYTHON", "python3")

default:
    @just --list

setup:
    {{python}} -m pip install -e '.[dev,ml]'

test:
    {{python}} -m pytest -q

check-clickhouse:
    @test -n "$CLICKHOUSE_HOST" || (echo "CLICKHOUSE_HOST is required. Copy .env.example to .env and fill it in."; exit 1)

afml-31: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/01_triple_barrier_xauusd_m1.py

afml-31-m5: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/01_triple_barrier_xauusd_m1.py --timeframe M5

afml-31-window start end: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/01_triple_barrier_xauusd_m1.py --start "{{start}}" --end "{{end}}"

afml-31-clean:
    rm -f data/processed/afml/ch03/exercise_3_1_xauusd_m1_labels.csv
    rm -f data/processed/afml/ch03/exercise_3_1_xauusd_m1_summary.json
    rm -f data/processed/afml/ch03/exercise_3_1_xauusd_m5_labels.csv
    rm -f data/processed/afml/ch03/exercise_3_1_xauusd_m5_summary.json

check-afml-31-output:
    @test -f data/processed/afml/ch03/exercise_3_1_xauusd_m1_labels.csv || (echo "Exercise 3.1 labels are required. Run: just afml-31"; exit 1)

afml-33: check-afml-31-output
    {{python}} studies/afml/ch03-labeling/scripts/02_vertical_barrier_zero.py

afml-33-clean:
    rm -f data/processed/afml/ch03/exercise_3_3_xauusd_m1_labels.csv
    rm -f data/processed/afml/ch03/exercise_3_3_xauusd_m1_summary.json

afml-34: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/03_ma_crossover_meta_labeling.py

afml-34-bb: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/03_ma_crossover_meta_labeling.py --primary bb-reversion

afml-34-m5: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/03_ma_crossover_meta_labeling.py --timeframe M5

afml-34-bb-m5: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/03_ma_crossover_meta_labeling.py --timeframe M5 --primary bb-reversion

afml-34-clean:
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m1_meta_labels.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m1_dataset.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m1_predictions.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m1_summary.json
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m1_bb_reversion_meta_labels.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m1_bb_reversion_dataset.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m1_bb_reversion_predictions.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m1_bb_reversion_summary.json
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m5_meta_labels.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m5_dataset.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m5_predictions.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m5_summary.json
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m5_bb_reversion_meta_labels.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m5_bb_reversion_dataset.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m5_bb_reversion_predictions.csv
    rm -f data/processed/afml/ch03/exercise_3_4_xauusd_m5_bb_reversion_summary.json

afml-monthly: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/04_monthly_diagnostics.py

afml-monthly-bb: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/04_monthly_diagnostics.py --primary bb-reversion

afml-monthly-m5: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/04_monthly_diagnostics.py --timeframe M5

afml-monthly-bb-m5: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/04_monthly_diagnostics.py --timeframe M5 --primary bb-reversion

afml-monthly-clean:
    rm -f data/processed/afml/ch03/monthly_diagnostics_*.csv
    rm -f data/processed/afml/ch03/monthly_diagnostics_*.json

afml-walk-forward: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/05_walk_forward_validation.py

afml-walk-forward-bb: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/05_walk_forward_validation.py --primary bb-reversion

afml-walk-forward-m5: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/05_walk_forward_validation.py --timeframe M5

afml-walk-forward-bb-m5: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/05_walk_forward_validation.py --timeframe M5 --primary bb-reversion

afml-walk-forward-clean:
    rm -f data/processed/afml/ch03/walk_forward_*.csv
    rm -f data/processed/afml/ch03/walk_forward_*.json

afml-tb-sweep: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/06_triple_barrier_sweep.py

afml-tb-sweep-intraday: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/06_triple_barrier_sweep.py --run-name intraday --num-days-grid "0.0208333333,0.0416666667,0.1666666667,1.0" --pt-sl-grid "0.5,0.5;1.0,1.0;1.5,1.0;1.0,1.5;1.5,1.5;2.0,1.0;1.0,2.0;2.0,2.0"

afml-tb-sweep-clean:
    rm -f data/processed/afml/ch03/triple_barrier_sweep_*.csv
    rm -f data/processed/afml/ch03/triple_barrier_sweep_*.json

afml-mtf-sweep: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/07_mtf_barrier_sweep.py

afml-mtf-ohlc-sweep: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/08_mtf_ohlc_barrier_sweep.py

afml-mtf-ohlc-sweep-m15: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/08_mtf_ohlc_barrier_sweep.py --event-timeframe M15 --path-timeframe M1 --pt-sl-grid "0.5,0.5;1.0,1.0;1.5,1.5" --num-days-grid "0.1666666667,0.3333333333,1.0"

afml-mtf-ohlc-sweep-h1: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/08_mtf_ohlc_barrier_sweep.py --event-timeframe H1 --path-timeframe M1 --pt-sl-grid "0.5,0.5;1.0,1.0;1.5,1.5" --num-days-grid "0.1666666667,0.3333333333,1.0" --min-daily-bars 12

afml-kronos-tb-labeler: check-clickhouse
    {{python}} studies/afml/ch03-labeling/scripts/09_kronos_tb_labeler_dataset.py

afml-mtf-sweep-clean:
    rm -f data/processed/afml/ch03/mtf_barrier_sweep_*.csv
    rm -f data/processed/afml/ch03/mtf_barrier_sweep_*.json
    rm -f data/processed/afml/ch03/mtf_ohlc_barrier_sweep_*.csv
    rm -f data/processed/afml/ch03/mtf_ohlc_barrier_sweep_*.json
    rm -rf data/processed/afml/ch03/kronos
