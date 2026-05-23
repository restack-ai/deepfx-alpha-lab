#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfx_alpha_lab.data.clickhouse import load_ohlcv
from deepfx_alpha_lab.labeling import (
    add_vertical_barrier,
    get_bins,
    get_daily_vol,
    get_events,
    symmetric_cusum_filter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AFML Exercise 3.1 on XAUUSD M1 bars.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--pt", type=float, default=1.0)
    parser.add_argument("--sl", type=float, default=1.0)
    parser.add_argument("--num-days", type=float, default=1.0)
    parser.add_argument("--vol-span", type=int, default=100)
    parser.add_argument("--min-ret", type=float, default=0.0)
    parser.add_argument("--min-daily-bars", type=int, default=60)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "afml" / "ch03",
    )
    return parser.parse_args()


def filter_sparse_days(frame: pd.DataFrame, min_daily_bars: int) -> pd.DataFrame:
    if min_daily_bars <= 0 or frame.empty:
        return frame

    session = pd.Series(frame.index.normalize(), index=frame.index)
    counts = session.map(session.value_counts())
    return frame.loc[counts >= min_daily_bars]


def summarize(
    bars: pd.DataFrame,
    filtered_bars: pd.DataFrame,
    daily_vol: pd.Series,
    threshold: float,
    t_events: pd.DatetimeIndex,
    events: pd.DataFrame,
    bins: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    holding_minutes = ((bins["t1"] - bins.index).dt.total_seconds() / 60.0) if not bins.empty else pd.Series(dtype=float)

    return {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start": str(filtered_bars.index.min()) if not filtered_bars.empty else None,
        "end": str(filtered_bars.index.max()) if not filtered_bars.empty else None,
        "raw_rows_after_dedup": int(len(bars)),
        "rows_after_sparse_day_filter": int(len(filtered_bars)),
        "min_daily_bars": int(args.min_daily_bars),
        "daily_vol_count": int(daily_vol.count()),
        "daily_vol_mean": float(daily_vol.mean()) if daily_vol.count() else None,
        "cusum_threshold": float(threshold),
        "cusum_events": int(len(t_events)),
        "triple_barrier_events": int(len(events)),
        "labeled_events": int(len(bins)),
        "pt_sl": [float(args.pt), float(args.sl)],
        "num_days": float(args.num_days),
        "label_distribution": {str(k): int(v) for k, v in bins["bin"].value_counts(dropna=False).sort_index().items()},
        "barrier_distribution": {str(k): int(v) for k, v in bins["type"].value_counts(dropna=False).sort_index().items()},
        "avg_holding_minutes": float(holding_minutes.mean()) if not holding_minutes.empty else None,
        "median_holding_minutes": float(holding_minutes.median()) if not holding_minutes.empty else None,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bars = load_ohlcv(args.symbol, args.timeframe, start=args.start, end=args.end)
    if bars.empty:
        raise RuntimeError(f"No bars found for {args.symbol} {args.timeframe}")

    bars = bars.loc[(bars["close"] > 0) & (bars["high"] >= bars["low"])]
    filtered_bars = filter_sparse_days(bars, args.min_daily_bars)
    close = filtered_bars["close"].dropna()
    if close.empty:
        raise RuntimeError("No usable close prices after filtering")

    daily_vol = get_daily_vol(close, span=args.vol_span).dropna()
    if daily_vol.empty:
        raise RuntimeError("Daily volatility is empty; widen the date range or lower filters")

    threshold = float(daily_vol.mean())
    t_events = symmetric_cusum_filter(close, threshold=threshold)
    t1 = add_vertical_barrier(t_events, close, num_days=args.num_days)
    events = get_events(
        close=close,
        t_events=t_events,
        pt_sl=[args.pt, args.sl],
        trgt=daily_vol,
        min_ret=args.min_ret,
        t1=t1,
    )
    bins = get_bins(close, events)

    labels = events.join(bins[["ret", "bin"]], how="inner")
    labels.index.name = "t0"

    label_path = args.output_dir / f"exercise_3_1_{args.symbol.lower()}_{args.timeframe.lower()}_labels.csv"
    summary_path = args.output_dir / f"exercise_3_1_{args.symbol.lower()}_{args.timeframe.lower()}_summary.json"

    labels.to_csv(label_path)
    summary = summarize(bars, filtered_bars, daily_vol, threshold, t_events, events, bins, args)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"labels: {label_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
