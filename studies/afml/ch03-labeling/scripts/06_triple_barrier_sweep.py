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
    parser = argparse.ArgumentParser(description="Run triple-barrier parameter sweep.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframes", default="M1,M5")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--pt-sl-grid", default="0.5,0.5;0.5,1.0;1.0,1.0;1.0,2.0;2.0,1.0")
    parser.add_argument("--num-days-grid", default="0.25,0.5,1.0")
    parser.add_argument("--run-name", default="")
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


def parse_pt_sl_grid(value: str) -> list[tuple[float, float]]:
    pairs = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(",", maxsplit=1)
        pairs.append((float(left), float(right)))
    if not pairs:
        raise ValueError("--pt-sl-grid must contain at least one pt,sl pair")
    return pairs


def parse_float_grid(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def distribution(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def count_value(series: pd.Series, value: object) -> int:
    if series.empty:
        return 0
    return int((series == value).sum())


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def mean_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.mean())


def run_timeframe(
    symbol: str,
    timeframe: str,
    args: argparse.Namespace,
    pt_sl_grid: list[tuple[float, float]],
    num_days_grid: list[float],
) -> list[dict[str, object]]:
    bars = load_ohlcv(symbol, timeframe, start=args.start, end=args.end)
    if bars.empty:
        return [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "error": "no bars found",
            }
        ]

    bars = bars.loc[(bars["close"] > 0) & (bars["high"] >= bars["low"])]
    filtered_bars = filter_sparse_days(bars, args.min_daily_bars)
    close = filtered_bars["close"].dropna()
    daily_vol = get_daily_vol(close, span=args.vol_span).dropna()
    if close.empty or daily_vol.empty:
        return [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "error": "no usable close or daily volatility",
            }
        ]

    threshold = float(daily_vol.mean())
    t_events = symmetric_cusum_filter(close, threshold=threshold)

    rows: list[dict[str, object]] = []
    for num_days in num_days_grid:
        t1 = add_vertical_barrier(t_events, close, num_days=num_days)
        for pt, sl in pt_sl_grid:
            events = get_events(
                close=close,
                t_events=t_events,
                pt_sl=[pt, sl],
                trgt=daily_vol,
                min_ret=args.min_ret,
                t1=t1,
            )
            bins = get_bins(close, events)
            labels = events.join(bins[["ret", "bin"]], how="inner")
            labels.index.name = "t0"

            labeled_events = int(len(labels))
            pos = count_value(labels["bin"], 1) if labeled_events else 0
            neg = count_value(labels["bin"], -1) if labeled_events else 0
            pt_count = count_value(labels["type"], "pt") if labeled_events else 0
            sl_count = count_value(labels["type"], "sl") if labeled_events else 0
            t1_count = count_value(labels["type"], "t1") if labeled_events else 0
            holding_minutes = (
                (labels["t1"] - labels.index).dt.total_seconds() / 60.0
                if labeled_events
                else pd.Series(dtype=float)
            )
            pos_rate = ratio(pos, labeled_events)
            balance_gap = abs(pos_rate - 0.5) if pos_rate is not None else None

            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start": str(close.index.min()),
                    "end": str(close.index.max()),
                    "rows_after_filter": int(len(filtered_bars)),
                    "daily_vol_mean": threshold,
                    "cusum_events": int(len(t_events)),
                    "pt": float(pt),
                    "sl": float(sl),
                    "num_days": float(num_days),
                    "triple_barrier_events": int(len(events)),
                    "labeled_events": labeled_events,
                    "label_neg": neg,
                    "label_pos": pos,
                    "label_pos_rate": pos_rate,
                    "label_balance_gap": balance_gap,
                    "pt_count": pt_count,
                    "sl_count": sl_count,
                    "t1_count": t1_count,
                    "pt_rate": ratio(pt_count, labeled_events),
                    "sl_rate": ratio(sl_count, labeled_events),
                    "t1_rate": ratio(t1_count, labeled_events),
                    "median_holding_minutes": float(holding_minutes.median()) if not holding_minutes.empty else None,
                    "avg_holding_minutes": float(holding_minutes.mean()) if not holding_minutes.empty else None,
                    "mean_ret": mean_or_none(labels["ret"]) if labeled_events else None,
                    "mean_ret_label_pos": mean_or_none(labels.loc[labels["bin"] == 1, "ret"]) if labeled_events else None,
                    "mean_ret_label_neg": mean_or_none(labels.loc[labels["bin"] == -1, "ret"]) if labeled_events else None,
                    "label_distribution": distribution(labels["bin"]) if labeled_events else {},
                    "barrier_distribution": distribution(labels["type"]) if labeled_events else {},
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timeframes = [item.strip() for item in args.timeframes.split(",") if item.strip()]
    pt_sl_grid = parse_pt_sl_grid(args.pt_sl_grid)
    num_days_grid = parse_float_grid(args.num_days_grid)

    rows: list[dict[str, object]] = []
    for timeframe in timeframes:
        rows.extend(run_timeframe(args.symbol, timeframe, args, pt_sl_grid, num_days_grid))

    sweep = pd.DataFrame(rows)
    start_key = pd.Timestamp(args.start).strftime("%Y%m")
    end_key = (pd.Timestamp(args.end) - pd.Timedelta(days=1)).strftime("%Y%m")
    run_suffix = f"_{args.run_name}" if args.run_name else ""
    stem = f"triple_barrier_sweep_{args.symbol.lower()}_{'_'.join(tf.lower() for tf in timeframes)}{run_suffix}_{start_key}_{end_key}"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    sweep.to_csv(csv_path, index=False)

    summary = {
        "symbol": args.symbol,
        "timeframes": timeframes,
        "start": args.start,
        "end": args.end,
        "pt_sl_grid": [[pt, sl] for pt, sl in pt_sl_grid],
        "num_days_grid": num_days_grid,
        "rows": sweep.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    display_cols = [
        "timeframe",
        "pt",
        "sl",
        "num_days",
        "labeled_events",
        "label_pos_rate",
        "label_balance_gap",
        "pt_rate",
        "sl_rate",
        "t1_rate",
        "median_holding_minutes",
    ]
    print(sweep[display_cols].to_string(index=False))
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()
