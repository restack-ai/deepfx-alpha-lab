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
from deepfx_alpha_lab.kronos import build_kronos_label_dataset
from deepfx_alpha_lab.labeling import add_vertical_barrier, get_daily_vol, resolve_ohlc_events, symmetric_cusum_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kronos-ready triple-barrier label dataset.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--event-timeframe", default="M15")
    parser.add_argument("--path-timeframe", default="M1")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--pt", type=float, default=0.5)
    parser.add_argument("--sl", type=float, default=0.5)
    parser.add_argument("--num-days", type=float, default=0.3333333333, help="Vertical barrier in days; 0.3333333333 ~= 8h")
    parser.add_argument("--lookback", type=int, default=96, help="Number of event-timeframe bars in each Kronos window")
    parser.add_argument("--vol-span", type=int, default=100)
    parser.add_argument("--min-ret", type=float, default=0.0)
    parser.add_argument("--min-daily-bars", type=int, default=60)
    parser.add_argument("--ambiguous-policy", choices=["sl_first", "pt_first", "ambiguous"], default="sl_first")
    parser.add_argument("--run-name", default="m15_m1_ohlc_pt05_sl05_8h")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "afml" / "ch03" / "kronos",
    )
    return parser.parse_args()


def filter_sparse_days(frame: pd.DataFrame, min_daily_bars: int) -> pd.DataFrame:
    if min_daily_bars <= 0 or frame.empty:
        return frame
    session = pd.Series(frame.index.normalize(), index=frame.index)
    counts = session.map(session.value_counts())
    return frame.loc[counts >= min_daily_bars]


def distribution(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    event_bars = load_ohlcv(args.symbol, args.event_timeframe, start=args.start, end=args.end)
    path_bars = load_ohlcv(args.symbol, args.path_timeframe, start=args.start, end=args.end)
    if event_bars.empty or path_bars.empty:
        raise RuntimeError("Event or path bars are empty")

    event_bars = event_bars.loc[(event_bars["close"] > 0) & (event_bars["high"] >= event_bars["low"])]
    path_bars = path_bars.loc[(path_bars["close"] > 0) & (path_bars["high"] >= path_bars["low"])]
    event_bars = filter_sparse_days(event_bars, args.min_daily_bars)
    path_bars = filter_sparse_days(path_bars, args.min_daily_bars)

    event_close = event_bars["close"].dropna()
    daily_vol = get_daily_vol(event_close, span=args.vol_span).dropna()
    if event_close.empty or path_bars.empty or daily_vol.empty:
        raise RuntimeError("No usable event close, path bars, or daily volatility")

    threshold = float(daily_vol.mean())
    t_events = symmetric_cusum_filter(event_close, threshold=threshold)
    t1 = add_vertical_barrier(t_events, path_bars["close"], num_days=args.num_days)
    labels = resolve_ohlc_events(
        entry_close=event_close,
        path_bars=path_bars,
        t_events=t_events,
        t1=t1,
        trgt=daily_vol,
        pt=args.pt,
        sl=args.sl,
        min_ret=args.min_ret,
        ambiguous_policy=args.ambiguous_policy,
    )
    if labels.empty:
        raise RuntimeError("No labels generated")

    dataset = build_kronos_label_dataset(event_bars, labels, lookback=args.lookback)
    start_key = pd.Timestamp(args.start).strftime("%Y%m")
    end_key = (pd.Timestamp(args.end) - pd.Timedelta(days=1)).strftime("%Y%m")
    stem = f"kronos_tb_labeler_{args.symbol.lower()}_{args.run_name}_{start_key}_{end_key}"
    paths = dataset.save(args.output_dir, stem)
    labels_path = args.output_dir / f"{stem}_labels.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    labels.to_csv(labels_path)

    label_type_series = pd.Series(dataset.y_type).map({0: "pt", 1: "sl", 2: "t1", 3: "ambiguous"})
    summary = {
        "symbol": args.symbol,
        "event_timeframe": args.event_timeframe,
        "path_timeframe": args.path_timeframe,
        "path_mode": "ohlc",
        "ambiguous_policy": args.ambiguous_policy,
        "start": str(event_bars.index.min()),
        "end": str(event_bars.index.max()),
        "event_rows": int(len(event_bars)),
        "path_rows": int(len(path_bars)),
        "daily_vol_mean": threshold,
        "cusum_events": int(len(t_events)),
        "labels": int(len(labels)),
        "dataset_rows": int(len(dataset.y_type)),
        "skipped_for_lookback": int(len(labels) - len(dataset.y_type)),
        "lookback": int(args.lookback),
        "feature_columns": dataset.feature_columns,
        "x_shape": list(dataset.x.shape),
        "pt_sl": [float(args.pt), float(args.sl)],
        "num_days": float(args.num_days),
        "label_type_distribution": distribution(label_type_series),
        "bin_distribution": distribution(pd.Series(dataset.y_bin)),
        "ret_mean": float(dataset.y_ret.mean()) if len(dataset.y_ret) else None,
        "ret_std": float(dataset.y_ret.std()) if len(dataset.y_ret) else None,
        "outputs": {"npz": str(paths["npz"]), "metadata": str(paths["metadata"]), "labels": str(labels_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"npz: {paths['npz']}")
    print(f"metadata: {paths['metadata']}")
    print(f"labels: {labels_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
