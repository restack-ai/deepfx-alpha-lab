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
from deepfx_alpha_lab.labeling import add_vertical_barrier, get_daily_vol, symmetric_cusum_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M5 event + M1 path triple-barrier sweep.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--event-timeframe", default="M5")
    parser.add_argument("--path-timeframe", default="M1")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--pt-sl-grid", default="0.5,0.5;1.0,1.0;1.5,1.5;2.0,2.0")
    parser.add_argument("--num-days-grid", default="0.1666666667,1.0")
    parser.add_argument("--vol-span", type=int, default=100)
    parser.add_argument("--min-ret", type=float, default=0.0)
    parser.add_argument("--min-daily-bars", type=int, default=60)
    parser.add_argument("--run-name", default="close_path")
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


def resolve_mtf_events(
    entry_close: pd.Series,
    path_close: pd.Series,
    t_events: pd.DatetimeIndex,
    t1: pd.Series,
    trgt: pd.Series,
    pt: float,
    sl: float,
    min_ret: float,
) -> pd.DataFrame:
    trgt = trgt.reindex(t_events).dropna()
    trgt = trgt[trgt > min_ret]
    t1 = t1.reindex(trgt.index).dropna()
    event_index = trgt.index.intersection(t1.index).intersection(entry_close.index)
    rows: list[dict[str, object]] = []

    for t0 in event_index:
        entry_price = float(entry_close.loc[t0])
        target = float(trgt.loc[t0])
        vertical_time = t1.loc[t0]
        path = path_close.loc[t0:vertical_time].dropna()
        if path.empty:
            continue

        path_returns = path / entry_price - 1.0
        pt_touch = path_returns[path_returns > pt * target]
        sl_touch = path_returns[path_returns < -sl * target]
        candidates = {
            "pt": pt_touch.index[0] if not pt_touch.empty else pd.NaT,
            "sl": sl_touch.index[0] if not sl_touch.empty else pd.NaT,
            "t1": vertical_time,
        }
        first_touch = min(value for value in candidates.values() if pd.notna(value))
        touch_type = next(name for name, value in candidates.items() if pd.notna(value) and value == first_touch)
        ret = float(path_close.loc[first_touch] / entry_price - 1.0)
        rows.append(
            {
                "t0": t0,
                "t1": first_touch,
                "trgt": target,
                "type": touch_type,
                "ret": ret,
                "bin": int(1 if ret > 0 else -1 if ret < 0 else 0),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["t1", "trgt", "type", "ret", "bin"])
    return pd.DataFrame(rows).set_index("t0").sort_index()


def summarize(labels: pd.DataFrame, metadata: dict[str, object], pt: float, sl: float, num_days: float) -> dict[str, object]:
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
    return {
        **metadata,
        "pt": float(pt),
        "sl": float(sl),
        "num_days": float(num_days),
        "labeled_events": labeled_events,
        "label_neg": neg,
        "label_pos": pos,
        "label_pos_rate": pos_rate,
        "label_balance_gap": abs(pos_rate - 0.5) if pos_rate is not None else None,
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
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pt_sl_grid = parse_pt_sl_grid(args.pt_sl_grid)
    num_days_grid = parse_float_grid(args.num_days_grid)

    event_bars = load_ohlcv(args.symbol, args.event_timeframe, start=args.start, end=args.end)
    path_bars = load_ohlcv(args.symbol, args.path_timeframe, start=args.start, end=args.end)
    if event_bars.empty or path_bars.empty:
        raise RuntimeError("Event or path bars are empty")

    event_bars = event_bars.loc[(event_bars["close"] > 0) & (event_bars["high"] >= event_bars["low"])]
    path_bars = path_bars.loc[(path_bars["close"] > 0) & (path_bars["high"] >= path_bars["low"])]
    event_bars = filter_sparse_days(event_bars, args.min_daily_bars)
    path_bars = filter_sparse_days(path_bars, args.min_daily_bars)

    event_close = event_bars["close"].dropna()
    path_close = path_bars["close"].dropna()
    daily_vol = get_daily_vol(event_close, span=args.vol_span).dropna()
    if event_close.empty or path_close.empty or daily_vol.empty:
        raise RuntimeError("No usable event close, path close, or daily volatility")

    threshold = float(daily_vol.mean())
    t_events = symmetric_cusum_filter(event_close, threshold=threshold)
    metadata = {
        "symbol": args.symbol,
        "event_timeframe": args.event_timeframe,
        "path_timeframe": args.path_timeframe,
        "path_mode": "close",
        "event_rows": int(len(event_bars)),
        "path_rows": int(len(path_bars)),
        "daily_vol_mean": threshold,
        "cusum_events": int(len(t_events)),
    }

    rows: list[dict[str, object]] = []
    label_frames: list[pd.DataFrame] = []
    for num_days in num_days_grid:
        t1 = add_vertical_barrier(t_events, path_close, num_days=num_days)
        for pt, sl in pt_sl_grid:
            labels = resolve_mtf_events(
                entry_close=event_close,
                path_close=path_close,
                t_events=t_events,
                t1=t1,
                trgt=daily_vol,
                pt=pt,
                sl=sl,
                min_ret=args.min_ret,
            )
            labels["pt"] = pt
            labels["sl"] = sl
            labels["num_days"] = num_days
            rows.append(summarize(labels, metadata, pt, sl, num_days))
            label_frames.append(labels)

    sweep = pd.DataFrame(rows)
    labels_all = pd.concat(label_frames).sort_index() if label_frames else pd.DataFrame()
    start_key = pd.Timestamp(args.start).strftime("%Y%m")
    end_key = (pd.Timestamp(args.end) - pd.Timedelta(days=1)).strftime("%Y%m")
    stem = (
        f"mtf_barrier_sweep_{args.symbol.lower()}_{args.event_timeframe.lower()}_event_"
        f"{args.path_timeframe.lower()}_path_{args.run_name}_{start_key}_{end_key}"
    )
    csv_path = args.output_dir / f"{stem}.csv"
    labels_path = args.output_dir / f"{stem}_labels.csv"
    json_path = args.output_dir / f"{stem}.json"
    sweep.to_csv(csv_path, index=False)
    labels_all.to_csv(labels_path)
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "pt_sl_grid": [[pt, sl] for pt, sl in pt_sl_grid],
                "num_days_grid": num_days_grid,
                "rows": sweep.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    display_cols = [
        "event_timeframe",
        "path_timeframe",
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
    print(f"labels: {labels_path}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()
