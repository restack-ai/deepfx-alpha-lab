#!/usr/bin/env python3
"""AFML Ch04 sample uniqueness study for DeepFX trade/event intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deepfx_alpha_lab.sample_weights import compute_interval_uniqueness, summarize_weighted_pnl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path, help="Trade/event CSV with entry/exit timestamps")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--entry-col", default="entry_time")
    parser.add_argument("--exit-col", default="exit_time")
    parser.add_argument("--pnl-col", default="pnl")
    parser.add_argument("--window-minutes", type=int, default=None, help="Filter rows when window_minutes column exists")
    parser.add_argument("--family", default=None, help="Optional family filter, e.g. BRK")
    parser.add_argument("--pool", default="symbol", choices=["symbol", "portfolio"], help="Concurrency pool")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = load_events(
        args.events,
        entry_col=args.entry_col,
        exit_col=args.exit_col,
        pnl_col=args.pnl_col,
        window_minutes=args.window_minutes,
        family=args.family,
    )
    pool_cols = ["symbol"] if args.pool == "symbol" else []
    weighted = compute_interval_uniqueness(events, by=pool_cols)

    summary_groups = [
        ["family"],
        ["symbol", "side"],
        ["family", "alignment"],
        ["symbol", "side", "alignment"],
        ["risk_regime", "alignment"],
    ]
    summaries: dict[str, pd.DataFrame] = {}
    for group_cols in summary_groups:
        available = [col for col in group_cols if col in weighted.columns]
        if available:
            summaries["__".join(available)] = summarize_weighted_pnl(weighted, group_cols=available, pnl_col=args.pnl_col)

    report = render_report(events_path=args.events, pool=args.pool, weighted=weighted, summaries=summaries, pnl_col=args.pnl_col)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    weighted.to_csv(args.out_dir / "weighted_events.csv", index=False)
    for name, frame in summaries.items():
        frame.to_csv(args.out_dir / f"summary_{name}.csv", index=False)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"wrote {args.out_dir / 'report.md'}")
    print(report)
    return 0


def load_events(
    path: Path,
    *,
    entry_col: str,
    exit_col: str,
    pnl_col: str,
    window_minutes: int | None,
    family: str | None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted({entry_col, exit_col, pnl_col, "symbol", "side"} - set(frame.columns))
    if missing:
        raise ValueError(f"events CSV missing required columns: {missing}")
    out = frame.copy()
    if window_minutes is not None and "window_minutes" in out.columns:
        out = out[out["window_minutes"] == window_minutes].copy()
    if family is not None:
        if "family" in out.columns:
            out = out[out["family"].fillna("").astype(str).str.upper() == family.upper()].copy()
        elif "reason" in out.columns:
            out = out[out["reason"].fillna("").astype(str).str.upper().str.startswith(family.upper())].copy()
    out["t0"] = pd.to_datetime(out[entry_col], utc=True)
    out["t1"] = pd.to_datetime(out[exit_col], utc=True, errors="coerce")
    out["t1"] = out["t1"].fillna(out["t0"])
    out.loc[out["t1"] < out["t0"], "t1"] = out.loc[out["t1"] < out["t0"], "t0"]
    out[pnl_col] = pd.to_numeric(out[pnl_col], errors="raise")
    if "event_id" not in out.columns:
        out["event_id"] = [f"event-{i:04d}" for i in range(len(out))]
    if "family" not in out.columns:
        out["family"] = ""
    if "alignment" not in out.columns:
        out["alignment"] = "unknown"
    if "risk_regime" not in out.columns:
        out["risk_regime"] = "unknown"
    return out.sort_values(["t0", "symbol"]).reset_index(drop=True)


def render_report(*, events_path: Path, pool: str, weighted: pd.DataFrame, summaries: dict[str, pd.DataFrame], pnl_col: str) -> str:
    effective_n = float(weighted["uniqueness"].sum()) if not weighted.empty else 0.0
    raw_n = int(len(weighted))
    compression = effective_n / raw_n if raw_n else 0.0
    lines = [
        "# AFML Ch04 Sample Uniqueness Study",
        "",
        "Research-only. No live trading action is taken.",
        "",
        "## Inputs",
        "",
        f"- Events CSV: `{events_path}`",
        f"- Concurrency pool: `{pool}`",
        f"- Raw events: {raw_n}",
        f"- Effective sample size: {effective_n:.4f}",
        f"- Effective/raw ratio: {compression:.4f}",
        f"- Mean uniqueness: {weighted['uniqueness'].mean():.4f}" if raw_n else "- Mean uniqueness: nan",
        f"- Mean concurrency: {weighted['concurrency_mean'].mean():.4f}" if raw_n else "- Mean concurrency: nan",
        "",
        "## Weighted summaries",
        "",
    ]
    for name, frame in summaries.items():
        lines.extend([f"### {name}", "", *_render_frame(frame, max_rows=40), ""])
    lines.extend(
        [
            "## Notes",
            "",
            "- `effective_n` is the sum of average uniqueness weights.",
            "- Uniqueness is computed as the duration-weighted average of `1 / concurrency` over each event interval.",
            "- Lower effective/raw ratio means raw trade count is overstating independent evidence.",
            f"- Weighted PnL uses `{pnl_col} * uniqueness`; units are whatever the source CSV uses.",
        ]
    )
    return "\n".join(lines)


def _render_frame(frame: pd.DataFrame, max_rows: int = 30) -> list[str]:
    if frame.empty:
        return ["- No rows."]
    lines: list[str] = []
    for row in frame.head(max_rows).to_dict(orient="records"):
        label_keys = [key for key in row if key not in _METRIC_KEYS and pd.notna(row[key])]
        label = ", ".join(f"{key}={row[key]}" for key in label_keys) or "row"
        lines.append(f"- {label}:")
        for key, value in row.items():
            if key in label_keys:
                continue
            lines.append(f"  - {key}: {_format_value(value)}")
    if len(frame) > max_rows:
        lines.append(f"- ... truncated {len(frame) - max_rows} rows")
    return lines


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


_METRIC_KEYS = {
    "raw_n",
    "effective_n",
    "mean_uniqueness",
    "raw_pnl_sum",
    "raw_pnl_mean",
    "weighted_pnl_sum",
    "weighted_pnl_mean",
    "win_rate",
    "weighted_win_rate",
    "mean_duration_minutes",
}


if __name__ == "__main__":
    raise SystemExit(main())
