#!/usr/bin/env python3
"""Run BRK × Kronos disagreement study from CSV exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepfx_alpha_lab.kronos.brk_disagreement import BrkDisagreementConfig, build_disagreement_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", required=True, type=Path, help="Kronos shadow signal CSV")
    parser.add_argument("--trades", required=True, type=Path, help="Live trade CSV")
    parser.add_argument("--ohlcv", type=Path, default=None, help="Optional M5 OHLCV CSV")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for report/CSV outputs")
    parser.add_argument("--start", default=None, help="Inclusive UTC start")
    parser.add_argument("--end", default=None, help="Exclusive UTC end")
    parser.add_argument("--windows", default="15,30,60,120,180,240", help="Comma-separated prior-signal windows in minutes")
    parser.add_argument("--primary-policy", default="strongest_abs_pred", choices=["discard_conflict", "strongest_abs_pred", "net_pred"])
    parser.add_argument("--brk-prefix", default="BRK")
    parser.add_argument("--placebo-window-minutes", type=int, default=120)
    parser.add_argument("--placebo-iterations", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    windows = tuple(int(part.strip()) for part in args.windows.split(",") if part.strip())
    config = BrkDisagreementConfig(
        windows_minutes=windows,
        primary_policy=args.primary_policy,
        brk_prefix=args.brk_prefix,
        start=args.start,
        end=args.end,
        placebo_iterations=args.placebo_iterations,
        placebo_window_minutes=args.placebo_window_minutes,
        random_seed=args.random_seed,
    )
    outputs = build_disagreement_study(
        signals_path=args.signals,
        trades_path=args.trades,
        ohlcv_path=args.ohlcv,
        config=config,
        out_dir=args.out_dir,
    )
    print(f"wrote {args.out_dir / 'report.md'}")
    print(outputs["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
