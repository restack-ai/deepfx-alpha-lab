#!/usr/bin/env python3
"""Run BRK-only vs BRK+Kronos incremental explanatory-power study."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepfx_alpha_lab.kronos.brk_incremental import BrkIncrementalConfig, build_incremental_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-trades", required=True, type=Path, help="aligned_trades.csv from BRK disagreement study")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--window-minutes", type=int, default=120)
    parser.add_argument("--target", default="pnl")
    parser.add_argument("--min-train-group-size", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BrkIncrementalConfig(
        window_minutes=args.window_minutes,
        target=args.target,
        min_train_group_size=args.min_train_group_size,
    )
    outputs = build_incremental_study(args.aligned_trades, config, args.out_dir)
    print(f"wrote {args.out_dir / 'report.md'}")
    print(outputs["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
