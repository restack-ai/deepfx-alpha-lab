#!/usr/bin/env python3
"""Generate research-only Kronos shadow markdown reports from CSV exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepfx_alpha_lab.kronos.shadow_report import ShadowReportConfig, build_shadow_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", required=True, type=Path, help="Kronos shadow signal CSV export")
    parser.add_argument("--trades", type=Path, default=None, help="Optional live trade CSV export")
    parser.add_argument("--period", default="daily", choices=["daily", "weekly"])
    parser.add_argument("--start", default=None, help="Inclusive UTC/ISO start timestamp or date")
    parser.add_argument("--end", default=None, help="Exclusive UTC/ISO end timestamp or date")
    parser.add_argument(
        "--policies",
        default="discard_conflict,strongest_abs_pred,net_pred",
        help="Comma-separated conflict policies",
    )
    parser.add_argument("--primary-policy", default="strongest_abs_pred")
    parser.add_argument("--threshold-bps", type=float, default=10.0, help="Optional net_pred threshold")
    parser.add_argument("--match-tolerance-minutes", type=int, default=15)
    parser.add_argument("--brk-prefix", default="BRK")
    parser.add_argument("--predicted-is-absolute", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="Markdown output path; defaults to stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policies = tuple(policy.strip() for policy in args.policies.split(",") if policy.strip())
    config = ShadowReportConfig(
        period=args.period,
        start=args.start,
        end=args.end,
        policies=policies,  # type: ignore[arg-type]
        primary_policy=args.primary_policy,  # type: ignore[arg-type]
        threshold_bps=args.threshold_bps,
        match_tolerance_minutes=args.match_tolerance_minutes,
        brk_prefix=args.brk_prefix,
        predicted_is_absolute=args.predicted_is_absolute,
    )
    report = build_shadow_report(args.signals, args.trades, config)
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
