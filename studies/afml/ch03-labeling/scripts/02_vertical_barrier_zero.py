#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "afml" / "ch03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AFML Exercise 3.3 using Exercise 3.1 labels.")
    parser.add_argument(
        "--input-labels",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "exercise_3_1_xauusd_m1_labels.csv",
    )
    parser.add_argument(
        "--input-summary",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "exercise_3_1_xauusd_m1_summary.json",
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "exercise_3_3_xauusd_m1_labels.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "exercise_3_3_xauusd_m1_summary.json",
    )
    return parser.parse_args()


def distribution(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def main() -> None:
    args = parse_args()
    if not args.input_labels.exists():
        raise FileNotFoundError(f"Missing Exercise 3.1 labels: {args.input_labels}")

    labels = pd.read_csv(args.input_labels, parse_dates=["t0", "t1"]).set_index("t0")
    if "type" not in labels or "bin" not in labels:
        raise RuntimeError("Input labels must contain type and bin columns")

    original = labels.copy()
    modified = labels.copy()
    modified.loc[modified["type"] == "t1", "bin"] = 0

    original_summary = {}
    if args.input_summary.exists():
        original_summary = json.loads(args.input_summary.read_text())

    changed = original["bin"] != modified["bin"]
    vertical = original["type"] == "t1"
    summary = {
        "source_labels": str(args.input_labels),
        "exercise": "3.3",
        "rule": "vertical barrier exits are labeled 0",
        "labeled_events": int(len(modified)),
        "vertical_barrier_events": int(vertical.sum()),
        "changed_labels": int(changed.sum()),
        "original_label_distribution": distribution(original["bin"]),
        "modified_label_distribution": distribution(modified["bin"]),
        "barrier_distribution": distribution(modified["type"]),
        "changed_from": distribution(original.loc[changed, "bin"]) if changed.any() else {},
        "exercise_3_1_summary": original_summary,
    }

    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    modified.to_csv(args.output_labels)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"labels: {args.output_labels}")
    print(f"summary: {args.output_summary}")


if __name__ == "__main__":
    main()
