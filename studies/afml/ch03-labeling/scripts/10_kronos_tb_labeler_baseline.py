#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfx_alpha_lab.kronos.baseline import (
    build_statistical_embeddings,
    evaluate_baseline_classifiers,
    load_npz_dataset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kronos triple-barrier labeler classifier baseline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "afml"
        / "ch03"
        / "kronos"
        / "kronos_tb_labeler_4symbols_multisymbol_m15_m1_ohlc_pt05_sl05_8h_202601_202605.npz",
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "afml" / "ch03" / "kronos",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y_type, event_times, feature_columns = load_npz_dataset(args.dataset)
    embeddings, embedding_columns = build_statistical_embeddings(x, feature_columns=feature_columns)
    results = evaluate_baseline_classifiers(
        embeddings,
        y_type,
        event_times,
        train_frac=args.train_frac,
        random_state=args.random_state,
    )
    payload = {
        "dataset": str(args.dataset),
        "target": "y_type",
        "embedding": "statistical_window_summary",
        "x_shape": list(x.shape),
        "embedding_shape": list(embeddings.shape),
        "feature_columns": feature_columns,
        "embedding_columns": embedding_columns,
        "train_frac": float(args.train_frac),
        "random_state": int(args.random_state),
        **results,
    }
    stem = args.dataset.stem
    out = args.output_dir / f"{stem}_stat_embedding_baseline.json"
    write_json(out, payload)
    print(out.read_text())
    print(f"summary: {out}")


if __name__ == "__main__":
    main()
