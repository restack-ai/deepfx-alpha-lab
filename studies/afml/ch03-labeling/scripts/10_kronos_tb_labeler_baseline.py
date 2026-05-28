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
from deepfx_alpha_lab.kronos.encoder import KronosEncoderConfig, build_frozen_kronos_embeddings


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
        "--embedding",
        choices=["statistical", "kronos"],
        default="statistical",
        help="Embedding backend: statistical summary baseline or frozen Kronos hidden states.",
    )
    parser.add_argument("--kronos-repo", type=Path, default=None, help="Path to cloned shiyu-coder/Kronos repo.")
    parser.add_argument("--kronos-model", default="NeoQuasar/Kronos-mini")
    parser.add_argument("--kronos-tokenizer", default="NeoQuasar/Kronos-Tokenizer-2k")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--pooling", choices=["last", "mean", "mean_last"], default="mean_last")
    parser.add_argument("--freq", default="15min", help="Fallback event bar frequency for older datasets without window_times.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "afml" / "ch03" / "kronos",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y_type, event_times, feature_columns, kronos_x, kronos_columns, window_times = load_npz_dataset(args.dataset)
    embedding_metadata = {}
    if args.embedding == "statistical":
        embeddings, embedding_columns = build_statistical_embeddings(x, feature_columns=feature_columns)
        embedding_name = "statistical_window_summary"
        suffix = "stat_embedding_baseline"
    else:
        if kronos_x is None or kronos_columns is None:
            raise SystemExit("error: dataset does not contain raw Kronos input windows; rebuild it with script 09")
        config = KronosEncoderConfig(
            model_id=args.kronos_model,
            tokenizer_id=args.kronos_tokenizer,
            kronos_repo=args.kronos_repo,
            device=args.device,
            batch_size=args.batch_size,
            max_context=args.max_context,
            pooling=args.pooling,
            freq=args.freq,
        )
        try:
            embeddings, embedding_columns, embedding_metadata = build_frozen_kronos_embeddings(
                kronos_x,
                event_times,
                feature_columns=kronos_columns,
                window_times=window_times,
                config=config,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        embedding_name = "frozen_kronos_hidden_state"
        suffix = f"kronos_{args.pooling}_embedding_baseline"
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
        "embedding": embedding_name,
        "embedding_metadata": embedding_metadata,
        "x_shape": list(x.shape),
        "kronos_x_shape": list(kronos_x.shape) if kronos_x is not None else None,
        "window_times_shape": list(window_times.shape) if window_times is not None else None,
        "embedding_shape": list(embeddings.shape),
        "feature_columns": feature_columns,
        "kronos_columns": kronos_columns,
        "embedding_columns": embedding_columns,
        "train_frac": float(args.train_frac),
        "random_state": int(args.random_state),
        **results,
    }
    stem = args.dataset.stem
    out = args.output_dir / f"{stem}_{suffix}.json"
    write_json(out, payload)
    print(out.read_text())
    print(f"summary: {out}")


if __name__ == "__main__":
    main()
