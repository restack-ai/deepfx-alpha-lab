from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Pooling = Literal["last", "mean", "mean_last"]


@dataclass(frozen=True)
class KronosEncoderConfig:
    """Runtime configuration for extracting frozen Kronos sequence embeddings."""

    model_id: str = "NeoQuasar/Kronos-mini"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-2k"
    kronos_repo: Path | None = None
    device: str | None = None
    batch_size: int = 32
    max_context: int = 512
    clip: float = 5.0
    pooling: Pooling = "mean_last"
    freq: str = "15min"


def reconstruct_window_timestamps(
    event_times: pd.DatetimeIndex | pd.Series | np.ndarray,
    *,
    lookback: int,
    freq: str = "15min",
) -> np.ndarray:
    """Reconstruct fixed-window timestamps ending at each event time."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    events = pd.DatetimeIndex(pd.to_datetime(event_times))
    step = pd.Timedelta(freq)
    offsets = pd.to_timedelta(np.arange(lookback - 1, -1, -1) * step.value, unit="ns")
    windows = []
    for event_time in events:
        windows.append(pd.DatetimeIndex(event_time - offsets).to_pydatetime().tolist())
    return np.asarray(windows, dtype=object)


def build_kronos_model_input(
    x: np.ndarray,
    *,
    feature_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Map alpha-lab fixed-window features to Kronos OHLCVA input.

    Kronos expects six columns: open, high, low, close, volume, amount.
    The alpha-lab dataset stores tick volume as ``tick_volume`` and does not
    have amount, so amount is approximated as volume * close.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape [n_events, lookback, n_features]")
    if x.shape[2] != len(feature_columns):
        raise ValueError("feature_columns length must match x.shape[2]")
    index = {name: idx for idx, name in enumerate(feature_columns)}
    required = ["open", "high", "low", "close"]
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"missing required OHLC columns: {missing}")
    volume_name = "volume" if "volume" in index else "tick_volume" if "tick_volume" in index else None
    if volume_name is None:
        raise ValueError("missing volume or tick_volume column")

    open_ = x[:, :, index["open"]]
    high = x[:, :, index["high"]]
    low = x[:, :, index["low"]]
    close = x[:, :, index["close"]]
    volume = x[:, :, index[volume_name]]
    amount = volume * close
    model_input = np.stack([open_, high, low, close, volume, amount], axis=2).astype("float32")
    return model_input, ["open", "high", "low", "close", "volume", "amount"]


def pool_sequence_embeddings(hidden: np.ndarray, *, pooling: Pooling) -> np.ndarray:
    """Pool [batch, seq, dim] Kronos hidden states to [batch, embedding_dim]."""
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, seq, dim]")
    if pooling == "last":
        return hidden[:, -1, :].astype("float32")
    if pooling == "mean":
        return hidden.mean(axis=1).astype("float32")
    if pooling == "mean_last":
        return np.concatenate([hidden.mean(axis=1), hidden[:, -1, :]], axis=1).astype("float32")
    raise ValueError(f"unsupported pooling: {pooling}")


def _prepare_kronos_import(kronos_repo: Path | None) -> None:
    if kronos_repo is not None:
        repo = Path(kronos_repo).expanduser().resolve()
        if not (repo / "model" / "kronos.py").exists():
            raise FileNotFoundError(f"Kronos repo not found or invalid: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))


def _load_kronos_classes(kronos_repo: Path | None):
    _prepare_kronos_import(kronos_repo)
    try:
        module = importlib.import_module("model")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Kronos source is not importable. Clone https://github.com/shiyu-coder/Kronos "
            "and pass --kronos-repo /path/to/Kronos, or install it on PYTHONPATH."
        ) from exc
    try:
        return module.Kronos, module.KronosTokenizer
    except AttributeError as exc:
        raise RuntimeError("Imported model module does not expose Kronos and KronosTokenizer") from exc


def _calc_time_stamps(window_timestamps: np.ndarray) -> np.ndarray:
    flat = pd.Series(pd.to_datetime(window_timestamps.reshape(-1)))
    time_df = pd.DataFrame(
        {
            "minute": flat.dt.minute,
            "hour": flat.dt.hour,
            "weekday": flat.dt.weekday,
            "day": flat.dt.day,
            "month": flat.dt.month,
        }
    )
    return time_df.to_numpy(dtype="float32").reshape(*window_timestamps.shape, 5)


def _normalize_for_kronos(x: np.ndarray, *, clip: float) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    norm = (x - mean) / (std + 1e-5)
    return np.clip(norm, -clip, clip).astype("float32")


def build_frozen_kronos_embeddings(
    x: np.ndarray,
    event_times: pd.DatetimeIndex,
    *,
    feature_columns: list[str],
    config: KronosEncoderConfig | None = None,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    """Extract frozen Kronos hidden-state embeddings for fixed-window events.

    This uses the upstream Kronos tokenizer to quantize OHLCVA windows, then runs
    the frozen autoregressive Kronos model through ``decode_s1`` and pools the
    returned context representation. No gradients or fine-tuning are used.
    """
    cfg = config or KronosEncoderConfig()
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.max_context <= 0:
        raise ValueError("max_context must be positive")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required for frozen Kronos embeddings. Install with `uv sync --extra kronos` "
            "or `uv add --optional kronos torch einops huggingface-hub safetensors tqdm`."
        ) from exc

    Kronos, KronosTokenizer = _load_kronos_classes(cfg.kronos_repo)
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_id)
    model = Kronos.from_pretrained(cfg.model_id)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = tokenizer.to(device).eval()
    model = model.to(device).eval()

    model_input, input_columns = build_kronos_model_input(x, feature_columns=feature_columns)
    window_timestamps = reconstruct_window_timestamps(event_times, lookback=x.shape[1], freq=cfg.freq)
    stamps = _calc_time_stamps(window_timestamps)
    norm_x = _normalize_for_kronos(model_input, clip=cfg.clip)

    hidden_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(norm_x), cfg.batch_size):
            end = min(start + cfg.batch_size, len(norm_x))
            xb = torch.from_numpy(norm_x[start:end]).to(device)
            sb = torch.from_numpy(stamps[start:end]).to(device)
            if xb.shape[1] > cfg.max_context:
                xb = xb[:, -cfg.max_context :, :]
                sb = sb[:, -cfg.max_context :, :]
            s1_ids, s2_ids = tokenizer.encode(xb, half=True)
            _, context = model.decode_s1(s1_ids, s2_ids, sb)
            pooled = pool_sequence_embeddings(context.detach().cpu().numpy(), pooling=cfg.pooling)
            hidden_batches.append(pooled)
    embeddings = np.concatenate(hidden_batches, axis=0).astype("float32")
    columns = [f"kronos_{cfg.pooling}_{idx}" for idx in range(embeddings.shape[1])]
    metadata: dict[str, object] = {
        "model_id": cfg.model_id,
        "tokenizer_id": cfg.tokenizer_id,
        "kronos_repo": str(cfg.kronos_repo) if cfg.kronos_repo else None,
        "device": device,
        "batch_size": cfg.batch_size,
        "max_context": cfg.max_context,
        "clip": cfg.clip,
        "pooling": cfg.pooling,
        "freq": cfg.freq,
        "input_columns": input_columns,
    }
    return embeddings, columns, metadata
