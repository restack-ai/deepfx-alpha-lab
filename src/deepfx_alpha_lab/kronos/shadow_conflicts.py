"""Research-only helpers for resolving Kronos shadow signal conflicts.

These helpers operate on offline forecast / shadow-signal rows. They do not read
live services, write decisions, or place orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

ConflictPolicy = Literal["discard_conflict", "strongest_abs_pred", "net_pred"]


@dataclass(frozen=True)
class ShadowConflictSummary:
    """Diagnostics from grouping shadow signals by symbol/time."""

    input_rows: int
    groups: int
    output_rows: int
    same_direction_duplicate_groups: int
    conflict_groups: int
    discarded_groups: int
    conflict_discarded_groups: int = 0
    zero_net_discarded_groups: int = 0
    threshold_discarded_groups: int = 0

    @property
    def discarded_rows(self) -> int:
        """Rows removed by either duplicate collapse or explicit group discard.

        Kept for backward-compatible reporting. For precise interpretation use
        the group-level discard reason fields above.
        """

        return self.input_rows - self.output_rows


_DIAGNOSTIC_COLUMNS = (
    "conflict_group",
    "group_size",
    "long_count",
    "short_count",
    "net_predicted_bps",
    "gross_predicted_bps",
    "max_abs_predicted_bps",
    "conflict_policy",
)


def apply_shadow_conflict_policy(
    signals: pd.DataFrame,
    *,
    policy: ConflictPolicy,
    group_columns: tuple[str, ...] = ("symbol", "time"),
    direction_column: str = "direction",
    predicted_column: str = "predicted_bps",
    threshold_bps: float | None = None,
    validate_direction_sign: bool = True,
) -> tuple[pd.DataFrame, ShadowConflictSummary]:
    """Collapse same symbol/time Kronos shadow rows into one research signal.

    AFML-style meta-labeling assumes a single primary side per event. If raw
    shadow output has both LONG and SHORT rows for the same event key, this
    function makes that ambiguity explicit and applies a deterministic research
    policy before downstream labeling/evaluation.

    Parameters
    ----------
    signals:
        Input rows containing at least `group_columns`, `direction_column`, and
        `predicted_column`.
    policy:
        - ``discard_conflict``: remove groups that contain both directions;
          collapse same-direction duplicates to the strongest absolute score.
        - ``strongest_abs_pred``: keep the row with maximum abs(predicted_bps)
          for every group, including conflicts.
        - ``net_pred``: sum signed predicted_bps within each group and emit one
          row with direction from the net sign. Groups whose absolute net is
          below `threshold_bps` are discarded when threshold is provided.
          Zero-net groups are always discarded.
    threshold_bps:
        Optional non-negative post-net threshold for ``net_pred``.
    validate_direction_sign:
        When true, enforce the default DeepFX shadow contract where LONG rows
        have positive `predicted_bps` and SHORT rows have negative values. If an
        upstream export stores absolute magnitudes, normalize signs before
        calling this helper or set this false deliberately.

    Returns
    -------
    (resolved, summary)
        ``resolved`` includes diagnostic columns:
        ``conflict_group``, ``group_size``, ``long_count``, ``short_count``,
        ``net_predicted_bps``, ``gross_predicted_bps``,
        ``max_abs_predicted_bps``, and ``conflict_policy``.
    """

    _validate_inputs(signals, group_columns, direction_column, predicted_column)
    if policy not in {"discard_conflict", "strongest_abs_pred", "net_pred"}:
        raise ValueError(f"unsupported conflict policy: {policy}")
    if threshold_bps is not None and threshold_bps < 0:
        raise ValueError("threshold_bps must be non-negative")

    normalized = _normalize_signals(signals, direction_column, predicted_column, validate_direction_sign)
    if normalized.empty:
        return _empty_resolved(normalized), ShadowConflictSummary(0, 0, 0, 0, 0, 0)

    resolved_rows: list[pd.Series] = []
    groups = list(normalized.groupby(list(group_columns), sort=True, dropna=False))
    conflict_groups = 0
    duplicate_groups = 0
    discarded_groups = 0
    conflict_discarded_groups = 0
    zero_net_discarded_groups = 0
    threshold_discarded_groups = 0

    for _, group in groups:
        diagnostics = _group_diagnostics(group, direction_column, predicted_column)
        group_size = int(diagnostics["group_size"])
        conflict_group = bool(diagnostics["conflict_group"])
        if group_size > 1 and not conflict_group:
            duplicate_groups += 1
        if conflict_group:
            conflict_groups += 1

        if policy == "discard_conflict" and conflict_group:
            discarded_groups += 1
            conflict_discarded_groups += 1
            continue

        if policy == "net_pred":
            net = float(group[predicted_column].astype(float).sum())
            if net == 0:
                discarded_groups += 1
                zero_net_discarded_groups += 1
                continue
            if threshold_bps is not None and abs(net) < threshold_bps:
                discarded_groups += 1
                threshold_discarded_groups += 1
                continue
            row = group.iloc[0].copy()
            row[predicted_column] = net
            row[direction_column] = "LONG" if net > 0 else "SHORT"
        else:
            row = _strongest_row(group, predicted_column)

        for column, value in diagnostics.items():
            row[column] = value
        row["conflict_policy"] = policy
        resolved_rows.append(row)

    resolved = pd.DataFrame(resolved_rows).reset_index(drop=True) if resolved_rows else _empty_resolved(normalized)
    summary = ShadowConflictSummary(
        input_rows=len(signals),
        groups=len(groups),
        output_rows=len(resolved),
        same_direction_duplicate_groups=duplicate_groups,
        conflict_groups=conflict_groups,
        discarded_groups=discarded_groups,
        conflict_discarded_groups=conflict_discarded_groups,
        zero_net_discarded_groups=zero_net_discarded_groups,
        threshold_discarded_groups=threshold_discarded_groups,
    )
    return resolved, summary


def _validate_inputs(
    signals: pd.DataFrame,
    group_columns: tuple[str, ...],
    direction_column: str,
    predicted_column: str,
) -> None:
    required = set(group_columns) | {direction_column, predicted_column}
    missing = sorted(required - set(signals.columns))
    if missing:
        raise ValueError(f"signals missing required columns: {missing}")


def _normalize_signals(
    signals: pd.DataFrame,
    direction_column: str,
    predicted_column: str,
    validate_direction_sign: bool,
) -> pd.DataFrame:
    normalized = signals.copy()
    normalized[direction_column] = normalized[direction_column].map(_normalize_direction)
    if normalized[direction_column].isna().any():
        bad = signals.loc[normalized[direction_column].isna(), direction_column].unique().tolist()
        raise ValueError(f"unsupported direction values: {bad}")
    normalized[predicted_column] = pd.to_numeric(normalized[predicted_column], errors="raise")
    values = normalized[predicted_column].astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{predicted_column} must contain finite numeric values")
    if validate_direction_sign:
        bad_sign = ((normalized[direction_column] == "LONG") & (values < 0)) | (
            (normalized[direction_column] == "SHORT") & (values > 0)
        )
        if bad_sign.any():
            raise ValueError(f"{predicted_column} sign is inconsistent with {direction_column}")
    return normalized


def _empty_resolved(frame: pd.DataFrame) -> pd.DataFrame:
    resolved = frame.iloc[0:0].copy().reset_index(drop=True)
    for column in _DIAGNOSTIC_COLUMNS:
        if column not in resolved.columns:
            resolved[column] = pd.Series(dtype="object")
    return resolved


def _normalize_direction(value: object) -> str | None:
    raw = str(value).strip().upper()
    if raw in {"LONG", "BUY", "1"}:
        return "LONG"
    if raw in {"SHORT", "SELL", "-1"}:
        return "SHORT"
    return None


def _group_diagnostics(group: pd.DataFrame, direction_column: str, predicted_column: str) -> dict[str, Any]:
    directions = group[direction_column]
    long_count = int((directions == "LONG").sum())
    short_count = int((directions == "SHORT").sum())
    predicted = group[predicted_column].astype(float)
    return {
        "conflict_group": long_count > 0 and short_count > 0,
        "group_size": int(len(group)),
        "long_count": long_count,
        "short_count": short_count,
        "net_predicted_bps": float(predicted.sum()),
        "gross_predicted_bps": float(predicted.abs().sum()),
        "max_abs_predicted_bps": float(predicted.abs().max()),
    }


def _strongest_row(group: pd.DataFrame, predicted_column: str) -> pd.Series:
    sort_frame = group.assign(_abs_pred=group[predicted_column].astype(float).abs())
    # Stable tie-break: strongest absolute prediction first, then original order.
    return sort_frame.sort_values("_abs_pred", ascending=False, kind="mergesort").iloc[0].drop(labels=["_abs_pred"])
