"""Dataset extraction for S1 — the rule-engine backtest corpus → model matrix.

Loads every ``stage='backtest'`` trade (chronologically by ``opened_at``), pulls
the 20 LOCKED model features out of each row's ``signal_reasoning`` JSON, and
derives the binary label from ``rr_actual``. The feature key list is IMPORTED from
``feature_builder`` (never hardcoded) so the training contract can never silently
drift from the serving contract.

Leakage note: this module only READS already-persisted, point-in-time feature
dicts (built by ``feature_builder`` at signal time during M7). It performs no
cross-row computation, no normalization, and no fitting — so nothing here can leak
future information into a row. All fitting happens per-fold in ``pipeline``/``model``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.trade import Trade
from app.services.feature_builder import FEATURE_KEYS_MODEL, FEATURE_SCHEMA_VERSION

# The two categorical model features (one-hot encoded); the rest are numeric and
# pass straight through to XGBoost (NaN-native). Derived from the locked contract,
# never a standalone hardcoded list.
CATEGORICAL_FEATURES: tuple[str, ...] = ("instrument_category", "session")
NUMERIC_FEATURES: tuple[str, ...] = tuple(
    k for k in FEATURE_KEYS_MODEL if k not in CATEGORICAL_FEATURES
)

_STAGE = "backtest"
# Sentinel for a missing categorical value at extraction time. feature_builder
# always emits both categoricals as non-null strings, so this should never appear
# in the real corpus; it exists only so a degenerate row can never crash the
# encoder's fit (handle_unknown='ignore' then blanks it at transform).
_CAT_MISSING = "__missing__"


@dataclass(frozen=True)
class Dataset:
    """The extracted training corpus.

    Attributes:
        frame: one row per trade, columns = the 20 model features + ``rr_actual`` +
            ``signal_time`` (== ``opened_at``) + ``holding_hours``. Row order is
            strictly ascending ``signal_time`` (chronological — never shuffled).
        label_threshold_r: the R threshold used to derive ``y`` (win if
            ``rr_actual >= threshold``).
        feature_schema_version: the ``feature_builder`` schema the rows were built
            with (guards against training on a stale contract).
    """

    frame: pd.DataFrame
    label_threshold_r: float
    feature_schema_version: int

    @property
    def X(self) -> pd.DataFrame:
        """Feature matrix — exactly the 20 model columns, in contract order."""
        return self.frame[list(FEATURE_KEYS_MODEL)]

    @property
    def y(self) -> np.ndarray:
        """Binary label: 1 if ``rr_actual >= label_threshold_r`` else 0."""
        return (self.frame["rr_actual"].to_numpy() >= self.label_threshold_r).astype(int)

    @property
    def rr(self) -> np.ndarray:
        """Realised R-multiple per trade (policy evaluation ground truth)."""
        return self.frame["rr_actual"].to_numpy(dtype=float)

    @property
    def signal_time(self) -> np.ndarray:
        """Signal timestamp per trade (== ``opened_at``); used for fold slicing."""
        return self.frame["signal_time"].to_numpy()

    def __len__(self) -> int:
        return len(self.frame)


def load_dataset(db: Session, label_threshold_r: float) -> Dataset:
    """Load and extract the ``stage='backtest'`` corpus into a :class:`Dataset`.

    Args:
        db: SQLAlchemy session (read-only).
        label_threshold_r: R threshold for the win label (``settings.ML_LABEL_THRESHOLD_R``).

    Returns:
        A :class:`Dataset` ordered chronologically by ``opened_at``.

    Raises:
        RuntimeError: if the corpus is empty, or rows carry mixed feature-schema
            versions (which would mean an inconsistent feature contract).
    """
    trades = (
        db.query(Trade)
        .filter(Trade.stage == _STAGE)
        .order_by(Trade.opened_at.asc())
        .all()
    )
    if not trades:
        raise RuntimeError(
            f"no trades with stage='{_STAGE}' — run the M7 backtest before training S1"
        )

    rows: list[dict] = []
    schema_versions: set[int] = set()
    for tr in trades:
        reasoning = tr.signal_reasoning or {}
        sv = reasoning.get("feature_schema_version")
        if sv is not None:
            schema_versions.add(int(sv))
        row = _extract_feature_row(reasoning)
        row["rr_actual"] = float(tr.rr_actual) if tr.rr_actual is not None else np.nan
        row["signal_time"] = tr.opened_at
        row["holding_hours"] = _holding_hours(tr.opened_at, tr.closed_at)
        rows.append(row)

    if len(schema_versions) > 1:
        raise RuntimeError(
            f"corpus mixes feature_schema_version={sorted(schema_versions)} — "
            f"retrain on a single-schema corpus (expected {FEATURE_SCHEMA_VERSION})"
        )

    columns = list(FEATURE_KEYS_MODEL) + ["rr_actual", "signal_time", "holding_hours"]
    frame = pd.DataFrame(rows, columns=columns)
    # Numeric features → float64 (NaN preserved for XGBoost); categoricals → object.
    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype("object")

    schema = schema_versions.pop() if len(schema_versions) == 1 else FEATURE_SCHEMA_VERSION
    return Dataset(
        frame=frame,
        label_threshold_r=label_threshold_r,
        feature_schema_version=schema,
    )


def _extract_feature_row(reasoning: dict) -> dict:
    """Pull exactly the 20 model keys from one ``signal_reasoning`` dict.

    Missing keys become NaN (numeric) or the categorical sentinel — never a
    KeyError. Booleans (the news flags) are coerced to float 1.0/0.0 so XGBoost
    treats them as numeric; ``None`` stays NaN.
    """
    row: dict = {}
    for key in NUMERIC_FEATURES:
        row[key] = _to_number(reasoning.get(key))
    for key in CATEGORICAL_FEATURES:
        val = reasoning.get(key)
        row[key] = val if isinstance(val, str) and val else _CAT_MISSING
    return row


def _to_number(value) -> float:
    """Coerce a JSON scalar to float. bool → 1.0/0.0; None → NaN; str-number → float."""
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _holding_hours(opened_at: datetime | None, closed_at: datetime | None) -> float:
    if opened_at is None or closed_at is None:
        return np.nan
    return (closed_at - opened_at).total_seconds() / 3600.0


def to_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a (sub)frame into the record dicts ``metrics``/``evaluate_promotion_gate``
    consume: ``{"rr_actual", "holding_hours", "signal_time"}`` per row."""
    return [
        {
            "rr_actual": float(r.rr_actual) if not pd.isna(r.rr_actual) else None,
            "holding_hours": float(r.holding_hours) if not pd.isna(r.holding_hours) else None,
            "signal_time": r.signal_time,
        }
        for r in frame.itertuples(index=False)
    ]
