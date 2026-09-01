"""Unit tests for app.services.ml.dataset — feature extraction + label rule.

Pure unit tests (no DB): they exercise the extraction helpers and the label
derivation on synthetic ``signal_reasoning`` dicts / frames.

Run: python -m pytest tests/test_ml_dataset.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.services.feature_builder import FEATURE_KEYS_MODEL
from app.services.ml.dataset import (
    CATEGORICAL_FEATURES,
    Dataset,
    NUMERIC_FEATURES,
    _CAT_MISSING,
    _extract_feature_row,
    _to_number,
    to_records,
)


# ── feature contract: extracted keys are EXACTLY the model keys (v2: 19) ──────
def test_extracted_row_keys_exactly_model_keys():
    reasoning = {k: 0.0 for k in FEATURE_KEYS_MODEL}
    reasoning["instrument_category"] = "major_usd"
    reasoning["session"] = "london"
    row = _extract_feature_row(reasoning)
    assert set(row.keys()) == set(FEATURE_KEYS_MODEL)


def test_numeric_and_categorical_partition_the_contract():
    assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(FEATURE_KEYS_MODEL)
    assert set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES) == set()
    assert set(CATEGORICAL_FEATURES) == {"instrument_category", "session"}


def test_extraction_ignores_non_model_keys():
    reasoning = {k: 1.0 for k in FEATURE_KEYS_MODEL}
    reasoning["instrument_category"] = "jpy_cross"
    reasoning["session"] = "asian"
    reasoning["us_10y"] = 1.23             # GATED tier — must NOT appear (v2)
    reasoning["atr14"] = 0.002             # PAYLOAD — must NOT appear
    row = _extract_feature_row(reasoning)
    assert "us_10y" not in row
    assert "atr14" not in row


# ── scalar coercion ───────────────────────────────────────────────────────────
def test_to_number_bool_none_and_numeric():
    assert _to_number(True) == 1.0
    assert _to_number(False) == 0.0
    assert math.isnan(_to_number(None))
    assert _to_number(3.4) == 3.4
    assert _to_number("2.5") == 2.5
    assert math.isnan(_to_number("not-a-number"))


def test_missing_categorical_becomes_sentinel():
    reasoning = {k: 0.0 for k in NUMERIC_FEATURES}  # categoricals absent
    row = _extract_feature_row(reasoning)
    assert row["instrument_category"] == _CAT_MISSING
    assert row["session"] == _CAT_MISSING


def test_missing_numeric_becomes_nan():
    reasoning = {"instrument_category": "major_usd", "session": "ny"}  # no numerics
    row = _extract_feature_row(reasoning)
    assert math.isnan(row["rsi14"])
    assert math.isnan(row["vix"])


# ── label rule (boundary rr == threshold) ─────────────────────────────────────
def _mini_dataset(rr_values: list[float], threshold: float) -> Dataset:
    frame = pd.DataFrame({
        "rr_actual": rr_values,
        "signal_time": pd.date_range("2025-01-01", periods=len(rr_values), freq="h"),
        "holding_hours": [1.0] * len(rr_values),
    })
    return Dataset(frame=frame, label_threshold_r=threshold, feature_schema_version=1)


def test_label_rule_boundary_at_threshold_is_a_win():
    ds = _mini_dataset([0.99, 1.0, 1.01, -1.0, 0.0], threshold=1.0)
    # rr >= 1.0 -> win; the boundary rr == 1.0 IS a win (>=, not >).
    assert list(ds.y) == [0, 1, 1, 0, 0]


def test_label_rule_respects_configured_threshold():
    ds = _mini_dataset([0.4, 0.5, 0.6], threshold=0.5)
    assert list(ds.y) == [0, 1, 1]


def test_rr_and_signal_time_preserved_alongside_features():
    ds = _mini_dataset([1.5, -1.0], threshold=1.0)
    assert list(ds.rr) == [1.5, -1.0]
    assert len(ds.signal_time) == 2


# ── record conversion for the metrics layer ───────────────────────────────────
def test_to_records_shape_and_nan_handling():
    frame = pd.DataFrame({
        "rr_actual": [1.5, np.nan],
        "signal_time": pd.date_range("2025-01-01", periods=2, freq="h"),
        "holding_hours": [10.0, np.nan],
    })
    recs = to_records(frame)
    assert recs[0]["rr_actual"] == 1.5 and recs[0]["holding_hours"] == 10.0
    assert recs[1]["rr_actual"] is None and recs[1]["holding_hours"] is None
