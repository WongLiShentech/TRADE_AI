"""The dataset fingerprint must be stable where data is unchanged and sensitive
where it is not. Both halves are load-bearing: a digest that drifts on a library
upgrade trains you to ignore mismatches, and a digest that misses a changed value
is worse than none because it certifies data it did not check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.ml.dataset import _FINGERPRINT_COLUMNS
from app.services.ml.fingerprint import dataset_fingerprint, feature_keys_hash

COLS = ["atr14", "rsi_h4", "session", "rr_actual", "signal_time"]


def _frame(n: int = 5, **overrides) -> pd.DataFrame:
    data = {
        "atr14": [0.001, 0.002, 0.003, 0.004, 0.005][:n],
        "rsi_h4": [30.0, 40.0, 50.0, 60.0, 70.0][:n],
        "session": ["london", "ny", "asian", "overlap", "london"][:n],
        "rr_actual": [1.0, -1.0, 0.5, 2.0, -0.3][:n],
        "signal_time": pd.date_range("2025-01-01", periods=n, freq="4h"),
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ── stability ────────────────────────────────────────────────────────────────
def test_identical_frames_hash_identically():
    assert dataset_fingerprint(_frame(), COLS) == dataset_fingerprint(_frame(), COLS)


def test_repeated_calls_on_one_frame_are_stable():
    """The numeric path copies and mutates its array (NaN zeroing, -0.0
    normalisation). If it mutated the caller's frame instead, the second call would
    hash different data."""
    f = _frame()
    first = dataset_fingerprint(f, COLS)
    assert dataset_fingerprint(f, COLS) == first
    assert f["atr14"].tolist() == _frame()["atr14"].tolist()  # frame untouched


def test_negative_zero_hashes_as_positive_zero():
    """-0.0 == 0.0 is True in every comparison a human would make, but the bit
    patterns differ. Without normalisation, a sign that no arithmetic can observe
    would report the corpus as changed."""
    a = _frame(atr14=[0.0, 0.002, 0.003, 0.004, 0.005])
    b = _frame(atr14=[-0.0, 0.002, 0.003, 0.004, 0.005])
    assert dataset_fingerprint(a, COLS) == dataset_fingerprint(b, COLS)


def test_nan_positions_are_stable_regardless_of_payload():
    """NaN is a first-class value here — six macro features are routinely NaN on
    live rows — and quiet-NaN payload bits are not portable between producers."""
    a = _frame(atr14=[np.nan, 0.002, 0.003, 0.004, 0.005])
    b = _frame(atr14=[float("nan"), 0.002, 0.003, 0.004, 0.005])
    assert dataset_fingerprint(a, COLS) == dataset_fingerprint(b, COLS)


def test_all_nan_column_is_stable():
    nans = [np.nan] * 5
    assert dataset_fingerprint(_frame(atr14=nans), COLS) == dataset_fingerprint(
        _frame(atr14=nans), COLS
    )


# ── sensitivity ──────────────────────────────────────────────────────────────
def test_one_changed_feature_value_changes_the_digest():
    """The case (trade_id, rr_actual) hashing misses entirely: a leakage fix changes
    stored feature VALUES while ids, names and outcomes all stay put."""
    a = _frame()
    b = _frame(atr14=[0.001, 0.002, 0.0030001, 0.004, 0.005])
    assert dataset_fingerprint(a, COLS) != dataset_fingerprint(b, COLS)


def test_nan_in_a_different_position_changes_the_digest():
    a = _frame(atr14=[np.nan, 0.002, 0.003, 0.004, 0.005])
    b = _frame(atr14=[0.001, np.nan, 0.003, 0.004, 0.005])
    assert dataset_fingerprint(a, COLS) != dataset_fingerprint(b, COLS)


def test_zero_and_nan_are_distinguishable():
    """NaN slots are zeroed before hashing, so a real 0.0 and a NaN would collide
    were the mask not hashed separately."""
    a = _frame(atr14=[0.0, 0.002, 0.003, 0.004, 0.005])
    b = _frame(atr14=[np.nan, 0.002, 0.003, 0.004, 0.005])
    assert dataset_fingerprint(a, COLS) != dataset_fingerprint(b, COLS)


def test_row_order_changes_the_digest():
    """Row order is significant: the walk-forward folds slice positionally, so the
    same rows in a different order train different models."""
    a = _frame()
    b = a.iloc[::-1].reset_index(drop=True)
    assert dataset_fingerprint(a, COLS) != dataset_fingerprint(b, COLS)


def test_changed_outcome_changes_the_digest():
    a = _frame()
    b = _frame(rr_actual=[1.0, -1.0, 0.5, 2.0, 0.3])
    assert dataset_fingerprint(a, COLS) != dataset_fingerprint(b, COLS)


def test_fewer_rows_changes_the_digest():
    assert dataset_fingerprint(_frame(5), COLS) != dataset_fingerprint(_frame(4), COLS)


def test_renaming_a_column_changes_the_digest():
    """Column names are hashed, not just values — a contract change with coincidentally
    identical numbers is still a different dataset."""
    a = _frame()
    b = a.rename(columns={"atr14": "atr_14"})
    assert dataset_fingerprint(a, COLS) != dataset_fingerprint(
        b, ["atr_14" if c == "atr14" else c for c in COLS]
    )


def test_column_order_changes_the_digest():
    reordered = [COLS[1], COLS[0]] + COLS[2:]
    assert dataset_fingerprint(_frame(), COLS) != dataset_fingerprint(_frame(), reordered)


def test_categorical_values_are_length_prefixed():
    """Without length prefixes ("ab","c") and ("a","bc") concatenate identically."""
    a = _frame(n=2, session=["ab", "c"])
    b = _frame(n=2, session=["a", "bc"])
    assert dataset_fingerprint(a, COLS) != dataset_fingerprint(b, COLS)


def test_datetime_shift_changes_the_digest():
    a = _frame()
    b = _frame(signal_time=pd.date_range("2025-01-02", periods=5, freq="4h"))
    assert dataset_fingerprint(a, COLS) != dataset_fingerprint(b, COLS)


# ── contract ─────────────────────────────────────────────────────────────────
def test_missing_column_raises_rather_than_hashing_less():
    """Skipping an absent column would hash a smaller dataset under the same scheme
    tag, colliding with a genuinely smaller one."""
    with pytest.raises(KeyError, match="absent from frame"):
        dataset_fingerprint(_frame(), COLS + ["not_a_column"])


def test_fingerprint_columns_match_what_load_dataset_builds():
    """_FINGERPRINT_COLUMNS must stay identical to the frame load_dataset produces.
    If they drift, the digest silently stops covering part of the data."""
    from app.services.feature_builder import FEATURE_KEYS_MODEL

    expected = tuple(FEATURE_KEYS_MODEL) + ("rr_actual", "signal_time", "holding_hours")
    assert _FINGERPRINT_COLUMNS == expected


def test_feature_keys_hash_is_order_sensitive_and_stable():
    assert feature_keys_hash(["a", "b"]) == feature_keys_hash(["a", "b"])
    assert feature_keys_hash(["a", "b"]) != feature_keys_hash(["b", "a"])
    assert feature_keys_hash(["a", "b"]) != feature_keys_hash(["a", "b", "c"])
    assert feature_keys_hash(["ab", "c"]) != feature_keys_hash(["a", "bc"])
