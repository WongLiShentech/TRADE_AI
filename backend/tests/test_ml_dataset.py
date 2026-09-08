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
    load_dataset,
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


# ── strategy scoping (multi-strategy corpus safety) ───────────────────────────
class _StubQuery:
    """Minimal stand-in for the SQLAlchemy chain load_dataset uses."""

    def __init__(self, rows):
        self._rows = rows
        self.filters = 0

    def filter(self, *_a, **_k):
        self.filters += 1
        return self

    def order_by(self, *_a, **_k):
        return self

    def all(self):
        return self._rows


class _StubSession:
    def __init__(self, rows):
        self.q = _StubQuery(rows)

    def query(self, *_a, **_k):
        return self.q


class _StubTrade:
    """Only `strategy_id` matters — the mixed-corpus guard runs BEFORE features
    are extracted, so a row that trips it never reaches feature parsing."""

    def __init__(self, strategy_id):
        self.strategy_id = strategy_id


def test_mixed_strategy_corpus_is_refused():
    """Two strategies in one corpus is a CORRUPT corpus, not a bigger one.

    The label is ``rr_actual >= threshold`` and rr_actual depends on the exit rule,
    so rows from a trailing-exit strategy and a pure-barrier one disagree about what
    a win IS. Training across both silently produces a worse model that looks fine —
    exactly the class of failure nothing downstream would ever surface.
    """
    db = _StubSession([_StubTrade(1), _StubTrade(2)])
    with pytest.raises(RuntimeError, match="mixes strategy_id"):
        load_dataset(db, 1.0)


def test_unattributed_rows_count_as_their_own_strategy():
    """A NULL strategy_id is not a wildcard — it is an unknown, and mixing an
    unknown with a known is exactly as unsafe as mixing two knowns."""
    db = _StubSession([_StubTrade(1), _StubTrade(None)])
    with pytest.raises(RuntimeError, match="mixes strategy_id"):
        load_dataset(db, 1.0)


def test_single_strategy_corpus_passes_the_guard():
    """The guard must not fire on the single-strategy case — that is every run
    this project has made to date, and breaking it would break M7/S1."""
    db = _StubSession([_StubTrade(1), _StubTrade(1)])
    # Passes the guard, then fails later on absent feature data. Reaching ANY
    # non-guard error proves the guard let it through.
    with pytest.raises(Exception) as exc:
        load_dataset(db, 1.0)
    assert "mixes strategy_id" not in str(exc.value)


def test_explicit_strategy_id_adds_a_filter_and_skips_the_guard():
    """Naming a strategy is the escape hatch: it scopes the query, so a corpus
    holding several strategies is fine as long as you say which one you mean."""
    db = _StubSession([_StubTrade(1), _StubTrade(2)])
    with pytest.raises(Exception) as exc:
        load_dataset(db, 1.0, strategy_id=1)
    assert "mixes strategy_id" not in str(exc.value)
    assert db.q.filters == 2  # stage filter + strategy filter


# ── deterministic row order (reproducibility) ────────────────────────────────
def test_corpus_order_is_total_and_reproducible(db, settings, corpus_strategy_id):
    """`ORDER BY opened_at` alone is NOT a total order on this corpus.

    Roughly 1,865 timestamps are shared by two or more trades — every pair signalling
    on the same H4 close — and Postgres may return tied rows in any order, differing
    between machines and after a dump/restore or VACUUM.

    That is not cosmetic. `train_production_model` splits train from validation
    POSITIONALLY, and on the strategy-1 corpus that boundary falls inside a group of
    three rows sharing one timestamp. Without a tie-break, `best_iteration`, the
    deployment threshold and the resulting `params_hash` could all vary run to run on
    byte-identical data — which was silently true until the dataset fingerprint made
    two machines disagree.
    """
    import pandas as pd

    ds = load_dataset(db, settings.ML_LABEL_THRESHOLD_R, strategy_id=corpus_strategy_id)
    st = pd.Series(ds.frame["signal_time"])

    # The hazard must actually exist, or this test proves nothing.
    assert st.duplicated().any(), (
        "no tied timestamps in this corpus — the ordering hazard this guards is "
        "absent, so the guard is untested rather than satisfied"
    )
    # Chronological order must still hold.
    assert st.is_monotonic_increasing, "corpus is not in chronological order"

    # And the load must be repeatable within a process.
    again = load_dataset(db, settings.ML_LABEL_THRESHOLD_R, strategy_id=corpus_strategy_id)
    assert ds.fingerprint() == again.fingerprint()


def test_load_dataset_breaks_timestamp_ties_on_a_stable_key():
    """Guards the tie-break itself: chronological order plus a stable surrogate.
    `id` survives dump/restore, so the order is identical across machines."""
    import inspect

    from app.services.ml import dataset as dataset_mod

    src = inspect.getsource(dataset_mod.load_dataset)
    assert "Trade.id.asc()" in src, (
        "load_dataset must break opened_at ties on a stable key, or row order — and "
        "therefore the train/validation split — is at the database's discretion"
    )
