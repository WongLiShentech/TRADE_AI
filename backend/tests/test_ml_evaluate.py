"""Tests for app.services.ml.evaluate — fold geometry, leakage integrity, gate wiring.

Two layers:
  * Pure structural tests on the fold split (no model): every fold's OOS is disjoint
    from its IS, and the IS train/validation split therefore never contains an OOS
    row — the core walk-forward leakage guarantee.
  * One end-to-end integration test against the live corpus that trains the three
    fold models and asserts the reported invariants (kept <= total, thresholds in
    range, ML trade count <= baseline).

Run: python -m pytest tests/test_ml_evaluate.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.backtester.runner import slice_folds
from app.services.ml.dataset import load_dataset
from app.services.ml.evaluate import (
    FoldEvaluation,
    WalkForwardResult,
    _build_windows,
    _degenerate_reason,
    evaluate_walk_forward,
    keep_fraction_report,
)


# ── degenerate guards ─────────────────────────────────────────────────────────
def test_degenerate_reasons():
    assert _degenerate_reason(n_is=100, n_val=20, n_oos=0) == "empty_oos"
    assert _degenerate_reason(n_is=100, n_val=0, n_oos=10) == "validation_tail_empty"
    assert _degenerate_reason(n_is=5, n_val=5, n_oos=10) == "training_slice_empty"
    assert _degenerate_reason(n_is=100, n_val=20, n_oos=10) is None


# ── chronological integrity: no OOS row ever lands in train/val ───────────────
def _synthetic_records(settings, n: int) -> list[dict]:
    windows = _build_windows(settings)
    start = windows[0].is_start
    end = windows[-1].oos_end
    times = pd.date_range(start, end, periods=n).floor("s")
    return [{"signal_time": ts.to_pydatetime(), "idx": i} for i, ts in enumerate(times)]


def test_oos_disjoint_from_is_and_trainval(settings):
    records = _synthetic_records(settings, 600)
    windows = _build_windows(settings)
    folds = slice_folds(records, windows)

    for fold in folds:
        is_idx = [r["idx"] for r in fold["is"]]
        oos_idx = [r["idx"] for r in fold["oos"]]
        # IS and OOS never overlap (embargo gap enforced by slice_folds).
        assert set(is_idx).isdisjoint(set(oos_idx))
        # The IS train/validation split is a partition of IS -> never reaches OOS.
        n_val = int(round(len(is_idx) * settings.ML_VALIDATION_FRACTION))
        if n_val >= 1 and len(is_idx) - n_val >= 1:
            train_idx = is_idx[:-n_val]
            val_idx = is_idx[-n_val:]
            assert set(train_idx).union(val_idx) == set(is_idx)
            assert set(train_idx).union(val_idx).isdisjoint(set(oos_idx))


def test_is_indices_are_chronological(settings):
    records = _synthetic_records(settings, 400)
    windows = _build_windows(settings)
    folds = slice_folds(records, windows)
    for fold in folds:
        is_idx = [r["idx"] for r in fold["is"]]
        assert is_idx == sorted(is_idx)  # order preserved -> IS tail is the latest val


def test_three_expanding_folds(settings):
    windows = _build_windows(settings)
    assert len(windows) == 3
    assert windows[0].is_start == windows[1].is_start == windows[2].is_start
    assert windows[0].is_end < windows[1].is_end < windows[2].is_end


# ── keep-fraction report: first-class filter-collapse visibility (QA H1) ──────
def _fe(fold: int, kept: int, oos_total: int, threshold: float, degen=None) -> FoldEvaluation:
    return FoldEvaluation(
        fold=fold, window={}, is_trade_count=0, oos_trade_count=oos_total,
        train_count=0, val_count=0, val_auc=float("nan"), threshold=threshold,
        kept=kept, keep_fraction=(kept / oos_total if oos_total else 0.0),
        val_expectancy=float("nan"), ml_filtered_metrics={}, baseline_metrics={},
        degenerate_reason=degen,
    )


def _result(folds: list[FoldEvaluation]) -> WalkForwardResult:
    return WalkForwardResult(
        n_trials=1, label_threshold_r=1.0, folds=folds,
        ml_passed=False, ml_gate_detail={}, baseline_passed=False,
        baseline_gate_detail={}, ml_combined_metrics={}, baseline_combined_metrics={},
    )


def test_keep_fraction_report_surfaces_per_fold_fractions():
    result = _result([
        _fe(1, kept=200, oos_total=400, threshold=0.5),   # 50%
        _fe(2, kept=46, oos_total=328, threshold=0.7),    # 14% — v1 fold-3 collapse shape
        _fe(3, kept=0, oos_total=0, threshold=0.0, degen="empty_oos"),
    ])
    rows = keep_fraction_report(result)

    assert [r["fold"] for r in rows] == [1, 2, 3]
    assert rows[0]["kept"] == 200 and rows[0]["oos_total"] == 400
    assert abs(rows[0]["keep_fraction"] - 0.5) < 1e-9
    assert abs(rows[1]["keep_fraction"] - 46 / 328) < 1e-9   # collapse is visible
    assert rows[2]["degenerate_reason"] == "empty_oos"
    # Every row carries the threshold that produced the kept set.
    assert rows[1]["threshold"] == 0.7


# ── end-to-end on the live corpus (trains the three fold models) ──────────────
def test_walk_forward_end_to_end_invariants(db, settings):
    dataset = load_dataset(db, settings.ML_LABEL_THRESHOLD_R)
    result = evaluate_walk_forward(db, settings, n_trials=1, dataset=dataset)

    assert len(result.folds) == 3
    for fe in result.folds:
        assert fe.kept <= fe.oos_trade_count            # filter never invents trades
        assert 0.0 <= fe.threshold <= 1.0
        assert fe.ml_filtered_metrics["trade_count"] == fe.kept
        if not fe.degenerate_reason:
            assert fe.val_count >= 1 and fe.train_count >= 1
    # ML-filtered keeps a subset of the baseline trades on the combined OOS.
    assert result.ml_combined_metrics["trade_count"] <= result.baseline_combined_metrics["trade_count"]
    assert isinstance(result.ml_passed, bool)
