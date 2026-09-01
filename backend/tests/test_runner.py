"""Unit tests for app.services.backtester.runner — the walk-forward slicing math.

Pure unit tests (no DB): they exercise the runner's testable helpers on synthetic
trade lists — fold-window construction, IS/OOS slicing with the embargo gap,
weekly-cap bypass, and outcome labeling. The DB-backed ``run_backtest`` itself is
exercised by the live end-to-end backtest, not here.

Run: python -m pytest tests/test_runner.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.services.backtester.runner import (
    fold_windows,
    iso_week_key,
    label_outcome,
    slice_folds,
    weekly_cap_blocks,
)

_START = datetime(2024, 1, 1, 0, 0, 0)
_END = _START + timedelta(hours=1000)
_BOUNDS = [0.3, 0.6, 0.9]                 # T1=+300h, T2=+600h, T3=+900h
_EMBARGO = timedelta(hours=50)


def _at(hours: float) -> dict:
    return {"signal_time": _START + timedelta(hours=hours)}


def _windows():
    return fold_windows(_START, _END, _BOUNDS, _EMBARGO)


# ── fold window construction ─────────────────────────────────────────────────
def test_fold_windows_expanding_is_and_embargoed_oos():
    w = _windows()
    assert len(w) == 3
    # Expanding IS: same start, growing end.
    assert w[0].is_start == w[1].is_start == w[2].is_start == _START
    assert w[0].is_end == _START + timedelta(hours=300)
    assert w[1].is_end == _START + timedelta(hours=600)
    assert w[2].is_end == _START + timedelta(hours=900)
    # OOS begins one embargo past each boundary.
    assert w[0].oos_start == _START + timedelta(hours=350)
    assert w[0].oos_end == _START + timedelta(hours=600)
    assert w[1].oos_start == _START + timedelta(hours=650)
    assert w[1].oos_end == _START + timedelta(hours=900)
    assert w[2].oos_start == _START + timedelta(hours=950)
    assert w[2].oos_end == _END           # last fold runs to the window end
    assert w[2].is_last and not w[0].is_last


# ── slicing: expanding IS, disjoint OOS, embargo gap ─────────────────────────
def test_slice_expanding_is():
    trades = [_at(100), _at(400), _at(700)]
    folds = slice_folds(trades, _windows())
    is_sizes = [len(f["is"]) for f in folds]
    # IS is monotonically non-decreasing (each fold's IS is a superset).
    assert is_sizes == [1, 2, 3]


def test_slice_oos_disjoint():
    trades = [_at(400), _at(700), _at(980)]  # one squarely in each OOS
    folds = slice_folds(trades, _windows())
    oos_sizes = [len(f["oos"]) for f in folds]
    assert oos_sizes == [1, 1, 1]
    # No trade appears in more than one OOS set.
    all_oos = [t["signal_time"] for f in folds for t in f["oos"]]
    assert len(all_oos) == len(set(all_oos))


def test_embargo_gap_trade_in_no_oos():
    # t=320 is inside fold-1's embargo gap (300, 350); t=620 inside fold-2's (600,650).
    trades = [_at(320), _at(620)]
    folds = slice_folds(trades, _windows())
    for f in folds:
        assert len(f["oos"]) == 0          # never OOS
    # But they DO belong to the IS side of a later fold (expanding IS).
    assert any(t["signal_time"] == _at(320)["signal_time"] for t in folds[1]["is"])
    assert any(t["signal_time"] == _at(620)["signal_time"] for t in folds[2]["is"])


def test_oos_boundary_exclusions():
    # oos_start is inclusive; the next boundary (== oos_end of a non-last fold) is exclusive.
    on_start = _at(350)     # exactly OOS1 start -> included in OOS1
    on_end = _at(600)       # exactly OOS1 end / T2 boundary -> excluded from OOS1, no OOS
    folds = slice_folds([on_start, on_end], _windows())
    assert len(folds[0]["oos"]) == 1
    assert folds[0]["oos"][0]["signal_time"] == on_start["signal_time"]
    # on_end lands in fold-2's embargo gap (600, 650) -> in no OOS at all.
    assert all(on_end not in f["oos"] for f in folds)


def test_last_fold_includes_window_end():
    trades = [_at(1000)]    # exactly window_end -> last fold OOS is inclusive
    folds = slice_folds(trades, _windows())
    assert len(folds[2]["oos"]) == 1


# ── weekly-cap bypass ────────────────────────────────────────────────────────
def test_weekly_cap_bypassed_when_flag_false():
    counts = {"2025-W07": 999}
    # respect_cap=False -> never blocks, regardless of count vs max.
    assert weekly_cap_blocks(counts, "2025-W07", max_per_week=8, respect_cap=False) is False


def test_weekly_cap_enforced_when_flag_true():
    counts = {"2025-W07": 8}
    assert weekly_cap_blocks(counts, "2025-W07", max_per_week=8, respect_cap=True) is True
    assert weekly_cap_blocks(counts, "2025-W07", max_per_week=9, respect_cap=True) is False
    assert weekly_cap_blocks({}, "2025-W07", max_per_week=8, respect_cap=True) is False


def test_iso_week_key_stable():
    assert iso_week_key(datetime(2025, 2, 12)) == iso_week_key(datetime(2025, 2, 13))


# ── outcome labeling thresholds ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "rr,expected",
    [
        (1.5, "win"),
        (0.06, "win"),
        (0.05, "breakeven"),   # boundary: not strictly > 0.05
        (0.0, "breakeven"),
        (-0.05, "breakeven"),  # boundary: not strictly < -0.05
        (-0.06, "loss"),
        (-1.0, "loss"),
    ],
)
def test_label_outcome(rr, expected):
    assert label_outcome(rr) == expected
