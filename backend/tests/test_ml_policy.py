"""Unit tests for app.services.ml.policy — filter-threshold selection.

Structural guarantee under test: the threshold is chosen from VALIDATION arrays
only (``probs`` + ``rr`` passed in), maximising expectancy subject to the
minimum-keep floor. The function's signature makes it impossible to consult OOS —
it never receives OOS data.

Run: python -m pytest tests/test_ml_policy.py -v
"""
from __future__ import annotations

import math

import numpy as np

from app.services.ml.policy import select_threshold


def test_threshold_isolates_the_winners_above_floor():
    # Two losers (low prob) + two winners (high prob); floor 0.2 lets us keep just
    # the winners -> expectancy 2.0 at threshold 0.8.
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    rr = np.array([-1.0, -1.0, 2.0, 2.0])
    choice = select_threshold(probs, rr, min_keep_fraction=0.2)
    assert choice.threshold == 0.8
    assert choice.kept == 2
    assert choice.val_expectancy == 2.0


def test_min_keep_floor_prevents_gaming_to_a_tiny_slice():
    # The single best trade (rr=5) sits at the top prob, but the floor of 0.5 forces
    # keeping >= half the trades, so the search cannot collapse to keep-1.
    probs = np.array([0.1, 0.2, 0.3, 0.9])
    rr = np.array([-1.0, -1.0, 1.0, 5.0])
    choice = select_threshold(probs, rr, min_keep_fraction=0.5)
    assert choice.kept >= 2
    assert choice.keep_fraction >= 0.5


def test_ties_break_toward_lower_threshold_more_trades():
    # Every kept subset has the same expectancy (all rr equal) -> keep-all wins
    # (lowest threshold, most trades).
    probs = np.array([0.2, 0.4, 0.6, 0.8])
    rr = np.array([1.0, 1.0, 1.0, 1.0])
    choice = select_threshold(probs, rr, min_keep_fraction=0.2)
    assert choice.kept == 4
    assert choice.threshold == 0.2


def test_empty_validation_returns_keep_all_zero():
    choice = select_threshold(np.array([]), np.array([]), min_keep_fraction=0.2)
    assert choice.threshold == 0.0
    assert choice.kept == 0
    assert math.isnan(choice.val_expectancy)


def test_keep_all_is_always_a_valid_fallback():
    # All losers: no positive-expectancy subset exists, but a choice is still made
    # (keep-all at the minimum probability) rather than crashing.
    probs = np.array([0.3, 0.5, 0.7])
    rr = np.array([-1.0, -1.0, -1.0])
    choice = select_threshold(probs, rr, min_keep_fraction=0.2)
    assert choice.kept >= 1
    assert choice.threshold == 0.3  # min prob -> keep all
