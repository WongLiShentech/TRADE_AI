"""Filter-threshold selection for S1 — the ML-filtered strategy's decision rule.

The ML-filtered strategy KEEPS an OOS trade iff its predicted ``P(win) >= threshold``.
The threshold is chosen PER FOLD on that fold's IN-SAMPLE VALIDATION TAIL ONLY —
never on OOS data. This is the central leakage guarantee of the policy: the OOS
folds are scored with a threshold that was fixed before any OOS trade was seen.

Objective: maximise the kept set's expectancy (mean realised R) subject to keeping
at least ``min_keep_fraction`` of the validation trades. The floor is an
anti-gaming constraint — without it the search could "win" by keeping a handful of
lucky trades (e.g. 3 of 200), which would neither be a real strategy nor survive
OOS. Ties in expectancy are broken toward the LOWER threshold (keep more trades),
which is the more conservative, less overfit choice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdChoice:
    """Outcome of a threshold search on a validation set."""

    threshold: float
    kept: int
    total: int
    keep_fraction: float
    val_expectancy: float  # mean R of the kept validation trades (NaN if none kept)


def select_threshold(
    probs: np.ndarray,
    rr: np.ndarray,
    min_keep_fraction: float,
) -> ThresholdChoice:
    """Choose the P(win) threshold that maximises validation expectancy.

    Args:
        probs: predicted P(win) for each VALIDATION trade (in-sample tail only).
        rr: realised R-multiple for the same validation trades (aligned to ``probs``).
        min_keep_fraction: the kept set must be at least this fraction of the
            validation trades (``settings.ML_MIN_KEEP_FRACTION``) — anti-gaming floor.

    Returns:
        A :class:`ThresholdChoice`. If the validation set is empty, returns a
        keep-all threshold of 0.0.

    Notes:
        Candidate thresholds are the distinct predicted probabilities (each such
        value defines a distinct kept set via ``probs >= threshold``). The
        keep-all case (threshold = min prob) always satisfies the floor, so a valid
        choice always exists. Never consults ``rr`` of anything outside ``probs`` —
        i.e. never OOS.
    """
    n = len(probs)
    if n == 0:
        return ThresholdChoice(threshold=0.0, kept=0, total=0, keep_fraction=0.0, val_expectancy=float("nan"))

    min_keep = max(1, int(np.ceil(min_keep_fraction * n)))
    candidates = np.unique(probs)  # ascending; each is a valid ">=" cut point

    best: ThresholdChoice | None = None
    for thr in candidates:
        mask = probs >= thr
        kept = int(mask.sum())
        if kept < min_keep:
            continue
        exp = float(rr[mask].mean())
        choice = ThresholdChoice(
            threshold=float(thr),
            kept=kept,
            total=n,
            keep_fraction=kept / n,
            val_expectancy=exp,
        )
        # Maximise expectancy; tie-break toward the LOWER threshold (more trades).
        if best is None or (choice.val_expectancy, -choice.threshold) > (best.val_expectancy, -best.threshold):
            best = choice

    # The keep-all cut always satisfies the floor, so ``best`` is never None here.
    assert best is not None
    return best
