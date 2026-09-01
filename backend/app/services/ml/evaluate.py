"""Walk-forward evaluation for S1 — the ML-filtered strategy vs the rule baseline.

Mirrors M7's three-fold expanding walk-forward EXACTLY: the fold windows are built
with the same ``fold_windows``/``slice_folds`` helpers over the same execution
window (``EXECUTION_WINDOW_START/END``), fractional bounds (``BACKTEST_FOLD_BOUNDS``)
and embargo (``BACKTEST_EMBARGO_BARS``) the rule backtest used, so the ML-filtered
OOS folds line up one-to-one with the rule baseline's OOS folds.

Per fold (the leakage discipline, LEAK-6):

1. Slice trades by ``signal_time`` into this fold's IS and OOS (embargo gap excluded
   from both, per ``slice_folds``).
2. Split IS chronologically: the last ``ML_VALIDATION_FRACTION`` becomes the
   validation tail; the earlier part is the training slice.
3. Fit the one-hot encoder AND the XGBoost model on the TRAINING SLICE ONLY (never
   the validation tail, never OOS). Early stopping uses the validation tail.
4. Choose the P(win) filter threshold on the validation tail ONLY (``policy``).
5. Score OOS: keep OOS trades with ``P(win) >= threshold``. Compute metrics + the
   promotion gate on the kept set, and — for a side-by-side — on the full
   (unfiltered) OOS set, which is exactly the rule baseline for that fold.

Nothing fit or selected here ever touches an OOS row until step 5's scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from sklearn.metrics import roc_auc_score
from sqlalchemy.orm import Session

from app.config import Settings
from app.domain.timeframes import get_timeframe
from app.services.backtester import metrics as M
from app.services.backtester.runner import fold_windows, slice_folds
from app.services.ml.dataset import Dataset, load_dataset, to_records
from app.services.ml.model import build_model, fit_with_early_stopping
from app.services.ml.pipeline import build_encoder
from app.services.ml.policy import ThresholdChoice, select_threshold


@dataclass
class FoldEvaluation:
    """One fold's ML-filtered vs unfiltered-baseline outcome."""

    fold: int
    window: dict
    is_trade_count: int
    oos_trade_count: int
    train_count: int
    val_count: int
    val_auc: float
    threshold: float
    kept: int
    keep_fraction: float
    val_expectancy: float
    ml_filtered_metrics: dict
    baseline_metrics: dict
    ml_gate: dict = field(default_factory=dict)
    baseline_gate: dict = field(default_factory=dict)
    degenerate_reason: str | None = None


@dataclass
class WalkForwardResult:
    """Full S1 evaluation across all folds + combined + gate verdicts."""

    n_trials: int
    label_threshold_r: float
    folds: list[FoldEvaluation]
    ml_passed: bool
    ml_gate_detail: dict
    baseline_passed: bool
    baseline_gate_detail: dict
    ml_combined_metrics: dict
    baseline_combined_metrics: dict


def evaluate_walk_forward(
    db: Session,
    settings: Settings,
    n_trials: int,
    dataset: Dataset | None = None,
) -> WalkForwardResult:
    """Run the full S1 walk-forward evaluation.

    Args:
        db: SQLAlchemy session (reads the corpus; no writes).
        settings: config — fold geometry, hyperparameters, gate thresholds.
        n_trials: honest independent-backtest count for the deflated-Sharpe
            selection-bias correction (from the durable counter).
        dataset: optionally a pre-loaded :class:`Dataset` (else loaded here).

    Returns:
        A :class:`WalkForwardResult` with per-fold + combined metrics and both the
        ML-filtered and unfiltered-baseline promotion-gate verdicts.
    """
    if dataset is None:
        dataset = load_dataset(db, settings.ML_LABEL_THRESHOLD_R)

    windows = _build_windows(settings)
    # Slice by signal_time, carrying each trade's row index so we can subset the frame.
    records = [
        {"signal_time": st, "idx": i}
        for i, st in enumerate(dataset.frame["signal_time"].tolist())
    ]
    sliced = slice_folds(records, windows)

    x_all = dataset.X
    y_all = dataset.y
    rr_all = dataset.rr

    fold_evals: list[FoldEvaluation] = []
    ml_oos_sets: list[list[dict]] = []
    baseline_oos_sets: list[list[dict]] = []

    for i, (fw, fold) in enumerate(zip(windows, sliced)):
        is_idx = [r["idx"] for r in fold["is"]]
        oos_idx = [r["idx"] for r in fold["oos"]]
        window = {
            "is_start": fw.is_start.isoformat(),
            "is_end": fw.is_end.isoformat(),
            "oos_start": fw.oos_start.isoformat(),
            "oos_end": fw.oos_end.isoformat(),
        }

        oos_frame = dataset.frame.iloc[oos_idx]
        baseline_records = to_records(oos_frame)

        n_val = int(round(len(is_idx) * settings.ML_VALIDATION_FRACTION))
        degenerate = _degenerate_reason(len(is_idx), n_val, len(oos_idx))
        if degenerate is not None:
            # No trainable model: keep-all fallback (threshold 0) so the fold is
            # still reported honestly rather than crashing the run.
            ml_records = baseline_records
            fold_evals.append(FoldEvaluation(
                fold=i + 1, window=window,
                is_trade_count=len(is_idx), oos_trade_count=len(oos_idx),
                train_count=max(0, len(is_idx) - n_val), val_count=n_val,
                val_auc=float("nan"), threshold=0.0, kept=len(ml_records),
                keep_fraction=1.0 if ml_records else 0.0, val_expectancy=float("nan"),
                ml_filtered_metrics=_metrics_block(ml_records, n_trials, settings),
                baseline_metrics=_metrics_block(baseline_records, n_trials, settings),
                degenerate_reason=degenerate,
            ))
            ml_oos_sets.append(ml_records)
            baseline_oos_sets.append(baseline_records)
            continue

        train_idx = is_idx[:-n_val]
        val_idx = is_idx[-n_val:]

        encoder = build_encoder()
        encoder.fit(x_all.iloc[train_idx])          # LEAK-6: fit on training slice only
        x_train = encoder.transform(x_all.iloc[train_idx])
        x_val = encoder.transform(x_all.iloc[val_idx])
        x_oos = encoder.transform(oos_frame[x_all.columns])

        model = build_model(settings, early_stopping=True)
        fit_with_early_stopping(model, x_train, y_all[train_idx], x_val, y_all[val_idx])

        p_val = model.predict_proba(x_val)[:, 1]
        rr_val = rr_all[val_idx]
        choice: ThresholdChoice = select_threshold(p_val, rr_val, settings.ML_MIN_KEEP_FRACTION)

        p_oos = model.predict_proba(x_oos)[:, 1]
        keep_mask = p_oos >= choice.threshold
        ml_records = to_records(oos_frame[keep_mask])

        fold_evals.append(FoldEvaluation(
            fold=i + 1, window=window,
            is_trade_count=len(is_idx), oos_trade_count=len(oos_idx),
            train_count=len(train_idx), val_count=len(val_idx),
            val_auc=_safe_auc(y_all[val_idx], p_val),
            threshold=choice.threshold, kept=len(ml_records),
            keep_fraction=len(ml_records) / len(oos_idx) if oos_idx else 0.0,
            val_expectancy=choice.val_expectancy,
            ml_filtered_metrics=_metrics_block(ml_records, n_trials, settings),
            baseline_metrics=_metrics_block(baseline_records, n_trials, settings),
        ))
        ml_oos_sets.append(ml_records)
        baseline_oos_sets.append(baseline_records)

    # ── promotion gate: ML-filtered vs unfiltered baseline on the same folds ──────
    ml_passed, ml_gate = M.evaluate_promotion_gate(ml_oos_sets, n_trials, settings)
    base_passed, base_gate = M.evaluate_promotion_gate(baseline_oos_sets, n_trials, settings)
    for fe, g_ml, g_base in zip(fold_evals, ml_gate["folds"], base_gate["folds"]):
        fe.ml_gate = g_ml
        fe.baseline_gate = g_base

    ml_combined = _metrics_block([t for s in ml_oos_sets for t in s], n_trials, settings)
    base_combined = _metrics_block([t for s in baseline_oos_sets for t in s], n_trials, settings)

    return WalkForwardResult(
        n_trials=n_trials,
        label_threshold_r=settings.ML_LABEL_THRESHOLD_R,
        folds=fold_evals,
        ml_passed=ml_passed,
        ml_gate_detail=ml_gate,
        baseline_passed=base_passed,
        baseline_gate_detail=base_gate,
        ml_combined_metrics=ml_combined,
        baseline_combined_metrics=base_combined,
    )


def keep_fraction_report(result: WalkForwardResult) -> list[dict]:
    """Per-fold OOS keep-fraction summary — a first-class filter-collapse watch.

    QA H1 (S1-v1): the ML filter can quietly collapse to a tiny kept slice on an
    OOS fold (v1's fold-3 kept 46 of 328 = 14%), which makes that fold's metrics a
    small-sample artefact rather than a real edge. Surfacing kept/total + the
    fraction NEXT TO the gate table makes such collapses impossible to miss.

    Returns:
        One dict per fold: ``fold``, ``kept``, ``oos_total``, ``keep_fraction``,
        ``threshold``, ``degenerate_reason`` — in fold order.
    """
    return [
        {
            "fold": fe.fold,
            "kept": fe.kept,
            "oos_total": fe.oos_trade_count,
            "keep_fraction": fe.keep_fraction,
            "threshold": fe.threshold,
            "degenerate_reason": fe.degenerate_reason,
        }
        for fe in result.folds
    ]


# ── helpers ───────────────────────────────────────────────────────────────────
def _build_windows(settings: Settings):
    """Reconstruct M7's fold windows from config — identical geometry to the runner."""
    window_start = _naive(settings.EXECUTION_WINDOW_START)
    window_end = _naive(settings.EXECUTION_WINDOW_END)
    period_hours = get_timeframe(_first_granularity(settings)).period_hours
    embargo_delta = timedelta(hours=int(settings.BACKTEST_EMBARGO_BARS) * period_hours)
    bounds = [float(x.strip()) for x in settings.BACKTEST_FOLD_BOUNDS.split(",") if x.strip()]
    return fold_windows(window_start, window_end, bounds, embargo_delta)


def _first_granularity(settings: Settings) -> str:
    for g in settings.SIGNAL_GRANULARITIES.split(","):
        g = g.strip()
        if g:
            return g
    raise RuntimeError("SIGNAL_GRANULARITIES is empty — cannot determine the trading timeframe")


def _degenerate_reason(n_is: int, n_val: int, n_oos: int) -> str | None:
    if n_oos == 0:
        return "empty_oos"
    if n_val < 1:
        return "validation_tail_empty"
    if n_is - n_val < 1:
        return "training_slice_empty"
    return None


def _metrics_block(trades: list[dict], n_trials: int, settings: Settings) -> dict:
    risk_pct = settings.RISK_PCT_PER_TRADE
    return {
        "trade_count": len(trades),
        "profit_factor": M.profit_factor(trades),
        "expectancy": M.expectancy(trades),
        "max_drawdown": M.max_drawdown(trades, risk_pct),
        "sharpe": M.sharpe(trades),
        "deflated_sharpe": M.deflated_sharpe(trades, n_trials),
        "probabilistic_sharpe": M.probabilistic_sharpe(trades),
        "win_rate": M.win_rate(trades),  # reporting-only — NOT part of the gate
        "avg_holding_hours": M.avg_holding_hours(trades),
        "outcome_breakdown": M.outcome_breakdown(trades),
    }


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return float("nan")  # AUC undefined when the validation tail is single-class
    return float(roc_auc_score(y_true, scores))


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
