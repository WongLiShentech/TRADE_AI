"""Unit tests for app.services.backtester.metrics.

Pure unit tests over hand-built toy trade lists (dicts) — no DB involved.

Run: python -m pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from app.services.backtester.metrics import (
    avg_holding_hours,
    deflated_sharpe,
    evaluate_promotion_gate,
    expectancy,
    max_drawdown,
    outcome_breakdown,
    probabilistic_sharpe,
    profit_factor,
    sharpe,
    win_rate,
)


def _isnan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _t(rr: float, hold: float = 10.0) -> dict:
    return {"rr_actual": rr, "holding_hours": hold}


@dataclass
class _FakePromotionSettings:
    BACKTEST_PROMOTION_PROFIT_FACTOR_MIN: float = 1.3
    BACKTEST_PROMOTION_EXPECTANCY_MIN: float = 0.15
    BACKTEST_PROMOTION_MAX_DD_MAX: float = 0.25
    BACKTEST_PROMOTION_MIN_OOS_TRADES: int = 50
    RISK_PCT_PER_TRADE: float = 0.01


# ── profit_factor ─────────────────────────────────────────────────────────────
def test_profit_factor_hand_computed():
    # gross win = 2+3=5, gross loss = |-1|+|-2| = 3 -> PF = 5/3
    trades = [_t(2.0), _t(3.0), _t(-1.0), _t(-2.0)]
    assert profit_factor(trades) == pytest.approx(5.0 / 3.0)


def test_profit_factor_no_losses_is_inf():
    trades = [_t(1.0), _t(2.0)]
    assert profit_factor(trades) == math.inf


def test_profit_factor_empty_is_nan():
    assert _isnan(profit_factor([]))


# ── expectancy ────────────────────────────────────────────────────────────────
def test_expectancy_hand_computed():
    trades = [_t(2.0), _t(-1.0), _t(0.5)]
    assert expectancy(trades) == pytest.approx((2.0 - 1.0 + 0.5) / 3.0)


def test_expectancy_empty_is_nan():
    assert _isnan(expectancy([]))


# ── max_drawdown (account-terms, compounding on risk_pct) ────────────────────
def test_max_drawdown_hand_computed():
    # Compounding equity at risk_pct=0.01 (baseline 1.0):
    #   1.0 -> *(1+2.0*0.01)=1.02   (new peak 1.02)
    #       -> *(1+(-4.0)*0.01)=1.02*0.96=0.9792
    #       -> *(1+0.5*0.01)=0.9792*1.005=0.984096
    # dd at each step: (1.02-1.02)/1.02=0, (1.02-0.9792)/1.02=0.04 (exact),
    #                  (1.02-0.984096)/1.02=0.0352 (approx)
    # -> deepest drawdown is right after the -4.0R trade: exactly 0.04 (4% of account).
    trades = [_t(2.0), _t(-4.0), _t(0.5)]
    dd = max_drawdown(trades, risk_pct=0.01)
    assert dd == pytest.approx(0.04)


def test_max_drawdown_scales_with_risk_pct():
    # Same R-sequence at double the risk_pct -> roughly double the account drawdown
    # (not exactly 2x due to compounding, but materially larger).
    trades = [_t(2.0), _t(-4.0), _t(0.5)]
    dd_1pct = max_drawdown(trades, risk_pct=0.01)
    dd_2pct = max_drawdown(trades, risk_pct=0.02)
    assert dd_2pct > dd_1pct
    assert dd_1pct < 0.10   # a 4R loss at 1% risk/trade is a small account drawdown...
    assert dd_1pct != pytest.approx(4.0 / 3.0)  # ...never the old (wrong) raw-R scale


def test_max_drawdown_no_losses_is_zero():
    trades = [_t(1.0), _t(2.0)]
    assert max_drawdown(trades, risk_pct=0.01) == pytest.approx(0.0)


def test_max_drawdown_empty_is_nan():
    assert _isnan(max_drawdown([], risk_pct=0.01))


# ── sharpe ────────────────────────────────────────────────────────────────────
def test_sharpe_hand_computed():
    rrs = [1.0, 2.0, -1.0, 0.5]
    mean = sum(rrs) / len(rrs)
    var = sum((r - mean) ** 2 for r in rrs) / (len(rrs) - 1)
    expected = mean / math.sqrt(var)
    trades = [_t(r) for r in rrs]
    assert sharpe(trades) == pytest.approx(expected)


def test_sharpe_single_trade_is_nan():
    assert _isnan(sharpe([_t(1.0)]))


# ── outcome_breakdown ─────────────────────────────────────────────────────────
def test_outcome_breakdown_buckets():
    trades = [
        _t(2.0),    # full_win (>= 1.9)
        _t(1.9),    # full_win (boundary, inclusive)
        _t(1.0),    # partial
        _t(0.05),   # breakeven (boundary, inclusive)
        _t(-0.05),  # breakeven (boundary, inclusive)
        _t(-0.06),  # loss
        _t(-2.0),   # loss
    ]
    counts = outcome_breakdown(trades)
    assert counts == {"full_win": 2, "partial": 1, "breakeven": 2, "loss": 2}
    assert sum(counts.values()) == len(trades)


def test_outcome_breakdown_empty():
    assert outcome_breakdown([]) == {"full_win": 0, "partial": 0, "breakeven": 0, "loss": 0}


# ── win_rate (reporting-only; reuses outcome_breakdown's win/breakeven boundary) ──
def test_win_rate_hand_computed():
    # Same 7-trade mix as test_outcome_breakdown_buckets: full_win=2, partial=1,
    # breakeven=2, loss=2 -> win_rate = (full_win + partial) / total = 3/7.
    trades = [
        _t(2.0), _t(1.9), _t(1.0), _t(0.05), _t(-0.05), _t(-0.06), _t(-2.0),
    ]
    assert win_rate(trades) == pytest.approx(3.0 / 7.0)


def test_win_rate_matches_label_outcome_boundary():
    # rr=0.05 is breakeven (not a win) under outcome_breakdown's <= boundary,
    # exactly matching the runner's label_outcome ("win" only if rr > 0.05).
    trades = [_t(0.06), _t(0.05), _t(-0.05)]
    assert win_rate(trades) == pytest.approx(1.0 / 3.0)


def test_win_rate_all_losses_is_zero():
    trades = [_t(-1.0), _t(-2.0), _t(-0.5)]
    assert win_rate(trades) == pytest.approx(0.0)


def test_win_rate_empty_is_nan():
    # Matches every other ratio-style metric in this module (profit_factor,
    # expectancy, sharpe, ...): NaN for an undefined/empty sample, not 0.0.
    assert _isnan(win_rate([]))


# ── avg_holding_hours ─────────────────────────────────────────────────────────
def test_avg_holding_hours():
    trades = [_t(1.0, hold=10.0), _t(-1.0, hold=20.0), _t(0.5, hold=30.0)]
    assert avg_holding_hours(trades) == pytest.approx(20.0)


def test_avg_holding_hours_empty_is_nan():
    assert _isnan(avg_holding_hours([]))


# ── probabilistic / deflated sharpe ───────────────────────────────────────────
def _toy_series(n: int, seed_offset: float = 0.0) -> list[dict]:
    """A synthetic, mildly-positive, non-degenerate R series (deterministic, no
    RNG dependency) so skew/kurtosis are well-defined and PSR/DSR are computable."""
    rrs = [0.3 + 0.5 * math.sin(i + seed_offset) for i in range(n)]
    return [_t(r) for r in rrs]


def test_psr_needs_at_least_3_trades():
    assert _isnan(probabilistic_sharpe(_toy_series(2)))
    assert not _isnan(probabilistic_sharpe(_toy_series(10)))


def test_psr_zero_stdev_is_nan():
    assert _isnan(probabilistic_sharpe([_t(1.0), _t(1.0), _t(1.0)]))


def test_dsr_decreases_as_n_trials_grows():
    trades = _toy_series(30)
    dsr_1 = deflated_sharpe(trades, n_trials=1)
    dsr_10 = deflated_sharpe(trades, n_trials=10)
    dsr_100 = deflated_sharpe(trades, n_trials=100)
    assert dsr_1 > dsr_10 > dsr_100


def test_dsr_invalid_n_trials_is_nan():
    assert _isnan(deflated_sharpe(_toy_series(30), n_trials=0))


def test_dsr_small_sample_is_nan():
    assert _isnan(deflated_sharpe(_toy_series(2), n_trials=10))


# ── promotion gate ─────────────────────────────────────────────────────────────
def _passing_fold(n: int = 60) -> list[dict]:
    """A fold engineered to clear every threshold in _FakePromotionSettings:
    PF > 1.3, expectancy > 0.15, max DD < 0.25, n >= 50, DSR > 0."""
    rrs = [0.3 if i % 3 != 0 else -0.1 for i in range(n)]
    return [_t(r) for r in rrs]


def test_promotion_gate_all_folds_pass():
    folds = [_passing_fold(), _passing_fold(), _passing_fold()]
    passed, detail = evaluate_promotion_gate(folds, n_trials=1, settings=_FakePromotionSettings())
    assert passed is True
    assert len(detail["folds"]) == 3
    assert all(f["passed"] for f in detail["folds"])
    assert detail["combined"]["deflated_sharpe_positive"] is True


def test_promotion_gate_fails_on_min_trades():
    # Fold 3 has only 10 trades (< min 50) -> that fold fails -> overall fails.
    folds = [_passing_fold(), _passing_fold(), _passing_fold(n=10)]
    passed, detail = evaluate_promotion_gate(folds, n_trials=1, settings=_FakePromotionSettings())
    assert passed is False
    assert detail["folds"][2]["oos_sample_size_pass"] is False
    assert detail["folds"][2]["passed"] is False
    # The other two folds still individually pass.
    assert detail["folds"][0]["passed"] is True
    assert detail["folds"][1]["passed"] is True


def test_promotion_gate_fails_on_poor_expectancy():
    losing_fold = [_t(0.1 if i % 5 == 0 else -0.05) for i in range(60)]
    folds = [_passing_fold(), _passing_fold(), losing_fold]
    passed, detail = evaluate_promotion_gate(folds, n_trials=1, settings=_FakePromotionSettings())
    assert passed is False
    assert detail["folds"][2]["passed"] is False


def test_promotion_gate_empty_folds_fails():
    passed, detail = evaluate_promotion_gate([], n_trials=1, settings=_FakePromotionSettings())
    assert passed is False
    assert detail["folds"] == []
