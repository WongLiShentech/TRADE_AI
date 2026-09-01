"""Backtest metrics + the M7 promotion gate.

Pure functions over a list of trade records — each record may be a ``dict``
(``{"rr_actual": ..., "holding_hours": ...}``) or an ORM row / any object exposing
those as attributes (e.g. ``app.models.trade.Trade``). Nothing here touches the
database; the Part B runner supplies the trade lists (e.g. one OOS fold's rows)
and persists the results onto ``BacktestRun``.

R-multiple convention
----------------------
Every function operates on ``rr_actual`` (the realised R-multiple from
``backtester.simulator.ExitResult.rr_actual``) — a per-trade, risk-normalised
return. No dollar P&L, pip values or position sizing enter this module (those
are ``RiskEngine`` concerns) — this keeps every function instrument-agnostic.

Sharpe / Deflated Sharpe / Probabilistic Sharpe
------------------------------------------------
``sharpe`` is the raw per-trade Sharpe ratio (mean R / stdev R) with **no time
annualisation** — trades don't arrive on a fixed calendar cadence, so scaling by
sqrt(252) or similar would be meaningless here. It is a trade-level ratio only.

``probabilistic_sharpe`` (PSR) and ``deflated_sharpe`` (DSR) implement
Bailey, D. H., & López de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
Journal of Portfolio Management, 40(5), 94-107::

    PSR(SR*) = Φ( (SR - SR*) * sqrt(N-1) / sqrt(1 - γ3*SR + ((γ4-1)/4)*SR^2) )

where ``SR`` is the sample Sharpe of the trade R-series, ``N`` its sample size,
``γ3`` the sample skewness and ``γ4`` the sample (non-excess, i.e. normal == 3)
kurtosis. ``probabilistic_sharpe`` evaluates this at a caller-supplied benchmark
``SR*`` (default 0.0 — "is the true Sharpe greater than zero?").

``deflated_sharpe`` evaluates the same PSR formula at ``SR* = E[max SR_n]``, the
expected maximum Sharpe ratio one would observe by chance across ``n_trials``
independent backtests (repeated backtesting inflates the apparent best Sharpe by
~sqrt(n_trials); DSR corrects for that selection bias)::

    E[max SR_n] ~= sqrt(1 / (N - 1)) * ( (1-gamma)*Phi^-1(1 - 1/n_trials)
                                        + gamma  *Phi^-1(1 - 1/(n_trials*e)) )

with ``gamma ~= 0.5772156649`` (the Euler-Mascheroni constant), ``N`` the sample
size of THIS trade series (used to estimate the variance of the Sharpe
estimator), and ``n_trials`` the number of independent backtest configurations
evaluated against this dataset (per the M7 plan: the ``BacktestRun`` row count,
supplied by the caller — never hardcoded to 1, which would silently disable the
deflation).

``Φ`` / ``Φ⁻¹`` use ``statistics.NormalDist`` (stdlib — no numpy/scipy dependency
in this project).

Max drawdown — account terms, not raw R
-----------------------------------------
``max_drawdown`` takes a ``risk_pct`` (``settings.RISK_PCT_PER_TRADE``) and builds
a COMPOUNDING account-equity curve (``E_i = E_{i-1} * (1 + rr_actual_i *
risk_pct)``), not a raw cumulative-R curve. See the function docstring for the
full rationale — in short, ``rr_actual`` is risk-normalised (unitless R), so
without scaling by the fraction of the account actually risked per trade, a 4R
losing streak would read as a "400% drawdown" against ``BACKTEST_PROMOTION_MAX_DD_MAX``
(an ACCOUNT-drawdown ceiling), making the gate unpassable by construction.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Sequence

from app.config import Settings

_NAN = float("nan")
_EULER_MASCHERONI = 0.5772156649015329
_NORMAL = NormalDist()

# Floors compounding equity so a pathological loss streak (rr*risk_pct <= -1, an
# account wipeout) cannot invert the curve to a negative value — the real-world
# analogue is a margin call / stop-out at zero equity (a 100% drawdown), not a
# negative one. See max_drawdown().
_EQUITY_FLOOR = 1e-9

# Outcome-bucket thresholds (module constants, not hyperparameters — the
# labeling convention itself, per the M7 spec's Component 1c column definitions).
# Priority order matters: BREAKEVEN is the narrow band around zero and is
# evaluated first so LOSS/PARTIAL never double-count its boundary trades.
FULL_WIN_RR_MIN = 1.9      # rr_actual >= this -> full_win
BREAKEVEN_RR_ABS_MAX = 0.05  # |rr_actual| <= this -> breakeven (checked before loss/partial)
LOSS_RR_MAX = -0.05        # rr_actual <= this (and outside breakeven) -> loss
# Anything left over (BREAKEVEN_RR_ABS_MAX < rr_actual < FULL_WIN_RR_MIN) -> partial.


# ── record accessors (dict or ORM row) ────────────────────────────────────────
def _get(trade: Any, key: str) -> Any:
    if isinstance(trade, dict):
        return trade.get(key)
    return getattr(trade, key, None)


def _rr(trade: Any) -> float:
    v = _get(trade, "rr_actual")
    return float(v) if v is not None else _NAN


def _hold_hours(trade: Any) -> float:
    v = _get(trade, "holding_hours")
    return float(v) if v is not None else _NAN


def _rr_series(trades: Sequence[Any]) -> list[float]:
    return [_rr(t) for t in trades if not _isnan(_rr(t))]


def _isnan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ── core metrics ──────────────────────────────────────────────────────────────
def profit_factor(trades: Sequence[Any]) -> float:
    """Gross wins / gross losses (both in R). NaN if there are no trades (or no
    trades with a resolved rr_actual). ``inf`` if there are wins but zero losses
    (uncapped by design — the promotion gate's ``> BACKTEST_PROMOTION_PROFIT_FACTOR_MIN``
    check treats ``inf`` as an unambiguous pass; callers that persist the value
    should be aware it can be non-finite)."""
    rrs = _rr_series(trades)
    if not rrs:
        return _NAN
    gross_win = sum(r for r in rrs if r > 0)
    gross_loss = -sum(r for r in rrs if r < 0)
    if gross_loss == 0.0:
        return math.inf if gross_win > 0 else _NAN
    return gross_win / gross_loss


def expectancy(trades: Sequence[Any]) -> float:
    """Mean realised R across all trades. NaN if there are no trades."""
    rrs = _rr_series(trades)
    if not rrs:
        return _NAN
    return sum(rrs) / len(rrs)


def max_drawdown(trades: Sequence[Any], risk_pct: float) -> float:
    """Maximum drawdown on a COMPOUNDING account-equity curve, as a fraction of peak.

    ``risk_pct`` is the fraction of the CURRENT account balance risked per trade —
    ``settings.RISK_PCT_PER_TRADE``, the platform's locked position-sizing input
    (CLAUDE.md: "risk %, stop distance, and position size are a locked triangle";
    ``risk_amount = balance * RISK_PCT_PER_TRADE`` is recomputed fresh per trade off
    the CURRENT balance — never a fixed dollar/R amount). The equity curve must
    therefore compound to be honest: starting from a normalised baseline of 1.0
    account-unit, ``E_i = E_{i-1} * (1 + rr_actual_i * risk_pct)`` (chronological
    list order). Peak is the running max of the curve (including the baseline);
    drawdown at each point is ``(peak - E_i) / peak``. Returns the max such
    fraction, always >= 0.

    Why compounding over additive: an additive R-curve (``E_i = E_{i-1} +
    rr_actual_i``, no ``risk_pct``) implicitly assumes a FIXED per-trade dollar
    risk that never shrinks after a loss or grows after a win. That conflates "R
    multiples lost" with "% of account lost" — e.g. a 4R losing streak reads as a
    "400% drawdown" instead of the true ~4% of account at 1% risk/trade, making
    ``BACKTEST_PROMOTION_MAX_DD_MAX`` (an ACCOUNT-drawdown ceiling, e.g. 0.25 =
    25% of peak equity — per the M7 plan) unpassable by construction. Compounding
    on ``risk_pct`` matches how the platform actually sizes trades and restores
    the gate to its intended scale.

    Equity is floored at ``_EQUITY_FLOOR`` (a small positive epsilon) so a
    pathological loss streak (``rr_actual * risk_pct <= -1``, i.e. an account
    wipeout) cannot invert the curve to a negative value — the real-world
    analogue is a margin call / stop-out at zero equity (a 100% drawdown), not a
    negative one.

    NaN if there are no trades.
    """
    rrs = _rr_series(trades)
    if not rrs:
        return _NAN
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in rrs:
        equity = max(equity * (1.0 + r * risk_pct), _EQUITY_FLOOR)
        peak = max(peak, equity)
        worst = max(worst, (peak - equity) / peak)
    return worst


def sharpe(trades: Sequence[Any]) -> float:
    """Per-trade Sharpe ratio: mean(R) / stdev(R), sample stdev (ddof=1).
    Annualisation-free (see module docstring) — a raw ratio over the trade
    sequence, not a calendar-scaled figure. NaN if N < 2 or stdev == 0."""
    rrs = _rr_series(trades)
    if len(rrs) < 2:
        return _NAN
    mean = sum(rrs) / len(rrs)
    var = sum((r - mean) ** 2 for r in rrs) / (len(rrs) - 1)
    std = math.sqrt(var)
    if std == 0.0:
        return _NAN
    return mean / std


def _sample_moments(rrs: Sequence[float]) -> tuple[float, float, float, float] | None:
    """Return (mean, std, skew, kurtosis) using sample (ddof=1) central moments.
    Kurtosis is NON-excess (normal == 3), matching the Bailey/Lopez de Prado PSR
    formula. Returns None if N < 3 (skew/kurtosis undefined) or stdev == 0."""
    n = len(rrs)
    if n < 3:
        return None
    mean = sum(rrs) / n
    var = sum((r - mean) ** 2 for r in rrs) / (n - 1)
    std = math.sqrt(var)
    if std == 0.0:
        return None
    m3 = sum((r - mean) ** 3 for r in rrs) / n
    m4 = sum((r - mean) ** 4 for r in rrs) / n
    skew = m3 / std**3
    kurt = m4 / std**4
    return mean, std, skew, kurt


def probabilistic_sharpe(trades: Sequence[Any], sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) — Bailey & Lopez de Prado (2014) PSR, see
    module docstring for the formula. NaN if N < 3 or stdev == 0 (skew/kurtosis
    / Sharpe undefined for a sample that small — the promotion gate fails on
    minimum sample size regardless)."""
    rrs = _rr_series(trades)
    moments = _sample_moments(rrs)
    if moments is None:
        return _NAN
    mean, std, skew, kurt = moments
    n = len(rrs)
    sr = mean / std
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if denom <= 0.0:
        return _NAN
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return _NORMAL.cdf(z)


def deflated_sharpe(trades: Sequence[Any], n_trials: int) -> float:
    """PSR evaluated at SR* = the expected maximum Sharpe ratio observable by
    chance across ``n_trials`` independent backtests (Bailey & Lopez de Prado
    2014 — see module docstring for both formulas). ``n_trials`` must be >= 1
    (the number of independent backtest configurations tried against this
    dataset, e.g. the BacktestRun row count — never hardcoded to 1, which would
    silently disable the deflation). NaN if N < 3, stdev == 0, or n_trials < 1."""
    if n_trials < 1:
        return _NAN
    rrs = _rr_series(trades)
    moments = _sample_moments(rrs)
    if moments is None:
        return _NAN
    n = len(rrs)

    if n_trials == 1:
        expected_max_sr = 0.0  # a single trial has no selection bias to deflate
    else:
        inv1 = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
        inv2 = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
        expected_max_sr = math.sqrt(1.0 / (n - 1)) * (
            (1.0 - _EULER_MASCHERONI) * inv1 + _EULER_MASCHERONI * inv2
        )
    return probabilistic_sharpe(trades, sr_benchmark=expected_max_sr)


def outcome_breakdown(trades: Sequence[Any]) -> dict:
    """Bucket counts by realised R (see module constants for thresholds):
    ``full_win`` (rr >= FULL_WIN_RR_MIN), ``breakeven`` (|rr| <= BREAKEVEN_RR_ABS_MAX,
    checked before loss/partial so boundary trades are never double-counted),
    ``loss`` (rr <= LOSS_RR_MAX, outside the breakeven band), ``partial``
    (everything else — the open band between breakeven and full_win). The four
    counts always sum to ``len(trades)`` with a resolved rr_actual."""
    counts = {"full_win": 0, "partial": 0, "breakeven": 0, "loss": 0}
    for r in _rr_series(trades):
        if abs(r) <= BREAKEVEN_RR_ABS_MAX:
            counts["breakeven"] += 1
        elif r <= LOSS_RR_MAX:
            counts["loss"] += 1
        elif r >= FULL_WIN_RR_MIN:
            counts["full_win"] += 1
        else:
            counts["partial"] += 1
    return counts


def win_rate(trades: Sequence[Any]) -> float:
    """Fraction of trades that are wins — reporting-only metric, NOT part of the
    promotion gate (see ``evaluate_promotion_gate``, unchanged by this function).

    "Win" reuses the EXACT win/breakeven boundary already used by
    ``outcome_breakdown`` (``BREAKEVEN_RR_ABS_MAX``) and the runner's
    ``label_outcome`` (``rr > 0.05``) — no new magic number. Computed as
    ``(full_win + partial) / total`` from :func:`outcome_breakdown`: those are
    exactly the two buckets whose realised R exceeds ``BREAKEVEN_RR_ABS_MAX``
    (``outcome_breakdown`` claims the ``<= BREAKEVEN_RR_ABS_MAX`` boundary for
    breakeven FIRST, so full_win+partial never double-counts it), i.e. identical
    to the set ``rr_actual > BREAKEVEN_RR_ABS_MAX`` — matching ``label_outcome``'s
    ``rr > 0.05`` "win" definition exactly (both thresholds are 0.05).

    NaN if there are no trades with a resolved ``rr_actual`` — matching every
    other ratio-style metric in this module (``profit_factor``, ``expectancy``,
    ``sharpe``, etc.), which return NaN rather than a misleading 0.0 for an
    undefined/empty sample.
    """
    total = len(_rr_series(trades))
    if total == 0:
        return _NAN
    breakdown = outcome_breakdown(trades)
    return (breakdown["full_win"] + breakdown["partial"]) / total


def avg_holding_hours(trades: Sequence[Any]) -> float:
    """Mean holding_hours across trades with a resolved value. NaN if none."""
    hours = [h for h in (_hold_hours(t) for t in trades) if not _isnan(h)]
    if not hours:
        return _NAN
    return sum(hours) / len(hours)


# ── promotion gate ────────────────────────────────────────────────────────────
def evaluate_promotion_gate(
    fold_results: Sequence[Sequence[Any]],
    n_trials: int,
    settings: Settings,
) -> tuple[bool, dict]:
    """Evaluate the M7 promotion gate over an arbitrary number of OOS folds
    (M7 uses three; this function does not hardcode that count).

    Per-fold requirement (ALL folds must pass, per the M7 plan's Component 5):
    ``profit_factor > BACKTEST_PROMOTION_PROFIT_FACTOR_MIN``,
    ``expectancy > BACKTEST_PROMOTION_EXPECTANCY_MIN``,
    ``max_drawdown < BACKTEST_PROMOTION_MAX_DD_MAX``,
    ``oos_sample_size >= BACKTEST_PROMOTION_MIN_OOS_TRADES``,
    ``deflated_sharpe(fold, n_trials) > 0``.

    Additionally, the deflated Sharpe of ALL folds concatenated must also be > 0
    (the plan: "Deflated Sharpe > 0 — per fold, AND on combined OOS").

    Args:
        fold_results: one sequence of trade records per OOS fold.
        n_trials: number of independent backtest configurations evaluated
            against this dataset (e.g. the BacktestRun row count) — fed straight
            into :func:`deflated_sharpe` for every fold and the combined set.
        settings: promotion thresholds (zero hardcoding).

    Returns:
        ``(passed, detail)`` — ``detail`` has ``"folds"`` (one metrics dict per
        fold, each carrying its own ``"passed"`` bool + individual gate flags)
        and ``"combined"`` (metrics dict for all folds concatenated, including
        ``"deflated_sharpe_positive"``).
    """
    fold_details: list[dict] = []
    all_folds_pass = True

    for trades in fold_results:
        pf = profit_factor(trades)
        exp = expectancy(trades)
        dd = max_drawdown(trades, settings.RISK_PCT_PER_TRADE)
        n_oos = len(_rr_series(trades))
        dsr = deflated_sharpe(trades, n_trials)

        gate = {
            "profit_factor": pf,
            "expectancy": exp,
            "max_drawdown": dd,
            "oos_sample_size": n_oos,
            "deflated_sharpe": dsr,
            "profit_factor_pass": _gt(pf, settings.BACKTEST_PROMOTION_PROFIT_FACTOR_MIN),
            "expectancy_pass": _gt(exp, settings.BACKTEST_PROMOTION_EXPECTANCY_MIN),
            "max_drawdown_pass": _lt(dd, settings.BACKTEST_PROMOTION_MAX_DD_MAX),
            "oos_sample_size_pass": n_oos >= settings.BACKTEST_PROMOTION_MIN_OOS_TRADES,
            "deflated_sharpe_pass": _gt(dsr, 0.0),
        }
        gate["passed"] = all(
            gate[k] for k in (
                "profit_factor_pass", "expectancy_pass", "max_drawdown_pass",
                "oos_sample_size_pass", "deflated_sharpe_pass",
            )
        )
        all_folds_pass = all_folds_pass and gate["passed"]
        fold_details.append(gate)

    combined_trades = [t for fold in fold_results for t in fold]
    combined_dsr = deflated_sharpe(combined_trades, n_trials)
    combined = {
        "oos_sample_size": len(_rr_series(combined_trades)),
        "deflated_sharpe": combined_dsr,
        "deflated_sharpe_positive": _gt(combined_dsr, 0.0),
    }

    passed = all_folds_pass and combined["deflated_sharpe_positive"] and len(fold_results) > 0
    return passed, {"folds": fold_details, "combined": combined}


def _gt(value: float, threshold: float) -> bool:
    return not _isnan(value) and value > threshold


def _lt(value: float, threshold: float) -> bool:
    return not _isnan(value) and value < threshold
