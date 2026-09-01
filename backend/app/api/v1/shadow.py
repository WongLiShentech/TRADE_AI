"""M8-Shadow API — the forward evidence, one HTTP call away.

Endpoints (all under ``/api/v1/shadow``):

* ``GET /decisions``   — paginated stream of live shadow decisions, filterable by
                         instrument, model decision, resolved/pending and date range.
* ``GET /performance`` — the money endpoint: model-take vs model-skip vs all-signals
                         cohort comparison over resolved shadow rows.

Why ``/performance`` is shaped this way
---------------------------------------
The M8-Shadow experiment asks exactly one question: *does the S1 filter improve the
rule engine's live results?* That is only answerable by comparing the trades the
model ACCEPTED against the ones it DECLINED — the skip cohort is the counterfactual,
not noise to be discarded. So three cohorts are reported side by side:

* ``model_take``  — rows the model said take. What live trading WOULD have looked like.
* ``model_skip``  — rows the model said skip. What the filter SAVED you from (or cost you).
* ``all_signals`` — take ∪ skip. What the unfiltered rule engine did. The baseline.

The filter adds value iff ``model_take`` beats ``all_signals``; it is actively
harmful if ``model_skip`` beats ``model_take``.

Scoring failures (``ml_decision IS NULL``) are EXCLUDED from all three cohorts and
reported separately under ``unscored``. They belong to no decision, so folding them
into ``all_signals`` would quietly contaminate the baseline with rows the model was
never given a chance to judge. Their count is surfaced because a growing number is a
data-quality alarm, not a neutral fact.

Ambiguous resolutions are excluded too (by default)
---------------------------------------------------
A row with ``ambiguous_resolution=True`` carries an outcome the resolver itself does
not trust: an SL-first tie-break, or a degraded walk over thin M1 where the simulator
can write a fabricated flat exit (exit == entry, ``rr_actual = 0.0``,
``closed_at == opened_at``). That flat row labels as ``breakeven`` and drags every
expectancy toward zero — a measurement artefact masquerading as a result.

So ambiguous rows are EXCLUDED from all three cohorts by default and reported in
their own ``ambiguous`` block with their own stats, so they stay visible (a rising
count is an M1-ingestion alarm) without diluting the comparison the experiment is
judged on. ``?include_ambiguous=true`` folds them back in for anyone who wants the
unfiltered view, and ``/decisions?ambiguous=...`` filters the row stream directly.

Metrics are computed by ``services.backtester.metrics`` — the SAME functions that
scored the M7 backtest, so a shadow expectancy and a backtest expectancy mean the
same thing. Non-finite results (NaN for an empty cohort, ``inf`` for a cohort with
no losses) are serialised as ``null`` rather than being clamped to a number that
would read as a real measurement.

Read-only: nothing here writes a row, resolves an outcome or touches a broker.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal, Optional, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.instrument import Instrument
from app.models.trade import Trade
from app.services.backtester import metrics as M
from app.services.ml.inference import DECISION_SKIP, DECISION_TAKE
from app.services.shadow import REASONING_SHADOW_KEY, STAGE_SHADOW

router = APIRouter()

# Query-facing name for rows whose model scoring failed (ml_decision IS NULL). Never
# a cohort — a decision that was never made cannot be evaluated as one.
DECISION_UNSCORED = "unscored"
# Cohort keys in the /performance response.
COHORT_TAKE = "model_take"
COHORT_SKIP = "model_skip"
COHORT_ALL = "all_signals"
# Key of the separate (never-a-cohort) ambiguous block in the /performance response.
BLOCK_AMBIGUOUS = "ambiguous"

_SECONDS_PER_HOUR = 3600.0


# ── response schemas ─────────────────────────────────────────────────────────
class ShadowDecision(BaseModel):
    """One observe-only decision: what the rule engine fired, what the model said
    about it, what the risk engine would have done with it, and how it turned out."""

    id: int
    instrument: str
    granularity: str | None
    direction: str
    entry_price: float
    stop_price: float
    tp_price: float
    exit_price: float | None
    rr_entry: float
    rr_actual: float | None
    outcome: str | None
    exit_reason: str | None
    ambiguous_resolution: bool
    resolved: bool
    holding_hours: float | None
    ml_probability: float | None
    ml_decision: str | None
    ml_model_id: str | None
    confluence_score: int | None
    session: str | None
    risk_passed: bool | None
    risk_rejection_reason: str | None
    opened_at: datetime
    closed_at: datetime | None


class DecisionPage(BaseModel):
    """A page of decisions plus the total matching the filters (for pagination)."""

    total: int
    limit: int
    offset: int
    items: list[ShadowDecision]


class CohortStats(BaseModel):
    """Performance of one decision cohort.

    ``total`` counts every row in the cohort; ``count`` counts only the RESOLVED ones
    the metrics are computed from, and ``pending`` the rest. Reporting all three keeps
    an early, mostly-unresolved cohort from reading as a tiny but complete sample.
    """

    total: int
    count: int = Field(description="resolved rows — the sample every metric below uses")
    pending: int
    win_rate: float | None
    expectancy: float | None = Field(description="mean realised R")
    profit_factor: float | None
    avg_holding_hours: float | None
    outcome_breakdown: dict[str, int]


class UnscoredStats(BaseModel):
    """Rows the model could not score (``ml_decision IS NULL``) — excluded from every
    cohort, surfaced here because a rising count is a data-quality alarm."""

    total: int
    resolved: int
    pending: int


class FilterEffect(BaseModel):
    """The comparison the experiment exists to make.

    ``expectancy_lift_vs_all`` > 0 means the filter improved on the unfiltered rule
    engine. ``expectancy_lift_vs_skip`` <= 0 means the model preferred the WORSE half
    of its own signals — the filter is inverted and must not be promoted.
    ``keep_rate`` is the fraction of signals the model would have traded, computed
    over the RESOLVED sample — the same rows the expectancies above are computed
    from — so the two describe one consistent sample rather than two overlapping
    ones. A very low keep rate can manufacture a flattering expectancy from a handful
    of trades, so it is reported next to the lift, never separately.
    """

    keep_rate: float | None
    expectancy_lift_vs_all: float | None
    expectancy_lift_vs_skip: float | None
    profit_factor_lift_vs_all: float | None


class AmbiguousStats(BaseModel):
    """Rows whose resolution the resolver itself does not trust.

    Excluded from every cohort unless ``include_ambiguous=true``, and reported here
    with their own metrics so they remain auditable. ``included_in_cohorts`` states
    plainly which way the request was served, so a number can never be misread as
    coming from the other mode.
    """

    included_in_cohorts: bool
    stats: CohortStats


class PerformanceReport(BaseModel):
    instrument: str | None
    start: datetime | None
    end: datetime | None
    cohorts: dict[str, CohortStats]
    unscored: UnscoredStats
    ambiguous: AmbiguousStats
    filter_effect: FilterEffect


# ── endpoints ────────────────────────────────────────────────────────────────
@router.get("/decisions", response_model=DecisionPage)
def list_decisions(
    instrument: Optional[str] = Query(
        None, description="instrument symbol, e.g. EUR_USD (any symbol — never hardcoded)"
    ),
    ml_decision: Optional[Literal["take", "skip", "unscored"]] = Query(
        None, description="'unscored' selects rows whose model scoring failed"
    ),
    status: Literal["all", "resolved", "pending"] = Query(
        "all", description="'resolved' = outcome written; 'pending' = awaiting resolution"
    ),
    ambiguous: Optional[bool] = Query(
        None,
        description="true → only rows flagged ambiguous_resolution; false → only "
                    "trustworthy resolutions; omitted → both",
    ),
    start: Optional[datetime] = Query(None, description="inclusive lower bound on opened_at (T)"),
    end: Optional[datetime] = Query(None, description="inclusive upper bound on opened_at (T)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Paginated live shadow decisions, newest signal first.

    Args:
        instrument: optional symbol filter (resolved against the instruments table).
        ml_decision: optional ``take`` / ``skip`` / ``unscored`` filter.
        status: resolved / pending / all.
        ambiguous: optional ``ambiguous_resolution`` filter — ``True`` isolates the
            untrustworthy labels for inspection, ``False`` gives the clean stream,
            omitted returns both.
        start: inclusive lower bound on ``opened_at`` (the signal time T).
        end: inclusive upper bound on ``opened_at``.
        limit: page size (1-500).
        offset: page offset.
        db: SQLAlchemy session.

    Returns:
        A :class:`DecisionPage`. ``total`` is the unpaginated match count.

    Raises:
        HTTPException: 404 if ``instrument`` is not a known symbol (a typo must not
            silently return an empty page that reads as "no signals yet").
    """
    query = _filtered(db, instrument, ml_decision, status, start, end, ambiguous)
    total = query.count()
    rows = query.order_by(Trade.opened_at.desc()).offset(offset).limit(limit).all()
    symbols = _symbol_map(db, [r.instrument_id for r in rows])
    return DecisionPage(
        total=total,
        limit=limit,
        offset=offset,
        items=[_to_decision(r, symbols) for r in rows],
    )


@router.get("/performance", response_model=PerformanceReport)
def performance(
    instrument: Optional[str] = Query(None, description="optional symbol filter"),
    start: Optional[datetime] = Query(None, description="inclusive lower bound on opened_at (T)"),
    end: Optional[datetime] = Query(None, description="inclusive upper bound on opened_at (T)"),
    include_ambiguous: bool = Query(
        False,
        description="fold ambiguous_resolution rows back into the cohorts. Default "
                    "false — their outcomes are untrusted and dilute expectancy.",
    ),
    db: Session = Depends(get_db),
):
    """Model-take vs model-skip vs all-signals, over resolved shadow rows.

    Args:
        instrument: optional symbol filter.
        start: inclusive lower bound on ``opened_at``.
        end: inclusive upper bound on ``opened_at``.
        include_ambiguous: when False (default) rows flagged ``ambiguous_resolution``
            are excluded from all three cohorts and reported only in the separate
            ``ambiguous`` block. When True they are folded into the cohorts as well —
            the block is still returned, so the two views are always reconcilable.
        db: SQLAlchemy session.

    Returns:
        A :class:`PerformanceReport`. Every metric is computed by
        ``backtester.metrics`` on the RESOLVED subset of its cohort; unresolved rows
        are counted as ``pending`` and never imputed.

    Raises:
        HTTPException: 404 if ``instrument`` is not a known symbol.
    """
    rows = _filtered(db, instrument, None, "all", start, end).all()

    ambiguous_rows = [r for r in rows if _is_ambiguous(r)]
    # Cohort population: ambiguous rows are dropped unless explicitly requested. A
    # PENDING row is never ambiguous (the flag is written at resolution), so this
    # cannot silently drop rows still awaiting an outcome.
    cohort_rows = rows if include_ambiguous else [r for r in rows if not _is_ambiguous(r)]

    take = [r for r in cohort_rows if r.ml_decision == DECISION_TAKE]
    skip = [r for r in cohort_rows if r.ml_decision == DECISION_SKIP]
    scored = take + skip
    unscored = [r for r in cohort_rows if r.ml_decision is None]

    cohorts = {
        COHORT_TAKE: _cohort(take),
        COHORT_SKIP: _cohort(skip),
        COHORT_ALL: _cohort(scored),
    }
    return PerformanceReport(
        instrument=instrument,
        start=start,
        end=end,
        cohorts=cohorts,
        unscored=UnscoredStats(
            total=len(unscored),
            resolved=sum(1 for r in unscored if _is_resolved(r)),
            pending=sum(1 for r in unscored if not _is_resolved(r)),
        ),
        ambiguous=AmbiguousStats(
            included_in_cohorts=include_ambiguous,
            stats=_cohort(ambiguous_rows),
        ),
        filter_effect=_filter_effect(cohorts),
    )


# ── internals ────────────────────────────────────────────────────────────────
def _filtered(
    db: Session,
    instrument: Optional[str],
    ml_decision: Optional[str],
    status: str,
    start: Optional[datetime],
    end: Optional[datetime],
    ambiguous: Optional[bool] = None,
):
    """Build the shared filtered query over ``stage='shadow'`` rows.

    ONE definition, used by both endpoints, so ``/decisions`` and ``/performance``
    can never disagree about which rows a given filter selects.
    """
    query = db.query(Trade).filter(Trade.stage == STAGE_SHADOW)
    # A blank ``?instrument=`` is how a client spells "no filter"; only a NON-empty
    # unknown symbol is a typo worth a 404.
    instrument = instrument.strip() if instrument else None
    if instrument:
        inst = db.query(Instrument).filter_by(symbol=instrument).first()
        if inst is None:
            raise HTTPException(status_code=404, detail=f"instrument '{instrument}' not found")
        query = query.filter(Trade.instrument_id == inst.id)
    if ml_decision == DECISION_UNSCORED:
        query = query.filter(Trade.ml_decision.is_(None))
    elif ml_decision is not None:
        query = query.filter(Trade.ml_decision == ml_decision)
    if status == "resolved":
        query = query.filter(Trade.closed_at.isnot(None))
    elif status == "pending":
        query = query.filter(Trade.closed_at.is_(None))
    if ambiguous is not None:
        # The column is NOT NULL with a False default, but treat NULL as "not
        # ambiguous" so a legacy row written before the column existed still filters
        # into the trustworthy stream rather than vanishing from both sides.
        if ambiguous:
            query = query.filter(Trade.ambiguous_resolution.is_(True))
        else:
            query = query.filter(
                (Trade.ambiguous_resolution.is_(False))
                | (Trade.ambiguous_resolution.is_(None))
            )
    if start is not None:
        query = query.filter(Trade.opened_at >= _naive(start))
    if end is not None:
        query = query.filter(Trade.opened_at <= _naive(end))
    return query


def _cohort(rows: Sequence[Trade]) -> CohortStats:
    """Metrics for one cohort, computed on its RESOLVED subset only.

    An unresolved row has no ``rr_actual``; including it would either require
    imputing an outcome (never) or would silently deflate every mean. So it is
    counted as ``pending`` and excluded from the sample.
    """
    resolved = [r for r in rows if _is_resolved(r)]
    records = [
        {"rr_actual": r.rr_actual, "holding_hours": _holding_hours(r)} for r in resolved
    ]
    return CohortStats(
        total=len(rows),
        count=len(resolved),
        pending=len(rows) - len(resolved),
        win_rate=_finite(M.win_rate(records)),
        expectancy=_finite(M.expectancy(records)),
        profit_factor=_finite(M.profit_factor(records)),
        avg_holding_hours=_finite(M.avg_holding_hours(records)),
        outcome_breakdown=M.outcome_breakdown(records),
    )


def _filter_effect(cohorts: dict[str, CohortStats]) -> FilterEffect:
    """Derive the lift metrics from the cohort stats — never from a second pass over
    the rows, so the headline comparison and the per-cohort table cannot disagree.

    ``.count`` (resolved rows) is used throughout, including for ``keep_rate``, so
    every number here describes the same sample.
    """
    take, skip, everything = cohorts[COHORT_TAKE], cohorts[COHORT_SKIP], cohorts[COHORT_ALL]
    return FilterEffect(
        keep_rate=(take.count / everything.count) if everything.count else None,
        expectancy_lift_vs_all=_delta(take.expectancy, everything.expectancy),
        expectancy_lift_vs_skip=_delta(take.expectancy, skip.expectancy),
        profit_factor_lift_vs_all=_delta(take.profit_factor, everything.profit_factor),
    )


def _to_decision(trade: Trade, symbols: dict[int, str]) -> ShadowDecision:
    shadow_meta = (trade.signal_reasoning or {}).get(REASONING_SHADOW_KEY) or {}
    return ShadowDecision(
        id=trade.id,
        instrument=symbols.get(trade.instrument_id, str(trade.instrument_id)),
        granularity=shadow_meta.get("granularity"),
        direction=trade.direction,
        entry_price=trade.entry_price,
        stop_price=trade.stop_price,
        tp_price=trade.tp_price,
        exit_price=trade.exit_price,
        rr_entry=trade.rr_entry,
        rr_actual=_finite(trade.rr_actual),
        outcome=trade.outcome,
        exit_reason=trade.exit_reason,
        ambiguous_resolution=bool(trade.ambiguous_resolution),
        resolved=_is_resolved(trade),
        holding_hours=_holding_hours(trade),
        ml_probability=_finite(trade.ml_probability),
        ml_decision=trade.ml_decision,
        ml_model_id=trade.ml_model_id,
        confluence_score=trade.confluence_score,
        session=trade.session,
        risk_passed=shadow_meta.get("risk_passed"),
        risk_rejection_reason=shadow_meta.get("risk_rejection_reason"),
        opened_at=trade.opened_at,
        closed_at=trade.closed_at,
    )


def _symbol_map(db: Session, instrument_ids: Sequence[int]) -> dict[int, str]:
    """One query for every symbol on the page (never one per row)."""
    ids = set(instrument_ids)
    if not ids:
        return {}
    rows = db.query(Instrument.id, Instrument.symbol).filter(Instrument.id.in_(ids)).all()
    return {row.id: row.symbol for row in rows}


def _is_ambiguous(trade: Trade) -> bool:
    """Whether the resolver flagged this row's outcome as untrustworthy.

    NULL is treated as False for the same reason the query filter does: a legacy row
    predating the column is "not known to be ambiguous", not "ambiguous".
    """
    return trade.ambiguous_resolution is True


def _is_resolved(trade: Trade) -> bool:
    """Resolution is defined by ``closed_at`` — the exact column the resolver's
    idempotency selector keys on, so the API and the resolver agree by construction."""
    return trade.closed_at is not None


def _holding_hours(trade: Trade) -> float | None:
    if trade.closed_at is None or trade.opened_at is None:
        return None
    return (trade.closed_at - trade.opened_at).total_seconds() / _SECONDS_PER_HOUR


def _finite(value: Any) -> float | None:
    """NaN / ±inf / None → ``None``.

    ``metrics`` returns NaN for an undefined sample and ``inf`` for a profit factor
    with zero losses. Both are honest in Python and meaningless in JSON — emitting
    ``null`` says "not measurable", where any numeric substitute would read as a
    measurement that was never taken.
    """
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _delta(a: float | None, b: float | None) -> float | None:
    return None if (a is None or b is None) else a - b


def _naive(dt: datetime) -> datetime:
    """Query bounds arrive from HTTP possibly tz-aware; stored timestamps are naive UTC."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
