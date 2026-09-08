"""M7 backtester API.

Endpoints (all under ``/api/v1/backtester``):
* ``POST /run``            — run the walk-forward backtest synchronously (optional
                             ``instruments`` subset), return the run summary.
* ``GET  /runs``           — list runs, newest first.
* ``GET  /runs/{id}``      — one run's detail (incl. per-fold breakdown).
* ``GET  /runs/{id}/trades`` — paginated Trade rows persisted by that run's window.

The runner walks the configured EXECUTION_WINDOW (env-driven), so the request body
does not carry dates — it only optionally narrows the instrument universe.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models.backtest_run import BacktestRun
from app.models.trade import Trade
from app.services.backtester.runner import run_backtest

router = APIRouter()


# ── response schemas ─────────────────────────────────────────────────────────
class RunSummary(BaseModel):
    id: int
    strategy: str
    passed: bool
    in_sample_start: datetime
    in_sample_end: datetime
    oos_start: datetime
    oos_end: datetime
    trade_count: int
    oos_sample_size: int | None
    win_rate: float
    avg_rr: float
    expectancy: float
    max_drawdown: float
    sharpe: float
    profit_factor: float | None
    deflated_sharpe: float | None
    probabilistic_sharpe: float | None
    avg_holding_hours: float | None
    trades_full_win: int | None
    trades_partial: int | None
    trades_breakeven: int | None
    trades_loss: int | None
    run_at: datetime

    class Config:
        from_attributes = True


class RunDetail(RunSummary):
    fold_breakdown: dict | None


class RunResult(BaseModel):
    run_id: int
    passed: bool
    instruments: list[str]
    decision_bars_evaluated: int
    signals_fired: int
    trades_simulated: int
    rejections: dict[str, int]
    runtime_seconds: float
    detail: RunDetail


class TradeRow(BaseModel):
    id: int
    instrument_id: int
    direction: str
    entry_price: float
    exit_price: float | None
    stop_price: float
    tp_price: float
    rr_entry: float
    rr_actual: float | None
    outcome: str | None
    exit_reason: str | None
    ambiguous_resolution: bool
    session: str | None
    opened_at: datetime
    closed_at: datetime | None
    signal_reasoning: dict | None

    class Config:
        from_attributes = True


class RunRequest(BaseModel):
    # REQUIRED. A run with no strategy writes rows with a NULL strategy_id, and
    # `load_dataset` treats NULL as its own strategy — so one API-triggered run
    # permanently trips the mixed-corpus guard and blocks training until somebody
    # attributes those rows by hand. Making the caller name the strategy is the whole
    # cost of never having that happen.
    strategy_id: int
    instruments: list[str] | None = None


# ── endpoints ────────────────────────────────────────────────────────────────
@router.post("/run", response_model=RunResult, status_code=201)
def run(
    request: RunRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    result = run_backtest(
        db, settings, request.instruments, strategy_id=request.strategy_id
    )
    run_row = db.query(BacktestRun).get(result.run_id)
    return RunResult(
        run_id=result.run_id,
        passed=result.passed,
        instruments=result.instruments,
        decision_bars_evaluated=result.decision_bars_evaluated,
        signals_fired=result.signals_fired,
        trades_simulated=result.trades_simulated,
        rejections=result.rejections,
        runtime_seconds=result.runtime_seconds,
        detail=RunDetail.model_validate(run_row),
    )


@router.get("/runs", response_model=list[RunSummary])
def list_runs(db: Session = Depends(get_db)):
    return db.query(BacktestRun).order_by(BacktestRun.run_at.desc()).all()


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run_row = db.query(BacktestRun).get(run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail=f"backtest run {run_id} not found")
    return run_row


@router.get("/runs/{run_id}/trades", response_model=list[TradeRow])
def get_run_trades(
    run_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Trade rows whose signal falls inside this run's window (stage='backtest'),
    ordered by signal time. The schema stores no per-run FK on Trade, so rows are
    scoped by the run's [in_sample_start, oos_end] window + backtest stage."""
    run_row = db.query(BacktestRun).get(run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail=f"backtest run {run_id} not found")
    return (
        db.query(Trade)
        .filter(
            Trade.stage == "backtest",
            Trade.opened_at >= run_row.in_sample_start,
            Trade.opened_at <= run_row.oos_end,
        )
        .order_by(Trade.opened_at.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )
