from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db

router = APIRouter()


class BacktestRequest(BaseModel):
    instrument: str
    granularity: str
    strategy: str
    in_sample_start: datetime
    in_sample_end: datetime
    oos_start: datetime
    oos_end: datetime


class BacktestResult(BaseModel):
    id: int
    instrument_id: int
    strategy: str
    trade_count: int
    win_rate: float
    avg_rr: float
    max_drawdown: float
    sharpe: float
    expectancy: float
    passed: bool
    run_at: datetime


@router.post("/run", response_model=BacktestResult, status_code=201)
def run_backtest(request: BacktestRequest, db: Session = Depends(get_db)):
    raise NotImplementedError


@router.get("/runs", response_model=list[BacktestResult])
def list_backtest_runs(db: Session = Depends(get_db)):
    raise NotImplementedError
