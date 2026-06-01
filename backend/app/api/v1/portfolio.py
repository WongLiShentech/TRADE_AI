from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db

router = APIRouter()


class PortfolioSummary(BaseModel):
    balance: float
    unrealised_pnl: float
    open_positions: int
    equity_curve: list[dict]


@router.get("", response_model=PortfolioSummary)
def get_portfolio(db: Session = Depends(get_db)):
    raise NotImplementedError
