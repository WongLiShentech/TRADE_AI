from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.trade import TradeRead

router = APIRouter()


@router.get("", response_model=list[TradeRead])
def list_trades(
    stage: str | None = Query(None),
    instrument: str | None = Query(None),
    signal_source: str | None = Query(None),
    db: Session = Depends(get_db),
):
    raise NotImplementedError
