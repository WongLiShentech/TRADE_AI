from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class SignalRead(BaseModel):
    id: int
    instrument_id: int
    granularity: str
    direction: str
    entry: float
    stop: float
    target: float
    confidence_score: int
    score_breakdown: dict
    status: str
    rejection_reason: Optional[str] = None
    created_at: datetime
    expires_at: datetime

    model_config = {"from_attributes": True}


class EvaluateResponse(BaseModel):
    instrument: str
    granularity: str
    signal_fired: bool
    signal: Optional[SignalRead] = None
    reason: str
    units: Optional[int] = None
    risk_amount: Optional[float] = None
