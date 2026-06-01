from datetime import datetime

from pydantic import BaseModel


class CandleRead(BaseModel):
    id: int
    instrument_id: int
    granularity: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    model_config = {"from_attributes": True}


class CandleFetchResponse(BaseModel):
    instrument: str
    granularity: str
    stored: int   # new candles added this call
    total: int    # total candles in DB for this instrument + granularity
