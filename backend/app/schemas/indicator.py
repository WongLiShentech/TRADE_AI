from datetime import datetime

from pydantic import BaseModel


class IndicatorRead(BaseModel):
    id: int
    instrument_id: int
    granularity: str
    timestamp: datetime
    atr14: float | None
    rsi14: float | None
    swing_high: float | None
    swing_low: float | None
    donchian_high: float | None
    donchian_low: float | None

    model_config = {"from_attributes": True}


class IndicatorComputeResponse(BaseModel):
    instrument: str
    granularity: str
    computed: int   # new indicator rows stored this call
    total: int      # total indicator rows for this instrument + granularity
