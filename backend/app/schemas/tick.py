from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TickRead(BaseModel):
    instrument: str
    bid: float
    ask: float
    mid: float
    spread_pips: Optional[float]
    timestamp: datetime
