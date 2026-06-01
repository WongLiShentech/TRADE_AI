from datetime import datetime

from pydantic import BaseModel


class TradeRead(BaseModel):
    id: int
    instrument_id: int
    direction: str
    entry_price: float
    exit_price: float | None
    stop_price: float
    tp_price: float
    units: int
    risk_amount: float
    expected_pip_loss: float
    actual_pip_loss: float | None
    slippage: float | None
    rr_entry: float
    rr_actual: float | None
    signal_source: str
    stage: str
    outcome: str | None
    exit_reason: str | None
    opened_at: datetime
    closed_at: datetime | None

    model_config = {"from_attributes": True}
