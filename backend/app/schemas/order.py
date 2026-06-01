from datetime import datetime

from pydantic import BaseModel


class OrderCreate(BaseModel):
    instrument_id: int
    direction: str
    order_type: str
    units: int
    price: float | None = None
    stop_loss: float
    take_profit: float


class OrderRead(BaseModel):
    id: int
    instrument_id: int
    signal_id: int | None
    direction: str
    order_type: str
    units: int
    price: float | None
    stop_loss: float
    take_profit: float
    status: str
    broker_order_id: str | None
    created_at: datetime
    filled_at: datetime | None

    model_config = {"from_attributes": True}
