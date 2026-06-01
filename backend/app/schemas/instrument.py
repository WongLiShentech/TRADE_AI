from pydantic import BaseModel


class InstrumentBase(BaseModel):
    symbol: str
    display_name: str
    pip_size: float
    pip_location: int
    asset_class: str
    broker_id: str
    is_active: bool


class InstrumentRead(InstrumentBase):
    id: int

    model_config = {"from_attributes": True}


class InstrumentListResponse(BaseModel):
    total: int
    by_asset_class: dict[str, int]
    instruments: list[InstrumentRead]
