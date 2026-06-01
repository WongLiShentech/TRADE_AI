from datetime import datetime

from app.config import Settings
from app.services.fundamental.base import FundamentalDataProvider, FundamentalSnapshot


class FredFundamental(FundamentalDataProvider):
    """FRED (Federal Reserve Economic Data) — free, US-centric macro series."""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.FUNDAMENTAL_DATA_API_KEY
        self._base_url = "https://api.stlouisfed.org/fred"

    def get_snapshot(
        self,
        instrument: str,
        at: datetime,
    ) -> list[FundamentalSnapshot]:
        raise NotImplementedError("FRED fundamental provider not yet implemented")
