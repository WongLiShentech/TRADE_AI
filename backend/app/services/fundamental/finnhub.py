from datetime import datetime

from app.config import Settings
from app.services.fundamental.base import FundamentalDataProvider, FundamentalSnapshot


class FinnhubFundamental(FundamentalDataProvider):
    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.FUNDAMENTAL_DATA_API_KEY
        self._base_url = "https://finnhub.io/api/v1"

    def get_snapshot(
        self,
        instrument: str,
        at: datetime,
    ) -> list[FundamentalSnapshot]:
        raise NotImplementedError("Finnhub fundamental provider not yet implemented")
