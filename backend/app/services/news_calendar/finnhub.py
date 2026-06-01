"""
Finnhub calendar provider — backup to ForexFactory.

Stub. Constructor reads settings; get_events raises NotImplementedError until
M8 actually needs it.
"""
from datetime import datetime

from app.config import Settings
from app.services.news_calendar.base import NewsCalendarProvider, NewsEvent


class FinnhubCalendar(NewsCalendarProvider):
    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.NEWS_SENTIMENT_API_KEY  # reuse if user has Finnhub key
        self._base_url = "https://finnhub.io/api/v1"

    def get_events(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
    ) -> list[NewsEvent]:
        raise NotImplementedError("Finnhub calendar provider not yet implemented")
