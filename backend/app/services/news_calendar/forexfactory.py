"""
ForexFactory calendar provider.

ForexFactory publishes a public weekly JSON feed at:
  https://nfs.faireconomy.media/ff_calendar_thisweek.json
  https://nfs.faireconomy.media/ff_calendar_nextweek.json

The feed is community-maintained and may go offline without notice.
On any failure this provider logs a warning and returns [] — the platform
treats absence-of-news as "could not verify" and downgrades the trade's
classification_confidence to LOW (UNCERTAIN).

NOTE: This module is wired but not exercised. The first real call happens
in M8 (order execution). For now it exists so the trade_classifier scaffold
can import it cleanly.
"""
import logging
from datetime import datetime

import httpx

from app.config import Settings
from app.services.news_calendar.base import NewsCalendarProvider, NewsEvent

logger = logging.getLogger(__name__)

_THIS_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_NEXT_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"


class ForexFactoryCalendar(NewsCalendarProvider):
    def __init__(self, settings: Settings) -> None:
        self._timeout = 10.0

    def get_events(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
    ) -> list[NewsEvent]:
        currencies = self._currencies_from_instrument(instrument)
        events: list[NewsEvent] = []
        for url in (_THIS_WEEK_URL, _NEXT_WEEK_URL):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.get(url)
                    response.raise_for_status()
                events.extend(self._parse(response.json(), currencies, start, end))
            except Exception as exc:
                logger.warning("forexfactory fetch failed (%s): %s", url, exc)
        return events

    @staticmethod
    def _currencies_from_instrument(instrument: str) -> set[str]:
        # Forex symbols arrive as "EUR_USD". Non-forex instruments (CFDs, metals)
        # have other shapes — caller decides whether to call this provider for them.
        parts = instrument.replace("/", "_").split("_")
        return {p.upper() for p in parts if len(p) == 3}

    @staticmethod
    def _parse(
        raw: list[dict],
        currencies: set[str],
        start: datetime,
        end: datetime,
    ) -> list[NewsEvent]:
        out: list[NewsEvent] = []
        for item in raw:
            try:
                ts_str = item.get("date")
                if not ts_str:
                    continue
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts < start or ts > end:
                    continue
                ccy = (item.get("country") or "").upper()
                if currencies and ccy not in currencies:
                    continue
                out.append(
                    NewsEvent(
                        timestamp=ts,
                        currency=ccy,
                        title=item.get("title", ""),
                        impact=(item.get("impact") or "").lower(),
                        actual=_to_float(item.get("actual")),
                        forecast=_to_float(item.get("forecast")),
                        previous=_to_float(item.get("previous")),
                        source="forexfactory",
                    )
                )
            except Exception as exc:
                logger.debug("forexfactory parse skip: %s", exc)
        return out


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace("%", "").replace(",", ""))
    except (ValueError, TypeError):
        return None
