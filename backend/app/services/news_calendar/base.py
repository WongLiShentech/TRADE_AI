from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class NewsEvent:
    """Single economic-calendar event relevant to a currency or instrument."""
    timestamp: datetime
    currency: str           # e.g. "USD", "EUR"
    title: str              # e.g. "Non-Farm Payrolls"
    impact: str             # "low" | "medium" | "high"
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    source: str = ""


class NewsCalendarProvider(ABC):
    """
    Provider-agnostic interface for economic news calendars.
    Implementations: ForexFactory (default), Finnhub (backup stub).
    """

    @abstractmethod
    def get_events(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
    ) -> list[NewsEvent]:
        """
        Return events relevant to the instrument in the given window.
        Implementations must derive the relevant currencies from the
        instrument symbol (e.g. EUR_USD → ["EUR", "USD"]).
        On any failure (network, parse, deprecation) implementations
        SHOULD log a warning and return [] rather than raise — the
        platform treats absence-of-news as "could not verify" and
        downgrades classification confidence accordingly.
        """
        raise NotImplementedError
