from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SentimentScore:
    """Aggregate sentiment score over the lookback window."""
    instrument: str
    timestamp: datetime
    score: float            # -1.0 (very bearish) to 1.0 (very bullish)
    volume: int             # number of headlines aggregated
    source: str = ""


class NewsSentimentProvider(ABC):
    """
    Phase 1B scaffold. Real implementations will pull recent news headlines,
    score them with NLP, and return an aggregated score that ML can consume
    as a feature.
    """

    @abstractmethod
    def get_recent_sentiment(
        self,
        instrument: str,
        lookback_minutes: int,
    ) -> SentimentScore:
        raise NotImplementedError
