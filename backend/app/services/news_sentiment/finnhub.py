from app.config import Settings
from app.services.news_sentiment.base import NewsSentimentProvider, SentimentScore


class FinnhubSentiment(NewsSentimentProvider):
    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.NEWS_SENTIMENT_API_KEY
        self._base_url = "https://finnhub.io/api/v1"

    def get_recent_sentiment(
        self,
        instrument: str,
        lookback_minutes: int,
    ) -> SentimentScore:
        raise NotImplementedError("Finnhub sentiment provider not yet implemented")
