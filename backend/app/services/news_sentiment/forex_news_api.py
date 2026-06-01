from app.config import Settings
from app.services.news_sentiment.base import NewsSentimentProvider, SentimentScore


class ForexNewsApiSentiment(NewsSentimentProvider):
    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.NEWS_SENTIMENT_API_KEY

    def get_recent_sentiment(
        self,
        instrument: str,
        lookback_minutes: int,
    ) -> SentimentScore:
        raise NotImplementedError("forex_news_api sentiment provider not yet implemented")
