from app.config import Settings
from app.services.news_sentiment.base import NewsSentimentProvider


def get_news_sentiment(settings: Settings) -> NewsSentimentProvider:
    name = (settings.NEWS_SENTIMENT_PROVIDER or "").lower()
    if name == "finnhub":
        from app.services.news_sentiment.finnhub import FinnhubSentiment
        return FinnhubSentiment(settings)
    elif name == "forex_news_api":
        from app.services.news_sentiment.forex_news_api import ForexNewsApiSentiment
        return ForexNewsApiSentiment(settings)
    else:
        raise ValueError(
            f"Unknown NEWS_SENTIMENT_PROVIDER: {settings.NEWS_SENTIMENT_PROVIDER!r}"
        )
