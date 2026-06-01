from app.config import Settings
from app.services.news_calendar.base import NewsCalendarProvider


def get_news_calendar(settings: Settings) -> NewsCalendarProvider:
    name = (settings.NEWS_CALENDAR_PROVIDER or "").lower()
    if name == "forexfactory":
        from app.services.news_calendar.forexfactory import ForexFactoryCalendar
        return ForexFactoryCalendar(settings)
    elif name == "finnhub":
        from app.services.news_calendar.finnhub import FinnhubCalendar
        return FinnhubCalendar(settings)
    else:
        raise ValueError(
            f"Unknown NEWS_CALENDAR_PROVIDER: {settings.NEWS_CALENDAR_PROVIDER!r}"
        )
