from app.config import Settings
from app.services.fundamental.base import FundamentalDataProvider


def get_fundamental_data(settings: Settings) -> FundamentalDataProvider:
    name = (settings.FUNDAMENTAL_DATA_PROVIDER or "").lower()
    if name == "finnhub":
        from app.services.fundamental.finnhub import FinnhubFundamental
        return FinnhubFundamental(settings)
    elif name == "fred":
        from app.services.fundamental.fred import FredFundamental
        return FredFundamental(settings)
    else:
        raise ValueError(
            f"Unknown FUNDAMENTAL_DATA_PROVIDER: {settings.FUNDAMENTAL_DATA_PROVIDER!r}"
        )
