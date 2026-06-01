from functools import lru_cache

from sqlalchemy.orm import Session

from app.brokers.base import BrokerClient
from app.brokers.factory import get_broker_client
from app.config import Settings, get_settings


class BrokerRouter:
    """
    Single mandatory access layer for all broker operations.

    Routes to the correct BrokerClient by looking up the instrument's
    asset_class from the Instrument DB table, then mapping that to the
    broker configured for that asset class via BROKER_FOREX / BROKER_STOCKS /
    BROKER_CRYPTO env vars.

    Application code calls BrokerRouter — never a BrokerClient directly.

    Bootstrap note: for_instrument() requires the Instrument table to be
    populated (after M1 instrument discovery). On first run, use
    for_asset_class() or all_clients() directly to seed the DB.
    """

    def __init__(self, settings: Settings) -> None:
        # Build asset_class → BrokerClient map from configured env vars only.
        # Keys are lowercase asset class names matching Instrument.asset_class in DB.
        # Only asset classes with a configured broker are registered.
        self._clients: dict[str, BrokerClient] = {}
        if settings.BROKER_FOREX:
            self._clients["forex"] = get_broker_client(settings.BROKER_FOREX, settings)
        if settings.BROKER_STOCKS:
            self._clients["stocks"] = get_broker_client(settings.BROKER_STOCKS, settings)
        if settings.BROKER_CRYPTO:
            self._clients["crypto"] = get_broker_client(settings.BROKER_CRYPTO, settings)

    def for_instrument(self, instrument: str, db: Session) -> BrokerClient:
        """
        Look up instrument's asset_class from DB and return the correct client.
        Raises ValueError if instrument not found or asset class has no broker configured.
        """
        from app.models.instrument import Instrument

        inst = db.query(Instrument).filter_by(symbol=instrument).first()
        if inst is None:
            raise ValueError(f"Instrument not found: {instrument}")
        return self._for_asset_class(inst.asset_class)

    def for_asset_class(self, asset_class: str) -> BrokerClient:
        """
        Direct routing by asset class string.
        Used for instrument discovery before the DB is populated.
        """
        return self._for_asset_class(asset_class)

    def all_clients(self) -> dict[str, BrokerClient]:
        """
        Returns all configured broker clients keyed by asset class.
        Used for multi-broker instrument discovery in Phase 2.
        """
        return dict(self._clients)

    def _for_asset_class(self, asset_class: str) -> BrokerClient:
        client = self._clients.get(asset_class.lower())
        if client is None:
            raise ValueError(f"No broker configured for asset class: {asset_class}")
        return client


@lru_cache(maxsize=1)
def get_broker_router() -> "BrokerRouter":
    return BrokerRouter(get_settings())
