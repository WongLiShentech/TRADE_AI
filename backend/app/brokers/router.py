from functools import lru_cache

from sqlalchemy.orm import Session

from app.brokers.base import (
    BrokerClient,
    OrderPlacementDisabledError,
    OrderRequest,
    OrderResult,
)
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
        self._settings = settings
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

    def place_order(
        self, instrument: str, order: OrderRequest, db: Session
    ) -> OrderResult:
        """Route an order to the right broker — the guarded application entry point.

        BrokerRouter is the mandatory access layer, so this is where application code
        should place orders; it checks ``ORDER_PLACEMENT_ENABLED`` before it even
        resolves a client, so an observe-only platform never so much as looks up which
        broker would have received the order.

        The check is deliberately DUPLICATED here and in ``BrokerClient.place_order``.
        They guard different things: this one guards the routing layer application
        code is supposed to use, the base-class one guards the transport itself, so a
        caller holding a client reference obtained via ``for_instrument`` is equally
        blocked. Defense in depth on a flag whose failure mode is real money.

        Args:
            instrument: symbol to trade — resolved to an asset class via the DB, never
                hardcoded.
            order: the fully-specified order; units already derived by ``RiskEngine``.
            db: SQLAlchemy session used for the instrument → asset-class lookup.

        Returns:
            The broker's :class:`OrderResult`.

        Raises:
            OrderPlacementDisabledError: the platform is in observe-only mode.
            ValueError: unknown instrument, or no broker configured for its asset class.
        """
        self._assert_order_placement_enabled(order)
        return self.for_instrument(instrument, db).place_order(order)

    def order_placement_enabled(self) -> bool:
        """Whether the platform is currently permitted to send orders to a broker."""
        return self._settings.ORDER_PLACEMENT_ENABLED is True

    def _assert_order_placement_enabled(self, order: OrderRequest) -> None:
        if not self.order_placement_enabled():
            raise OrderPlacementDisabledError(
                "ORDER_PLACEMENT_ENABLED is "
                f"{self._settings.ORDER_PLACEMENT_ENABLED!r} — BrokerRouter refuses to "
                "route an order while the platform is in observe-only mode. "
                f"Rejected: {order.direction} {order.units} {order.instrument}."
            )

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
