from app.config import Settings
from app.brokers.base import BrokerClient


def get_broker_client(broker_name: str, settings: Settings) -> BrokerClient:
    name = broker_name.lower()

    if name == "oanda":
        from app.brokers.oanda import OandaClient
        return OandaClient(settings)
    elif name == "alpaca":
        from app.brokers.alpaca import AlpacaClient
        return AlpacaClient(settings)
    elif name == "binance":
        from app.brokers.binance import BinanceClient
        return BinanceClient(settings)
    else:
        raise ValueError(f"Unknown broker: {broker_name}")
