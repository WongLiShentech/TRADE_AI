import logging

from app.config import Settings
from app.services.alerts.base import AlertDelivery

logger = logging.getLogger("alerts")


class LogAlert(AlertDelivery):
    """Phase 1 default — writes alerts to the application logger. Works offline."""

    def __init__(self, settings: Settings) -> None:
        pass

    def send(self, subject: str, body: str) -> None:
        logger.warning("ALERT | %s | %s", subject, body)
