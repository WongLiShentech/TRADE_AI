from app.config import Settings
from app.services.alerts.base import AlertDelivery


class EmailAlert(AlertDelivery):
    def __init__(self, settings: Settings) -> None:
        pass

    def send(self, subject: str, body: str) -> None:
        raise NotImplementedError("EmailAlert not yet implemented")
