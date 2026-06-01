from abc import ABC, abstractmethod


class AlertDelivery(ABC):
    """Provider-agnostic alert delivery channel."""

    @abstractmethod
    def send(self, subject: str, body: str) -> None:
        raise NotImplementedError
