from app.config import Settings
from app.services.alerts.base import AlertDelivery


def get_alert_delivery(settings: Settings) -> AlertDelivery:
    name = (settings.ALERT_DELIVERY or "").lower()
    if name == "log":
        from app.services.alerts.log_alert import LogAlert
        return LogAlert(settings)
    elif name == "email":
        from app.services.alerts.email_alert import EmailAlert
        return EmailAlert(settings)
    elif name == "telegram":
        from app.services.alerts.telegram_alert import TelegramAlert
        return TelegramAlert(settings)
    else:
        raise ValueError(f"Unknown ALERT_DELIVERY: {settings.ALERT_DELIVERY!r}")
