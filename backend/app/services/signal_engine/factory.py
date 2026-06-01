"""
SignalEngine factory. Dispatches on SIGNAL_SOURCE env var.

Phase 1: only "rule_based" is implemented. Phase 1B will add "ml".
"""
from app.config import Settings
from app.services.signal_engine.base import SignalEngine
from app.services.signal_engine.rule_based import RuleBasedSignalEngine


def get_signal_engine(settings: Settings) -> SignalEngine:
    source = settings.SIGNAL_SOURCE.lower()
    if source == "rule_based":
        return RuleBasedSignalEngine()
    raise ValueError(f"Unknown SIGNAL_SOURCE: {settings.SIGNAL_SOURCE}")
