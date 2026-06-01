"""
Trade classifier — auto-classifies trades into:
  STRATEGY | NEWS | MANIPULATION | MANUAL | UNCERTAIN

Returns (classification, confidence, evidence_dict).

Confidence floor `MIN_ML_TRAINING_CONFIDENCE` (settings) determines whether
the trade is eligible for ML training. Below the floor, classification is
forced to UNCERTAIN regardless of the underlying type.

This service is wired but not exercised. M8 (order execution) will call it
at trade open. For now it exists so future work can import the contract.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.config import Settings
from app.services.news_calendar.factory import get_news_calendar
from app.services.news_calendar.base import NewsEvent

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    classification: str            # STRATEGY | NEWS | MANIPULATION | MANUAL | UNCERTAIN
    confidence: float              # 0.0–1.0
    evidence: dict                 # raw signals used to make the call
    news_events: list[NewsEvent]   # events found in window (may be empty)


def classify_trade(
    instrument: str,
    entry_time: datetime,
    spread_at_entry: float | None,
    instrument_typical_spread: float | None,
    settings: Settings,
    manual_close: bool = False,
) -> ClassificationResult:
    """
    Decide the classification of a closed (or just-opened) trade.

    Order of precedence:
      1. MANUAL — user closed before stop/target hit (passed in by caller)
      2. NEWS   — high-impact event within ±NEWS_CALENDAR_WINDOW_HOURS
      3. MANIPULATION — spread > SPREAD_ANOMALY_MULTIPLIER × typical
      4. STRATEGY — none of the above
      5. UNCERTAIN — confidence below MIN_ML_TRAINING_CONFIDENCE
    """
    evidence: dict = {
        "manual_close": manual_close,
        "spread_at_entry": spread_at_entry,
        "instrument_typical_spread": instrument_typical_spread,
    }

    if manual_close:
        return ClassificationResult(
            classification="MANUAL",
            confidence=1.0,
            evidence=evidence,
            news_events=[],
        )

    news_events, news_ok = _check_news_window(instrument, entry_time, settings)
    evidence["news_call_ok"] = news_ok
    has_high_impact = any(e.impact == "high" for e in news_events)
    evidence["high_impact_count"] = sum(1 for e in news_events if e.impact == "high")

    spread_ok = spread_at_entry is not None and instrument_typical_spread is not None
    spread_anomaly = False
    if spread_ok and instrument_typical_spread:
        ratio = spread_at_entry / instrument_typical_spread
        evidence["spread_ratio"] = ratio
        spread_anomaly = ratio > settings.SPREAD_ANOMALY_MULTIPLIER

    confidence = _compute_confidence(news_ok=news_ok, spread_ok=spread_ok)
    evidence["confidence"] = confidence

    if confidence < settings.MIN_ML_TRAINING_CONFIDENCE:
        return ClassificationResult(
            classification="UNCERTAIN",
            confidence=confidence,
            evidence=evidence,
            news_events=news_events,
        )

    if has_high_impact:
        classification = "NEWS"
    elif spread_anomaly:
        classification = "MANIPULATION"
    else:
        classification = "STRATEGY"

    return ClassificationResult(
        classification=classification,
        confidence=confidence,
        evidence=evidence,
        news_events=news_events,
    )


def _check_news_window(
    instrument: str,
    entry_time: datetime,
    settings: Settings,
) -> tuple[list[NewsEvent], bool]:
    window = timedelta(hours=settings.NEWS_CALENDAR_WINDOW_HOURS)
    start = entry_time - window
    end = entry_time + window
    try:
        calendar = get_news_calendar(settings)
        events = calendar.get_events(instrument, start, end)
        return events, True
    except Exception as exc:
        logger.warning("news calendar lookup failed: %s", exc)
        return [], False


def _compute_confidence(news_ok: bool, spread_ok: bool) -> float:
    # Both signals available → high confidence (1.0).
    # One missing → degraded (0.6).
    # Both missing → low (0.3).
    if news_ok and spread_ok:
        return 1.0
    if news_ok or spread_ok:
        return 0.6
    return 0.3
