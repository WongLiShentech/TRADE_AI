"""
SignalEngine ABC + output dataclass.

A SignalEngine evaluates a single instrument at the close of a candle and
returns SignalOutput on a valid signal or None. All gating decisions
(cooldown, session, spread, confluence threshold) live inside the engine —
the pipeline simply asks "is there a signal here?" and persists the result.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.config import Settings


@dataclass
class SignalOutput:
    instrument: str
    granularity: str
    direction: str            # BUY | SELL
    entry: float
    stop: float
    target: float
    confidence_score: int
    score_breakdown: dict     # {"trend": True, "rsi": True, "structure": False, "session": True, "spread": True}


class SignalEngine(ABC):
    @abstractmethod
    def evaluate(
        self,
        instrument: str,
        granularity: str,
        db: Session,
        settings: Settings,
        *,
        as_of: Optional[datetime] = None,
        quote: Optional[Any] = None,
    ) -> Optional[SignalOutput]:
        """Evaluate a single instrument at the close of a candle.

        Returns SignalOutput if all gating checks pass and the confluence
        threshold is met, otherwise None.

        Live (default): ``as_of``/``quote`` are None — the engine reads the latest
        stored rows and the live tick cache. Backtest: the M7 runner passes
        ``as_of=T`` (strictly after the decision bar's close) to bound every read
        ``timestamp <= T`` and ``quote`` (a decision-bar Bid/Ask close) in place of
        the tick cache. Implementations must keep the ``as_of=None`` path unchanged.
        """
        raise NotImplementedError
