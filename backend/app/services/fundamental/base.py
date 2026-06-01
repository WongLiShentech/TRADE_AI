from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FundamentalSnapshot:
    """
    Macro / fundamental data point relevant for ML feature engineering.
    Examples: 2Y yield differential, VIX level, USD index, central bank rate.
    """
    instrument: str
    timestamp: datetime
    metric: str             # e.g. "yield_diff_2y", "vix_level", "rate_diff"
    value: float
    source: str = ""


class FundamentalDataProvider(ABC):
    """
    Phase 2 scaffold. Real implementations pull macro data (rates, yields,
    VIX, USD index) for a given instrument and timestamp. ML pipeline (1B+)
    consumes these as features alongside technicals.
    """

    @abstractmethod
    def get_snapshot(
        self,
        instrument: str,
        at: datetime,
    ) -> list[FundamentalSnapshot]:
        raise NotImplementedError
