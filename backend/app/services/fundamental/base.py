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


@dataclass
class SeriesObservation:
    """
    One observation of a raw series (series-centric, not instrument-centric).
    `release_time` is computed by the caller as ref_period + a publication lag
    (see the macro_series registry) — kept out of the provider so the provider
    stays a dumb fetcher.
    """
    ref_period: str   # the period the value describes, ISO date e.g. "2025-04-01"
    value: float


@dataclass
class SeriesVintage:
    """
    One VINTAGE of a REVISED series' observation. Unlike SeriesObservation,
    `release_time` here is the TRUE date THIS vintage became public (from FRED
    `realtime_start`), NOT a synthetic ref_period + lag. Under full-bitemporal
    ingestion a single ref_period yields MULTIPLE SeriesVintage rows — the earliest
    `release_time` is the first release, each later one a revision — and `value` is
    the value that held over that vintage's realtime window. Feeds the bitemporal
    macro_data table directly (one DB row per vintage).
    """
    ref_period: str        # period the value describes, ISO date e.g. "2025-01-01"
    release_time: datetime # TRUE date this vintage became public (realtime_start)
    value: float           # value as of this vintage (first print or a revision)


class FundamentalDataProvider(ABC):
    """
    Real implementations pull macro data (rates, yields, VIX, CPI) for ML
    features. Two access shapes:
      - get_snapshot(instrument, at)  → per-instrument live snapshot (legacy)
      - get_series_vintages(series_id) → series-centric, point-in-time vintages
        (feeds the bitemporal macro_data table; per-pair math stays in feature_builder)
    """

    @abstractmethod
    def get_snapshot(
        self,
        instrument: str,
        at: datetime,
    ) -> list[FundamentalSnapshot]:
        raise NotImplementedError

    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
    ) -> list[SeriesObservation]:
        """Raw (ref_period, value) observations for a series. Override where supported."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_series"
        )

    def get_series_first_release(
        self,
        series_id: str,
        observation_start: str | None = None,
    ) -> list[SeriesVintage]:
        """
        First-release vintages (ref_period, TRUE release_time, first value) for a
        REVISED series. release_time is the genuine first-publication date, so the
        result is leakage-safe with no synthetic lag. Override where supported.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_series_first_release"
        )

    def get_series_all_vintages(
        self,
        series_id: str,
        observation_start: str | None = None,
    ) -> list[SeriesVintage]:
        """
        FULL BITEMPORAL vintages for a REVISED series: EVERY vintage of every
        observation, each with the TRUE date it became public (release_time). One
        ref_period yields multiple SeriesVintage rows — the earliest release_time is
        the first release, each later one a revision. Leakage-safe (no synthetic
        lag): a point-in-time lookup gated on release_time sees the first-release
        value before a revision and the revised value after. Override where supported.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_series_all_vintages"
        )

    def get_release_dates(
        self,
        release_id: int,
        start: str | None = None,
        end: str | None = None,
    ) -> list[str]:
        """ISO release dates (past + future scheduled) for a data release. Override where supported."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_release_dates"
        )
