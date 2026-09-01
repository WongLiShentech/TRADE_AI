"""
Single source of truth for trading-timeframe metadata.

Replaces the scattered, hand-maintained dicts (`_PERIOD_HOURS`,
`_CANDLES_PER_DAY`, `_EXPIRY_HOURS`) and the hard-coded H4-only scheduler cron.
Adding a new timeframe (e.g. M15, W) is now a single entry in `TIMEFRAMES`.

`get_timeframe()` raises on an unknown code — deliberately fail loud rather than
silently falling back to a default (the old `.get(code, 4)` pattern could use a
4-hour step / expiry for an unknown timeframe and silently corrupt data).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Timeframe:
    code: str                 # broker-facing granularity string ("M1", "H4", "D")
    period_minutes: int       # canonical bar length; everything else derives from this
    expiry_hours: float       # how long a signal on this timeframe stays valid
    cron_hours: str | None    # APScheduler `hour=` for the scheduled job; None = no scheduled job
    cron_minute: int = 1      # minute offset so the broker finalizes the closing candle
    # True  → the scheduled job runs the FULL candle-close pipeline (fetch → indicators
    #         → signal engine → risk → shadow observation).
    # False → the scheduled job is DATA REFRESH ONLY (fetch → indicators). Used by
    #         timeframes whose candles feed other timeframes' features but which must
    #         never author a signal of their own (D1 supplies the trend filter and the
    #         D1-derived features `dist_to_sma50_atr` / `d1_close` / `d1_sma50`).
    fires_signals: bool = True
    # Which price series the scheduled job maintains for this timeframe.
    # ("M",)      → decision series only (H4/D1 — indicators are Mid-derived).
    # ("B", "A")  → execution series only (M1 — the triple-barrier simulator resolves
    #               outcomes on Bid+Ask jointly and never reads M1 Mid).
    price_types: tuple[str, ...] = ("M",)
    # False → the scheduled job never calls indicator_service. M1 carries no indicator
    # rows by design (ATR/RSI/swings are computed on the TRADING timeframe only).
    computes_indicators: bool = True
    # Name of the Settings attribute holding this timeframe's trailing-window lookback
    # in HOURS. When set, the scheduled job re-fetches a fixed [now - lookback, now]
    # window instead of resuming incrementally from the newest stored candle.
    #
    # Why a trailing window rather than the incremental resume the other timeframes
    # use: an incremental fetch resumes from the newest stored bar, so any ingestion
    # outage turns the next M1 run into an unbounded multi-month backfill inside a
    # scheduled job. A bounded trailing window has a fixed, predictable API cost per
    # run whatever the gap, and the overlap between consecutive runs makes it
    # self-healing for short outages (ON CONFLICT DO NOTHING makes the overlap free).
    # A genuine long gap is an operational backfill (scripts/ingest_bid_ask.py), not
    # something a cron job should silently attempt.
    trailing_window_setting: str | None = None

    @property
    def period_hours(self) -> float:
        return self.period_minutes / 60.0

    @property
    def candles_per_day(self) -> float:
        return 1440 / self.period_minutes


TIMEFRAMES: dict[str, Timeframe] = {
    # M1 fires no signals — it is the INTRABAR RESOLUTION series. The triple-barrier
    # simulator walks M1 Bid+Ask to decide whether a trade hit its stop or its target
    # and in what order, so M8-Shadow Phase 3 cannot honestly resolve a live shadow
    # row without live M1 (GAP-13). Without it every outcome would silently fall back
    # to the degraded signal-timeframe path and be flagged ambiguous_resolution=True.
    #
    # Schedule — hourly at :20 UTC, trailing-window top-up (M1_LIVE_LOOKBACK_HOURS):
    #   * Hourly (not streaming): M1 is high volume, and the resolver only needs the
    #     window to be COMPLETE by T + hold-horizon (~44h on H4), never in real time.
    #     Hourly with a multi-hour lookback gives large run-to-run overlap, so a few
    #     missed runs self-heal without any backfill.
    #   * Minute 20 keeps it clear of the H4 jobs (:01), the D1 refresh (23:05), the
    #     intraday fundamentals poll (:15) and the shadow resolver (:35).
    #   * price_types=("B","A"): the simulator reads Bid+Ask jointly and never M1 Mid,
    #     so fetching Mid would double the API cost for data nothing consumes.
    #   * computes_indicators=False: indicators are Mid-derived and live on the
    #     trading timeframe only.
    "M1": Timeframe(
        code="M1", period_minutes=1, expiry_hours=1 / 60,
        cron_hours="*", cron_minute=20, fires_signals=False,
        price_types=("B", "A"), computes_indicators=False,
        trailing_window_setting="M1_LIVE_LOOKBACK_HOURS",
    ),
    "H4": Timeframe(
        code="H4", period_minutes=240, expiry_hours=4,
        cron_hours="1,5,9,13,17,21", fires_signals=True,
    ),
    # D1 fires no signals of its own, but its candles ARE live inputs: the C1 trend
    # filter and the `dist_to_sma50_atr` / `d1_close` / `d1_sma50` features all read
    # D1. Before M8-Shadow there was no scheduled D1 job at all, so live D1 candles
    # went stale between manual ingests and every D1-derived feature on a live/shadow
    # row silently degraded. Hence a DATA-REFRESH-ONLY job (fires_signals=False).
    #
    # Schedule — 23:05 UTC daily:
    #   * OANDA's D1 bar closes at 21:00 UTC (NY DST) / 22:00 UTC (NY EST) — verified
    #     against the stored candle timestamps, which are exactly {21:00, 22:00}. 23:05
    #     is comfortably AFTER the close under both alignments (+2h05m / +1h05m).
    #   * It is BEFORE the next day's first H4 candle-close job (01:01 UTC), so every
    #     H4 signal of the trading day reads a fresh D1 trend leg.
    #   * Minute 5 (not the default 1) keeps it clear of the 21:01/22:01 H4 jobs and
    #     the 00:00 daily/weekly jobs.
    "D": Timeframe(
        code="D", period_minutes=1440, expiry_hours=24,
        cron_hours="23", cron_minute=5, fires_signals=False,
    ),
}


def get_timeframe(code: str) -> Timeframe:
    tf = TIMEFRAMES.get(code)
    if tf is None:
        raise ValueError(
            f"Unregistered timeframe '{code}'. Add it to TIMEFRAMES in "
            f"app/domain/timeframes.py (known: {', '.join(TIMEFRAMES)})."
        )
    return tf
