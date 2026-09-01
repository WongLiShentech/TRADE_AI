"""
Tier-1 news spine — the deterministic, safety-critical economic-calendar events
that drive the M8 no-trade hard gate + the `tier1_event_*` features.

Covers US data (CPI, NFP) + the 8 major central-bank rate decisions
(FOMC/ECB/BoE/BoJ/RBA/RBNZ/BoC/SNB). Two date sources per event:
  - FRED release id (dynamic, past + future scheduled)  → US CPI, Non-Farm Payrolls
  - curated static date list (web-sourced ONCE, cited)   → central-bank decisions
    (FRED exposes no usable release id for rate decisions — its FOMC id is daily noise).

EXTENSIBILITY (the only thing you touch to add a currency):
  Add the bank's decision dates as a `_XXX_DATES` tuple + one `SpineEvent` row below.
  `refresh_spine` gates on the ACTIVE currencies discovered at runtime, so an event
  only ingests when a live pair uses that currency — a new pair "lights up" its bank's
  events automatically, with zero engine changes. See feedback-modular-extensible.

Times are a local announce time + IANA tz; the ingest converts to UTC with correct
DST via zoneinfo. `all_day=True` (tentative-time events, e.g. BoJ) → block the whole
day for the currency.

Date provenance: ✅ = confirmed against the bank's official calendar (cited);
                 ~ = authored from known schedule, SPOT-CHECK before live trading.
Curated 2026-06-05 over the backtest window (2024-06 → 2026-06) + forward for live.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpineEvent:
    title: str                # e.g. "FOMC Rate Decision"
    currency: str             # "USD"
    announce_local: str       # "HH:MM" local announcement time
    tz: str                   # IANA tz, e.g. "America/New_York"
    is_tier1: bool = True
    all_day: bool = False     # tentative/window time → block the whole day
    impact: str = "high"
    fred_release_id: int | None = None   # if set, dates from FRED get_release_dates
    static_dates: tuple[str, ...] = ()    # else, curated ISO dates


# ── USD — FOMC policy decision (day-2 statement, 14:00 ET) ───────────────────────
# ✅ federalreserve.gov/monetarypolicy/fomccalendars.htm
_FOMC_DATES = (
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
)

# ── EUR — ECB Governing Council monetary-policy meeting (14:15 CET) ──────────────
# ✅ ecb.europa.eu/press/calendars/mgcgc (2024 H2 + 2025 + 2026 H1 confirmed); ~2024 H1
_ECB_DATES = (
    "2024-01-25", "2024-03-07", "2024-04-11", "2024-06-06", "2024-07-18",
    "2024-09-12", "2024-10-17", "2024-12-12",
    "2025-01-30", "2025-03-06", "2025-04-17", "2025-06-05", "2025-07-24",
    "2025-09-11", "2025-10-30", "2025-12-18",
    "2026-02-05", "2026-03-19", "2026-04-30", "2026-06-18",
)

# ── GBP — BoE MPC Bank Rate announcement (12:00 London) ──────────────────────────
# ✅ bankofengland.co.uk/monetary-policy/upcoming-mpc-dates (Dec-2025 + Feb/Jun-2026); ~rest
_BOE_DATES = (
    "2024-02-01", "2024-03-21", "2024-05-09", "2024-06-20", "2024-08-01",
    "2024-09-19", "2024-11-07", "2024-12-19",
    "2025-02-06", "2025-03-20", "2025-05-08", "2025-06-19", "2025-08-07",
    "2025-09-18", "2025-11-06", "2025-12-18",
    "2026-02-05", "2026-03-19", "2026-05-07", "2026-06-18",
)

# ── JPY — BoJ Monetary Policy Meeting (decision day; tentative time → all_day) ────
# ✅ boj.or.jp/en/mopo/mpmsche_minu (2025 confirmed); ~2024 + 2026
_BOJ_DATES = (
    "2024-01-23", "2024-03-19", "2024-04-26", "2024-06-14", "2024-07-31",
    "2024-09-20", "2024-10-31", "2024-12-19",
    "2025-01-24", "2025-03-19", "2025-05-01", "2025-06-17", "2025-07-31",
    "2025-09-19", "2025-10-30", "2025-12-19",
    "2026-01-23", "2026-03-19", "2026-04-30", "2026-06-16",
)

# ── AUD — RBA Monetary Policy Board decision (14:30 Sydney, day-2) ───────────────
# ✅ rba.gov.au/schedules-events/board-meeting-schedules (2026 confirmed); ~2024 + 2025
_RBA_DATES = (
    "2024-02-06", "2024-03-19", "2024-05-07", "2024-06-18", "2024-08-06",
    "2024-09-24", "2024-11-05", "2024-12-10",
    "2025-02-18", "2025-04-01", "2025-05-20", "2025-07-08", "2025-08-12",
    "2025-09-30", "2025-11-04", "2025-12-09",
    "2026-02-03", "2026-03-17", "2026-05-05", "2026-06-16",
)

# ── NZD — RBNZ Official Cash Rate decision (14:00 Auckland) ──────────────────────
# ✅ rbnz.govt.nz OCR decision dates (late-2026 confirmed); ~2024 + 2025 + 2026 H1
_RBNZ_DATES = (
    "2024-02-28", "2024-04-10", "2024-05-22", "2024-07-10", "2024-08-14",
    "2024-10-09", "2024-11-27",
    "2025-02-19", "2025-04-09", "2025-05-28", "2025-07-09", "2025-08-20",
    "2025-10-08", "2025-11-26",
    "2026-02-25", "2026-04-15", "2026-05-27",
)

# ── CAD — BoC policy interest rate announcement (09:45 ET) ───────────────────────
# ✅ bankofcanada.ca rate-announcement schedules (Mar/Apr-2026 confirmed); ~rest
_BOC_DATES = (
    "2024-01-24", "2024-03-06", "2024-04-10", "2024-06-05", "2024-07-24",
    "2024-09-04", "2024-10-23", "2024-12-11",
    "2025-01-29", "2025-03-12", "2025-04-16", "2025-06-04", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-10",
)

# ── CHF — SNB monetary policy assessment (09:30 Zurich, quarterly) ───────────────
# ✅ snb.ch monetary-policy decisions (2024 + 2025 + Mar-2026 confirmed); ~Jun-2026
_SNB_DATES = (
    "2024-03-21", "2024-06-20", "2024-09-26", "2024-12-12",
    "2025-03-20", "2025-06-19", "2025-09-25", "2025-12-11",
    "2026-03-19", "2026-06-18",
)


SPINE: list[SpineEvent] = [
    # US data Tier-1 — dates pulled live from FRED release/dates (past + future).
    SpineEvent("US CPI", "USD", "08:30", "America/New_York", fred_release_id=10),
    SpineEvent("US Non-Farm Payrolls", "USD", "08:30", "America/New_York", fred_release_id=50),
    # Central-bank rate decisions — curated, cited (FRED has no usable release id).
    SpineEvent("FOMC Rate Decision", "USD", "14:00", "America/New_York", static_dates=_FOMC_DATES),
    SpineEvent("ECB Rate Decision", "EUR", "14:15", "Europe/Berlin", static_dates=_ECB_DATES),
    SpineEvent("BoE Rate Decision", "GBP", "12:00", "Europe/London", static_dates=_BOE_DATES),
    # BoJ announces at a tentative time (no fixed minute) → all_day blocks the JPY day.
    SpineEvent("BoJ Rate Decision", "JPY", "00:00", "Asia/Tokyo", all_day=True, static_dates=_BOJ_DATES),
    SpineEvent("RBA Rate Decision", "AUD", "14:30", "Australia/Sydney", static_dates=_RBA_DATES),
    SpineEvent("RBNZ Rate Decision", "NZD", "14:00", "Pacific/Auckland", static_dates=_RBNZ_DATES),
    SpineEvent("BoC Rate Decision", "CAD", "09:45", "America/Toronto", static_dates=_BOC_DATES),
    SpineEvent("SNB Rate Decision", "CHF", "09:30", "Europe/Zurich", static_dates=_SNB_DATES),
]
