"""
Registry of macro/fundamental series to ingest from FRED into `macro_data`.

Single source of truth — adding a series is one entry here (mirrors the Timeframe
registry). These are external data-source reference identifiers (FRED series ids),
not business-logic magic strings; the secret (API key) and provider live in env.

Step 2 scope = yields + policy rates + VIX — all NON-revised market data, so we
fetch with FRED output_type=1 (latest) and set `release_time = ref_period +
publish_lag_days` (conservative / leakage-safe). Step 3.5 appends REVISED macro
indicators (CPI, unemployment) which need true ALFRED vintages (multi-row).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MacroSeries:
    name: str              # internal name stored in macro_data.series
    fred_id: str           # FRED series id
    category: str          # "yield" | "policy" | "vix" | "indicator"
    revised: bool          # True if values get revised (needs true ALFRED vintages); False = market data
    currency: str | None   # currency leg for yields/policy; None for global series (VIX)
    publish_lag_days: int   # release_time = ref_period + this — conservative publication lag (no leakage)
    obs_start_override: str | None = None  # per-series observation_start (ISO) overriding the global lookback;
                                           # revised indicators need ~13+ extra months for later YoY math
    role: str | None = None  # optional stable functional role for pair-INDEPENDENT global series
                             # (e.g. "us_10y_daily", "us_2y_daily", "vix"). Lets feature_builder
                             # resolve them by role, never by a hardcoded series-name literal.

    @property
    def cadence(self) -> str:
        """Publication cadence, derived from publish lag — daily market series post
        within days; OECD monthly indicators lag several weeks. Feature staleness
        floors key off this without a separate hardcoded list."""
        return "daily" if self.publish_lag_days <= _DAILY_CADENCE_MAX_LAG_DAYS else "monthly"


# A series publishing within this many days of its reference period is treated as
# daily-cadence for staleness purposes; slower (OECD monthly) series are monthly.
_DAILY_CADENCE_MAX_LAG_DAYS = 7


# Revised indicators want enough history before any backtest start that a 12-month
# YoY derivation (in feature_builder) has its lagged term. Cycle-2 moves the backtest
# start to 2021-06, so pull from 2020-01 (≥12-month CPI YoY lag before that start).
_INDICATOR_OBS_START = "2020-01-01"


# 8 currencies of the 10-pair universe: USD, EUR(DE), GBP(GB), JPY, AUD, NZD, CAD, CHF
_YIELD_10Y = {  # OECD long-term (10Y) government bond yields — MONTHLY, uniform methodology
    "US_10Y_YIELD": ("IRLTLT01USM156N", "USD"),
    "DE_10Y_YIELD": ("IRLTLT01DEM156N", "EUR"),
    "GB_10Y_YIELD": ("IRLTLT01GBM156N", "GBP"),
    "JP_10Y_YIELD": ("IRLTLT01JPM156N", "JPY"),
    "AU_10Y_YIELD": ("IRLTLT01AUM156N", "AUD"),
    "NZ_10Y_YIELD": ("IRLTLT01NZM156N", "NZD"),
    "CA_10Y_YIELD": ("IRLTLT01CAM156N", "CAD"),
    "CH_10Y_YIELD": ("IRLTLT01CHM156N", "CHF"),
}
_POLICY = {  # OECD short-term / immediate (call money) rates ≈ policy rate — MONTHLY
    "US_POLICY_RATE": ("IRSTCI01USM156N", "USD"),
    "DE_POLICY_RATE": ("IRSTCI01DEM156N", "EUR"),
    "GB_POLICY_RATE": ("IRSTCI01GBM156N", "GBP"),
    "JP_POLICY_RATE": ("IRSTCI01JPM156N", "JPY"),
    "AU_POLICY_RATE": ("IRSTCI01AUM156N", "AUD"),
    "NZ_POLICY_RATE": ("IRSTCI01NZM156N", "NZD"),
    "CA_POLICY_RATE": ("IRSTCI01CAM156N", "CAD"),
    "CH_POLICY_RATE": ("IRSTCI01CHM156N", "CHF"),
}

_MONTHLY_LAG = 45  # OECD MEI monthly series publish ~5-6 weeks after the reference month
_DAILY_LAG = 1     # daily market series (Treasury yields, VIX) post by next business day

# Revised macro indicators (Step 3.5). These ARE revised after first print, so the
# leakage rule requires the TRUE publication date of EACH vintage — fetched via FRED
# output_type=1 over the full realtime window, where every long row's realtime_start
# IS that vintage's publication date (first release + one row per revision).
# publish_lag_days is unused for these (revised=True path ignores it) but kept as a
# sane fallback. Store the RAW released index/level/rate; YoY/MoM is derived in
# feature_builder, not here.
# name -> (fred_id, role). role is the stable functional key feature_builder resolves
# by (never a hardcoded series-name literal) — mirrors the global-series role pattern.
_INDICATORS = {
    "US_CPI":           ("CPIAUCSL", "us_cpi"),             # CPI, all items, index (BLS) — revised
    "US_CORE_CPI":      ("CPILFESL", "us_core_cpi"),        # CPI less food & energy, index — revised
    "US_UNEMPLOYMENT":  ("UNRATE", "us_unemployment"),      # unemployment rate, % — revised
    "US_RETAIL_SALES":  ("RSAFS", "us_retail_sales"),       # advance retail & food services sales, $M — revised
    "EU_CPI":           ("CP0000EZ19M086NEST", "eu_cpi"),   # euro-area HICP all items, index (Eurostat) — revised
    # UK CPI: OECD MEI source (GBRCPIALLMINMEI) carries true ALFRED vintages but FRED
    # froze it at ref 2025-03 (OECD MEI discontinuation). Ingests correctly as a
    # revised series; data-currency is a separate source GAP, swap is one-line here.
    # No role — uk_cpi_yoy is dropped from the locked feature contract (frozen).
    "UK_CPI":           ("GBRCPIALLMINMEI", None),          # UK CPI all items, index (OECD MEI) — revised, frozen 2025-03
}

MACRO_SERIES: list[MacroSeries] = (
    [MacroSeries(n, fid, "yield", False, ccy, _MONTHLY_LAG) for n, (fid, ccy) in _YIELD_10Y.items()]
    + [MacroSeries(n, fid, "policy", False, ccy, _MONTHLY_LAG) for n, (fid, ccy) in _POLICY.items()]
    + [
        # US daily curve (feeds us_2s10s / us_10y; not used in the cross-country differential)
        MacroSeries("US_10Y_DAILY", "DGS10", "yield", False, "USD", _DAILY_LAG, role="us_10y_daily"),
        MacroSeries("US_2Y_DAILY", "DGS2", "yield", False, "USD", _DAILY_LAG, role="us_2y_daily"),
        # global risk sentiment
        MacroSeries("VIX", "VIXCLS", "vix", False, None, _DAILY_LAG, role="vix"),
        # global commodity — WTI crude (Cycle-2). Daily, non-revised market data;
        # mirrors VIX registration. NOT yet in any feature key list (Phase B decides).
        MacroSeries("WTI", "DCOILWTICO", "commodity", False, None, _DAILY_LAG, role="wti"),
    ]
    + [
        # Step 3.5 — revised indicators, FULL BITEMPORAL vintages (output_type=1 over
        # the full realtime window: every vintage, TRUE release_time per row)
        MacroSeries(n, fid, "indicator", True, None, _MONTHLY_LAG, _INDICATOR_OBS_START, role=role)
        for n, (fid, role) in _INDICATORS.items()
    ]
)


# ── Lookup helpers (single source of truth — feature_builder resolves series via
#    these, never via hardcoded series-name literals) ───────────────────────────
MACRO_SERIES_BY_NAME: dict[str, MacroSeries] = {s.name: s for s in MACRO_SERIES}


def series_name_for_currency(currency: str, category: str) -> str | None:
    """Resolve the monthly OECD series NAME for a currency leg + category
    ("yield" → 10Y, "policy" → policy rate). Returns None if the currency has no
    such leg. Daily US curve series are excluded (they are monthly-cadence-only
    lookups here) so the cross-country differential stays same-methodology."""
    for s in MACRO_SERIES:
        if s.category == category and s.currency == currency and s.cadence == "monthly":
            return s.name
    return None


def series_name_for_role(role: str) -> str | None:
    """Resolve a pair-independent global series NAME by its functional role."""
    for s in MACRO_SERIES:
        if s.role == role:
            return s.name
    return None
