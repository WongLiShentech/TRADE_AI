from datetime import datetime

import httpx

from app.config import Settings
from app.services.fundamental.base import (
    FundamentalDataProvider,
    FundamentalSnapshot,
    SeriesObservation,
    SeriesVintage,
)


class FredFundamental(FundamentalDataProvider):
    """
    FRED — free macro + market series. For Step 2 (non-revised market data:
    yields, policy rates, VIX) we use the default observation view
    (`output_type=1`); the caller derives `release_time` from a conservative
    publication lag. Revised series (CPI/unemployment, Step 3.5) will need true
    ALFRED vintages (`output_type`/`vintage_dates`) — handled separately.

    Errors NEVER include the request URL (it carries the api_key) — they are
    sanitized to series id + status only.
    """

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.FUNDAMENTAL_DATA_API_KEY
        self._base_url = "https://api.stlouisfed.org/fred"

    def get_snapshot(self, instrument, at) -> list[FundamentalSnapshot]:  # legacy live shape
        raise NotImplementedError("FRED uses get_series (series-centric)")

    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
    ) -> list[SeriesObservation]:
        if not self._api_key:
            raise ValueError("FUNDAMENTAL_DATA_API_KEY (FRED key) is not set")

        params: dict[str, object] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        if observation_start:
            params["observation_start"] = observation_start

        url = f"{self._base_url}/series/observations"
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:  # redact: response URL contains the api_key
            raise RuntimeError(f"FRED {series_id}: HTTP {e.response.status_code}") from None
        except httpx.HTTPError as e:
            raise RuntimeError(f"FRED {series_id}: {type(e).__name__}") from None

        out: list[SeriesObservation] = []
        for o in resp.json().get("observations", []):
            raw = o.get("value", ".")
            if raw in (".", "", None):
                continue  # FRED encodes missing as "."
            out.append(SeriesObservation(ref_period=o["date"], value=float(raw)))
        return out

    # FRED/ALFRED realtime bounds — spanning the full window makes output_type=4
    # return EVERY observation's first-release vintage (the default realtime window
    # is "today", which yields zero vintages and a 400). Not secrets, just API bounds.
    _RT_MIN = "1776-07-04"
    _RT_MAX = "9999-12-31"

    def get_series_first_release(
        self,
        series_id: str,
        observation_start: str | None = None,
    ) -> list[SeriesVintage]:
        """
        First-release vintages for a REVISED series via FRED `output_type=4`
        (observations, initial release only). Each returned observation's
        `realtime_start` IS its true first-publication date, so `release_time` is
        leakage-safe with no synthetic lag, and `value` is the FIRST print (not the
        latest revision). The full realtime window (_RT_MIN.._RT_MAX) is required —
        output_type=4 otherwise defaults realtime to today and returns no vintages.

        Errors NEVER include the request URL (it carries the api_key).
        """
        if not self._api_key:
            raise ValueError("FUNDAMENTAL_DATA_API_KEY (FRED key) is not set")

        params: dict[str, object] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "output_type": 4,                 # observations, initial release only
            "realtime_start": self._RT_MIN,
            "realtime_end": self._RT_MAX,
        }
        if observation_start:
            params["observation_start"] = observation_start

        url = f"{self._base_url}/series/observations"
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:  # redact: response URL contains the api_key
            raise RuntimeError(f"FRED {series_id}: HTTP {e.response.status_code}") from None
        except httpx.HTTPError as e:
            raise RuntimeError(f"FRED {series_id}: {type(e).__name__}") from None

        out: list[SeriesVintage] = []
        for o in resp.json().get("observations", []):
            raw = o.get("value", ".")
            if raw in (".", "", None):
                continue  # FRED encodes missing as "."
            rt = o.get("realtime_start")
            if not rt:
                continue  # no first-publication date → cannot place point-in-time
            out.append(
                SeriesVintage(
                    ref_period=o["date"],
                    release_time=datetime.strptime(rt, "%Y-%m-%d"),
                    value=float(raw),
                )
            )
        return out

    def get_series_all_vintages(
        self,
        series_id: str,
        observation_start: str | None = None,
    ) -> list[SeriesVintage]:
        """
        FULL BITEMPORAL vintages for a REVISED series via FRED `output_type=1`
        ("observations by real-time period", the default) over the FULL realtime
        window (_RT_MIN.._RT_MAX). In this mode FRED returns LONG format with ONE
        ROW PER VINTAGE: each row carries `date` (ref_period), `value` (that
        vintage's value), and `realtime_start` (the date THIS value became public =
        release_time). FRED already collapses consecutive equal values into a single
        row spanning [realtime_start, realtime_end], so each row = a distinct value
        that held over that window. The earliest realtime_start per ref_period is the
        first release; each subsequent row is a revision.

        NOTE: `output_type=2` (all-vintages WIDE) is unparseable and MUST NOT be used
        — output_type=1 + wide realtime is the validated all-vintages mechanism.

        `release_time` = realtime_start of each row, so the result is leakage-safe
        with no synthetic lag. Errors NEVER include the request URL (it carries the
        api_key).
        """
        if not self._api_key:
            raise ValueError("FUNDAMENTAL_DATA_API_KEY (FRED key) is not set")

        params: dict[str, object] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "output_type": 1,                 # observations by real-time period (all vintages, LONG)
            "realtime_start": self._RT_MIN,
            "realtime_end": self._RT_MAX,
        }
        if observation_start:
            params["observation_start"] = observation_start

        url = f"{self._base_url}/series/observations"
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:  # redact: response URL contains the api_key
            raise RuntimeError(f"FRED {series_id}: HTTP {e.response.status_code}") from None
        except httpx.HTTPError as e:
            raise RuntimeError(f"FRED {series_id}: {type(e).__name__}") from None

        out: list[SeriesVintage] = []
        for o in resp.json().get("observations", []):
            raw = o.get("value", ".")
            if raw in (".", "", None):
                continue  # FRED encodes missing as "."
            rt = o.get("realtime_start")
            if not rt:
                continue  # no publication date → cannot place point-in-time
            out.append(
                SeriesVintage(
                    ref_period=o["date"],
                    release_time=datetime.strptime(rt, "%Y-%m-%d"),
                    value=float(raw),
                )
            )
        return out

    def get_release_dates(
        self,
        release_id: int,
        start: str | None = None,
        end: str | None = None,
    ) -> list[str]:
        """
        ISO release dates for a FRED release (e.g. CPI, Employment Situation, FOMC).
        `include_release_dates_with_no_data=true` returns FUTURE scheduled dates too,
        so this serves both backtest history and forward live scheduling. Filtered
        client-side to [start, end].
        """
        if not self._api_key:
            raise ValueError("FUNDAMENTAL_DATA_API_KEY (FRED key) is not set")

        params: dict[str, object] = {
            "release_id": release_id,
            "api_key": self._api_key,
            "file_type": "json",
            "include_release_dates_with_no_data": "true",
            "sort_order": "asc",
            "limit": 10000,
        }
        url = f"{self._base_url}/release/dates"
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"FRED release {release_id}: HTTP {e.response.status_code}") from None
        except httpx.HTTPError as e:
            raise RuntimeError(f"FRED release {release_id}: {type(e).__name__}") from None

        dates = [d["date"] for d in resp.json().get("release_dates", []) if d.get("date")]
        if start:
            dates = [d for d in dates if d >= start]
        if end:
            dates = [d for d in dates if d <= end]
        return dates
