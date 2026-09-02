"""
APScheduler wrapper. Registered jobs:

- Sunday 00:00 UTC weekly chain:
    journal_generator → wiki_ingestor → wiki_promoter
- Daily 00:00 UTC:
    circuit_breaker check
- Candle jobs, one per Timeframe registry entry that declares `cron_hours`:
    H4 01:01/05:01/09:01/13:01/17:01/21:01 UTC → full candle-close pipeline
       (candles → indicators → signal engine → risk engine → M8 shadow observation)
    D  23:05 UTC                               → refresh only (candles → indicators)
    M1 hourly at :20 UTC                       → trailing-window Bid/Ask top-up
       (M1_LIVE_LOOKBACK_HOURS); the intrabar series the M8-Shadow outcome
       resolver walks to label live shadow trades (GAP-13)
- Shadow outcome resolver, every SHADOW_RESOLVER_INTERVAL_HOURS at :35 UTC:
    fills outcome/rr_actual/exit_* on pending stage='shadow' rows once their
    horizon has elapsed AND the M1 coverage to resolve them honestly exists.
- Fundamentals Cron A/B (Sun 22:00 + every FUNDAMENTAL_INTRADAY_REFRESH_HOURS at
  :15): news spine + macro refresh, followed by TWO watchdogs — news-spine
  staleness and per-series MACRO staleness. The macro watchdog checks the stored
  data, not the insert count, because "0 rows inserted" means both "already
  current" and "upstream feed is dead".

Lifecycle managed by main.py lifespan: started on app startup, stopped on
shutdown. Single global instance — fine for single-process Phase 1 dev.

Note: ingester is invoked but tolerant of missing ANTHROPIC_API_KEY
(returns a structured "skipped" result rather than raising).
"""
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_settings
from app.database import SessionLocal
from app.domain.macro_series import MACRO_SERIES
from app.domain.timeframes import TIMEFRAMES
from app.services.circuit_breaker import check_circuit_breaker
from app.services.journal_generator import generate_weekly_journal
from app.services.fundamental.refresh import refresh_macro, stale_series
from app.services.news_calendar.spine import refresh_spine, spine_staleness_hours
from app.services.pipeline import (
    run_candle_close_pipeline,
    run_candle_refresh_pipeline,
    run_trailing_window_refresh,
)
from app.services.shadow import resolve_pending
from app.services.wiki_ingestor import ingest_journal
from app.services.wiki_promoter import promote_pages

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None

# How late a job may fire and still run, in seconds.
#
# APScheduler's default misfire_grace_time is ONE SECOND: if the scheduler thread
# does not get to a job within 1s of its trigger time, the run is silently dropped
# with a "missed by" warning. On a burstable/shared vCPU (Oracle Cloud ARM free
# tier), a GC pause, a concurrent candle backfill or a noisy-neighbour scheduling
# delay clears that bar routinely — and every one of those is a lost H4 candle-close
# pipeline, i.e. a lost signal AND a lost shadow observation, with no way to tell
# afterwards that it should have run.
#
# 300s is chosen against the job spacing, not arbitrarily: the tightest cadence in
# this scheduler is one job per hour, and the H4/fundamentals/M1/resolver jobs are
# deliberately offset by 14+ minutes from each other (:01, :15, :20, :35). A job may
# therefore run up to 5 minutes late without any risk of colliding with the next
# one, and coalesce=True guarantees a backlog collapses to a single run.
_MISFIRE_GRACE_SECONDS = 300


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    settings = get_settings()
    # job_defaults applies to EVERY job, including any added later — a new job
    # cannot accidentally inherit the 1s default by forgetting the kwarg.
    _scheduler = BackgroundScheduler(
        timezone="UTC",
        job_defaults={
            "misfire_grace_time": _MISFIRE_GRACE_SECONDS,
            "coalesce": True,
            "max_instances": 1,
        },
    )
    _scheduler.add_job(
        _weekly_chain,
        CronTrigger(day_of_week="sun", hour=0, minute=0, timezone="UTC"),
        id="weekly_chain",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    _scheduler.add_job(
        _daily_circuit_breaker,
        CronTrigger(hour=0, minute=0, timezone="UTC"),
        id="daily_circuit_breaker",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    # Candle jobs are built from the Timeframe registry — one job per timeframe that
    # declares a cron schedule, with a minute offset so the broker finalizes the
    # closing candle before we fetch. The registry flags select the job body, so
    # adding/repointing a scheduled timeframe is a registry edit, never a scheduler
    # edit:
    #   fires_signals=True         → full candle-close pipeline (signals + risk +
    #                                shadow)                                    — H4
    #   fires_signals=False        → data refresh only (candles + indicators)    — D1
    #   + trailing_window_setting  → bounded trailing-window top-up of the
    #                                registry's price_types                      — M1
    # D1 is refresh-only at 23:05 UTC: after the 21:00/22:00 UTC daily close under
    # both NY alignments, and before the next day's first H4 job (01:01 UTC), so every
    # H4 signal reads a fresh D1 trend leg and fresh D1-derived features.
    # M1 is a trailing-window Bid/Ask top-up: the outcome resolver cannot label a live
    # shadow trade without intrabar Bid/Ask over its hold window (GAP-13).
    scheduled_codes: list[str] = []
    for tf in TIMEFRAMES.values():
        if tf.cron_hours is None:
            continue
        if tf.fires_signals:
            job, suffix, note = _make_candle_close_job(tf.code), "candle_close_pipeline", ""
        elif tf.trailing_window_setting is not None:
            job, suffix = _make_trailing_window_job(tf.code), "trailing_window_refresh"
            note = f" (trailing {getattr(settings, tf.trailing_window_setting)}h {'/'.join(tf.price_types)})"
        else:
            job, suffix, note = _make_candle_refresh_job(tf.code), "candle_refresh", " (refresh-only)"
        _scheduler.add_job(
            job,
            CronTrigger(hour=tf.cron_hours, minute=tf.cron_minute, timezone="UTC"),
            id=f"{tf.code.lower()}_{suffix}",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        )
        scheduled_codes.append(f"{tf.code}@{tf.cron_hours}:{tf.cron_minute:02d}{note}")

    # M8-Shadow Phase 3 — outcome resolver. Its OWN cron job, deliberately not bolted
    # onto a candle-close body: resolvability is driven by the hold horizon and M1
    # coverage, not by a bar closing, and a slow resolve pass must never delay the
    # signal path. Idempotent (selector is `closed_at IS NULL`), so overlapping or
    # repeated runs are harmless; max_instances=1 + coalesce keeps the threadpool
    # clean anyway. Minute 35 sits clear of the H4 (:01), fundamentals (:15) and M1
    # (:20) jobs, and gives the M1 top-up 15 minutes to land first.
    _scheduler.add_job(
        _shadow_resolver,
        CronTrigger(
            hour=f"*/{settings.SHADOW_RESOLVER_INTERVAL_HOURS}", minute=35, timezone="UTC"
        ),
        id="shadow_outcome_resolver",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    # Sandbox reconciliation + TIME EXIT. Registered unconditionally: turning
    # sandbox recording off must never strand an OPEN POSITION, and the pass is a
    # single indexed query when there is nothing to do. Minute 50 sits clear of the
    # H4 (:01), fundamentals (:15), M1 (:20) and shadow resolver (:35) jobs.
    #
    # This job is the ONLY thing that applies SIGNAL_MAX_HOLD_BARS to a live
    # position — OANDA enforces the attached stop and target, but knows nothing
    # about a time limit. Hourly, not every 4h, so a trade that closes early is
    # reconciled promptly rather than looking open for hours.
    _scheduler.add_job(
        _sandbox_sync,
        CronTrigger(hour="*", minute=50, timezone="UTC"),
        id="sandbox_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    # Fundamentals orchestration — Cron A (weekly reconcile at week-open) + Cron B
    # (intraday poll, every FUNDAMENTAL_INTRADAY_REFRESH_HOURS). Both refresh the news
    # spine (forward calendar) + macro (recent releases) idempotently. max_instances=1
    # + coalesce prevents overlap in the single-process threadpool.
    _scheduler.add_job(
        _weekly_fundamentals,
        CronTrigger(day_of_week="sun", hour=22, minute=0, timezone="UTC"),
        id="weekly_fundamentals", replace_existing=True, max_instances=1, coalesce=True,
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    _scheduler.add_job(
        _intraday_fundamentals,
        CronTrigger(hour=f"*/{settings.FUNDAMENTAL_INTRADAY_REFRESH_HOURS}", minute=15, timezone="UTC"),
        id="intraday_fundamentals", replace_existing=True, max_instances=1, coalesce=True,
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    _scheduler.start()
    logger.info(
        "scheduler started; jobs: weekly_chain (Sun 00:00), daily_circuit_breaker (00:00), "
        "weekly_fundamentals (Sun 22:00), intraday_fundamentals (*/%dh), "
        "shadow_outcome_resolver (*/%dh at :35), candle jobs: %s",
        settings.FUNDAMENTAL_INTRADAY_REFRESH_HOURS,
        settings.SHADOW_RESOLVER_INTERVAL_HOURS,
        ", ".join(scheduled_codes),
    )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is None:
        return
    _scheduler.shutdown(wait=False)
    _scheduler = None
    logger.info("scheduler stopped")


def _weekly_chain() -> None:
    """Sunday journal → wiki-ingest → strategy-page promotion.

    Gated on ``WIKI_INGEST_ENABLED``. The whole chain is vault I/O: it writes the
    journal under ``VAULT_PATH``, hands it to the Anthropic API, then rewrites
    strategy pages. On a server where the Obsidian vault does not exist (the
    deployment case — see backend/.env.production.example) every step is either a
    no-op or, worse, an Anthropic call that BURNS CREDITS summarising an empty
    journal into a directory nobody reads.

    ``wiki_ingestor.ingest_journal`` already self-gates on the same flag, but
    ``generate_weekly_journal`` and ``promote_pages`` do not — they would happily
    create a ``VAULT_PATH`` tree from nothing. Gating the whole chain in one place
    keeps the three services consistent and leaves the manual endpoints
    (POST /journal/generate, POST /wiki/ingest) working for the dev machine.
    """
    settings = get_settings()
    if not settings.WIKI_INGEST_ENABLED:
        logger.info(
            "weekly_chain skipped — WIKI_INGEST_ENABLED is false (no vault on this host)"
        )
        return
    db = SessionLocal()
    try:
        result = generate_weekly_journal(db, settings, week_offset=0)
        logger.info("scheduled journal: %s", result)
        try:
            ingest_result = ingest_journal(result["week_label"], settings)
            logger.info("scheduled ingest: %s", ingest_result)
        except ValueError as exc:
            logger.warning("scheduled ingest skipped: %s", exc)
        promote_result = promote_pages(db, settings)
        logger.info("scheduled promote: %s", promote_result)
    except Exception:
        logger.exception("weekly_chain failed")
    finally:
        db.close()


def _make_candle_close_job(code: str):
    """Build a full candle-close pipeline job bound to one timeframe code."""

    def _job() -> None:
        settings = get_settings()
        try:
            run_candle_close_pipeline(code, settings)
        except Exception:
            logger.exception("%s_candle_close_pipeline failed", code.lower())

    return _job


def _make_candle_refresh_job(code: str):
    """Build a DATA-REFRESH-ONLY job (candles + indicators) for one timeframe code.

    Used by timeframes with ``fires_signals=False`` (D1): their candles feed other
    timeframes' decisions and features, so they must stay fresh, but they must never
    author a signal or reach the risk/shadow path.
    """

    def _job() -> None:
        settings = get_settings()
        try:
            run_candle_refresh_pipeline(code, settings)
        except Exception:
            logger.exception("%s_candle_refresh failed", code.lower())

    return _job


def _make_trailing_window_job(code: str):
    """Build a bounded trailing-window candle top-up job for one timeframe code.

    Used by high-volume timeframes (M1) whose registry entry names a lookback
    setting: each run re-fetches ``[now - lookback, now]`` rather than resuming from
    the newest stored bar, so a scheduled run can never turn into an unbounded
    backfill after an outage.
    """

    def _job() -> None:
        settings = get_settings()
        try:
            run_trailing_window_refresh(code, settings)
        except Exception:
            logger.exception("%s_trailing_window_refresh failed", code.lower())

    return _job


def _sandbox_sync() -> None:
    """Reconcile open sandbox positions with the broker and apply the time exit.

    Registered unconditionally and no-ops immediately outside sandbox mode. The
    alternative — gating registration on the mode — would mean that switching back
    to observe with a position open leaves it open forever, because the only job
    that could close it is no longer scheduled.
    """
    settings = get_settings()
    db = SessionLocal()
    try:
        result = sync_open_trades(db, settings)
        if result.get("checked"):
            logger.info("scheduled sandbox sync: %s", result)
    except Exception:                                              # noqa: BLE001
        logger.exception("scheduled sandbox sync failed")
    finally:
        db.close()


def _shadow_resolver() -> None:
    """Resolve pending M8-Shadow rows whose outcome has become knowable.

    Registered unconditionally (not behind ``SHADOW_MODE_ENABLED``): turning shadow
    RECORDING off must not strand rows that were already recorded and still deserve
    an honest outcome. With an empty queue the pass is a single indexed query.
    """
    settings = get_settings()
    db = SessionLocal()
    try:
        result = resolve_pending(db, settings)
        logger.info("scheduled shadow resolver: %s", result)
    except Exception:
        logger.exception("shadow_outcome_resolver failed")
    finally:
        db.close()


def _daily_circuit_breaker() -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        result = check_circuit_breaker(db, settings)
        logger.info("scheduled circuit_breaker: %s", result)
    except Exception:
        logger.exception("daily_circuit_breaker failed")
    finally:
        db.close()


def _refresh_fundamentals(settings, db) -> None:
    """Shared Cron A/B body: refresh the news spine (forward calendar) + macro
    (recent releases), then alert on any per-source failure or global staleness."""
    now = datetime.now(timezone.utc)
    spine_start = (now - timedelta(days=2)).strftime("%Y-%m-%d")       # catch near-term reschedules
    spine_end = (now + timedelta(days=settings.NEWS_REFRESH_FORWARD_DAYS)).strftime("%Y-%m-%d")
    spine_summary = refresh_spine(db, settings, spine_start, spine_end)
    macro_summary = refresh_macro(db, settings, settings.FUNDAMENTAL_REFRESH_LOOKBACK_DAYS)
    logger.info("fundamentals refresh: spine=%s macro=%s", spine_summary, macro_summary)
    # Per-source failure alert — fixes the fail-open watchdog (a dead source returns -1
    # but wouldn't move global max(fetched_at), so the staleness check alone misses it).
    failed = [k for k, v in {**spine_summary, **macro_summary}.items() if v == -1]
    if failed:
        logger.warning("fundamentals refresh FAILURES: %s", failed)
    stale = spine_staleness_hours(db)
    if stale is not None and stale > settings.FUNDAMENTAL_STALENESS_ALERT_HOURS:
        logger.warning(
            "news spine STALE: %.1fh (> %.0fh threshold)",
            stale, settings.FUNDAMENTAL_STALENESS_ALERT_HOURS,
        )
    _check_macro_staleness(settings, db)


def _check_macro_staleness(settings, db) -> None:
    """Warn when a macro series' NEWEST release has aged past its cadence threshold.

    ``refresh_macro`` returns 0 for a series both when it is already current and when
    its upstream feed died — the two are indistinguishable from the insert count
    alone. That ambiguity is what let the daily FRED series (US 2s10s, VIX, WTI) go
    three weeks stale in silence while ``build_features`` NaN-ed six of the nineteen
    model-core features on every live signal.

    So the check is made against the DATA, not the insert count, using the exact
    thresholds ``feature_builder`` applies: any series the feature builder would
    refuse to read is named in a WARNING here.

    Alert routing note: the sibling news-spine staleness check logs rather than
    routing through ``services.alerts`` — this deliberately matches it, so both
    fundamentals watchdogs move to the alert channel together in one change rather
    than drifting apart.
    """
    stale = stale_series(db, settings)
    if not stale:
        logger.info("macro staleness: all %d registered series fresh", len(MACRO_SERIES))
        return
    detail = ", ".join(
        f"{r.series}({r.cadence}, "
        f"{'NO DATA' if r.age_days is None else f'{r.age_days}d'} > {r.threshold_days}d)"
        for r in stale
    )
    logger.warning(
        "macro series STALE (%d/%d) — every feature derived from these is NaN on live "
        "signals: %s",
        len(stale), len(MACRO_SERIES), detail,
    )


def _weekly_fundamentals() -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        _refresh_fundamentals(settings, db)
    except Exception:
        logger.exception("weekly_fundamentals failed")
    finally:
        db.close()


def _intraday_fundamentals() -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        _refresh_fundamentals(settings, db)
    except Exception:
        logger.exception("intraday_fundamentals failed")
    finally:
        db.close()
