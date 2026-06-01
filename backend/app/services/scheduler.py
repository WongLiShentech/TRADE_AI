"""
APScheduler wrapper. Registered jobs:

- Sunday 00:00 UTC weekly chain:
    journal_generator → wiki_ingestor → wiki_promoter
- Daily 00:00 UTC:
    circuit_breaker check

Lifecycle managed by main.py lifespan: started on app startup, stopped on
shutdown. Single global instance — fine for single-process Phase 1 dev.

Note: ingester is invoked but tolerant of missing ANTHROPIC_API_KEY
(returns a structured "skipped" result rather than raising).
"""
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_settings
from app.database import SessionLocal
from app.services.circuit_breaker import check_circuit_breaker
from app.services.journal_generator import generate_weekly_journal
from app.services.pipeline import run_candle_close_pipeline
from app.services.wiki_ingestor import ingest_journal
from app.services.wiki_promoter import promote_pages

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _weekly_chain,
        CronTrigger(day_of_week="sun", hour=0, minute=0, timezone="UTC"),
        id="weekly_chain",
        replace_existing=True,
    )
    _scheduler.add_job(
        _daily_circuit_breaker,
        CronTrigger(hour=0, minute=0, timezone="UTC"),
        id="daily_circuit_breaker",
        replace_existing=True,
    )
    # M5: candle-close pipeline. One-minute offset so the broker has time to
    # finalize the closing candle before we fetch. H4 is the only signal
    # timeframe per SIGNAL_GRANULARITIES; D1 candles are still ingested for
    # the trend filter but D1 never fires its own signals.
    _scheduler.add_job(
        _h4_pipeline_job,
        CronTrigger(hour="1,5,9,13,17,21", minute=1, timezone="UTC"),
        id="h4_candle_close_pipeline",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "scheduler started; jobs: weekly_chain (Sun 00:00 UTC), "
        "daily_circuit_breaker (00:00 UTC), "
        "h4_candle_close_pipeline (HH:01 every 4h)"
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
    settings = get_settings()
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


def _h4_pipeline_job() -> None:
    settings = get_settings()
    try:
        run_candle_close_pipeline("H4", settings)
    except Exception:
        logger.exception("h4_candle_close_pipeline failed")


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


# Prevent unused-import complaints; datetime/timezone reserved for future jitter logic.
_ = (datetime, timezone)
