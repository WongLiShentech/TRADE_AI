import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s — %(message)s")

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.brokers.router import get_broker_router
from app.config import Settings, get_settings
from app.database import Base, SessionLocal, engine
import app.models  # noqa: F401  — ensure all models register with Base.metadata
from app.api.v1.router import router as v1_router
from app.models.instrument import Instrument as InstrumentModel
from app.services import price_stream
from app.services.ml import inference as inf
from app.services.price_stream import start_stream, stop_stream
from app.services.scheduler import start_scheduler, stop_scheduler

logger = logging.getLogger(__name__)

settings = get_settings()

# Health vocabulary. Named constants so the healthcheck contract is defined in one
# place and never spelled as an inline literal by a caller.
HEALTH_OK = "ok"
HEALTH_DEGRADED = "degraded"
HEALTH_UNHEALTHY = "unhealthy"

# Whether this process was SUPPOSED to have a live price stream. Set during
# lifespan: with zero active instruments the stream is legitimately never started,
# and a health probe must not report that as a fault.
_stream_expected: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema is created and migrated by Alembic (`alembic upgrade head`), not by
    # create_all(). In the container this runs in entrypoint.sh before uvicorn;
    # in dev it is a manual step. Alembic stays the single source of truth.
    global _stream_expected

    _ensure_bot_state_row()
    _assert_shadow_model_loadable(settings)
    start_scheduler()

    broker_router = get_broker_router()
    db = SessionLocal()
    try:
        active = db.query(InstrumentModel).filter_by(is_active=True).all()
        symbols = [i.symbol for i in active]
    finally:
        db.close()

    if symbols:
        await start_stream(symbols, broker_router, settings)
        _stream_expected = True
        logger.info("price stream started for %d instruments", len(symbols))
    else:
        _stream_expected = False
        logger.info(
            "no active instruments — price stream skipped "
            "(run POST /instruments/sync then restart)"
        )

    try:
        yield
    finally:
        await stop_stream()
        stop_scheduler()


def _ensure_bot_state_row() -> None:
    """Make sure the single-row bot_state table has its row before any request."""
    from app.models.bot_state import BotState

    db = SessionLocal()
    try:
        existing = db.query(BotState).filter_by(id=1).first()
        if existing is None:
            db.add(BotState(id=1, paused=False))
            db.commit()
    finally:
        db.close()


def _assert_shadow_model_loadable(settings: Settings) -> None:
    """Refuse to start when shadow mode is on but its ML artifact will not load.

    This is a FAIL-LOUD startup assertion, deliberately different from the per-cycle
    behaviour in ``services.pipeline._load_shadow_model`` (which logs and continues,
    correctly: a transient problem must never take the rule engine down mid-session).

    The two are not in conflict — they answer different questions:

      * At startup: "is this deployment configured correctly?" A missing artifact
        here is a DEPLOYMENT defect, not a transient. The classic instance is a
        container built with ``models/`` excluded from the build context (the
        directory is gitignored, so it is absent from a fresh clone too). The
        process would boot, pass every TCP/HTTP probe, run its scheduler on time,
        and record ZERO shadow rows — for as long as nobody looked. An always-on
        observation deployment whose only output is silence is the exact failure
        this platform cannot detect from the outside.
      * Per cycle: "can we observe THIS signal?" Log and carry on.

    Crashing at boot converts an invisible, indefinite data loss into a container
    that visibly refuses to start — which the restart policy and the operator both
    notice immediately.

    Raises:
        RuntimeError: ``SHADOW_MODE_ENABLED`` is true and ``ML_MODEL_PATH`` cannot be
            loaded and contract-validated.
    """
    if not settings.SHADOW_MODE_ENABLED:
        logger.info("SHADOW_MODE_ENABLED=false — ML artifact not loaded at startup")
        return
    try:
        loaded = inf.load_model(settings)
    except Exception as exc:
        raise RuntimeError(
            f"SHADOW_MODE_ENABLED=true but the ML artifact at ML_MODEL_PATH="
            f"{settings.ML_MODEL_PATH!r} could not be loaded "
            f"({type(exc).__name__}: {exc}). Refusing to start: the process would "
            f"otherwise look healthy while recording zero shadow observations. "
            f"Resolved path: {inf.resolve_model_path(settings)}. If this is a "
            f"container, confirm backend/models/ reached the image — .dockerignore "
            f"must NOT exclude it. To run without ML observation set "
            f"SHADOW_MODE_ENABLED=false explicitly."
        ) from exc
    logger.info(
        "startup check OK — ML artifact loaded: model_id=%s promoted=%s",
        loaded.model_id, loaded.promoted,
    )


app = FastAPI(title="Trading Platform", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1")


# ── health / readiness ───────────────────────────────────────────────────────
class ComponentHealth(BaseModel):
    """Verdict for one checked subsystem."""

    status: str = Field(..., description=f"{HEALTH_OK} | {HEALTH_DEGRADED} | {HEALTH_UNHEALTHY}")
    detail: Optional[str] = Field(None, description="Human-readable reason, if not ok")


class HealthResponse(BaseModel):
    """Aggregate readiness of the process.

    ``status`` is the worst component status. HTTP 503 accompanies
    ``unhealthy``; ``degraded`` still returns 200 (the process is doing its job,
    something is merely suspicious — a weekend market close makes the price cache
    legitimately stale, and a container must not be restarted for that).
    """

    status: str
    checked_at: datetime
    database: ComponentHealth
    price_stream: ComponentHealth
    ml_model: ComponentHealth


def _check_database() -> ComponentHealth:
    """Prove a real session can execute a real statement (not just that a pool exists)."""
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return ComponentHealth(status=HEALTH_OK, detail=None)
    except Exception as exc:  # noqa: BLE001 — any DB failure is unhealthy
        return ComponentHealth(
            status=HEALTH_UNHEALTHY, detail=f"{type(exc).__name__}: {exc}"
        )
    finally:
        db.close()


def _price_staleness_budget_seconds(settings: Settings) -> float:
    """Seconds of tick silence tolerated before the stream is called degraded.

    Derived from existing config rather than a new knob:
    ``STREAM_HEARTBEAT_TIMEOUT_SECONDS × STREAM_MAX_RECONNECT_RETRIES`` is exactly
    the stream's OWN budget — one watchdog window per reconnect attempt, up to the
    point where it escalates to ERROR and alerts. Reporting degraded at the same
    instant the stream starts shouting keeps the two signals consistent instead of
    contradicting each other.
    """
    return float(settings.STREAM_HEARTBEAT_TIMEOUT_SECONDS) * float(
        settings.STREAM_MAX_RECONNECT_RETRIES
    )


def _check_price_stream(settings: Settings) -> ComponentHealth:
    """Is the stream task alive, and has it delivered a tick recently?"""
    if not _stream_expected:
        return ComponentHealth(
            status=HEALTH_OK,
            detail="no active instruments — price stream intentionally not started",
        )
    if not price_stream.is_running():
        # The reconnect loop retries indefinitely, so a FINISHED task means the
        # coroutine itself died. Nothing will bring it back without a restart.
        return ComponentHealth(
            status=HEALTH_UNHEALTHY, detail="price stream task is not running"
        )

    last = price_stream.last_tick_at()
    if last is None:
        return ComponentHealth(
            status=HEALTH_DEGRADED, detail="price stream started but no tick received yet"
        )
    age = (datetime.now(timezone.utc) - last).total_seconds()
    budget = _price_staleness_budget_seconds(settings)
    if age > budget:
        return ComponentHealth(
            status=HEALTH_DEGRADED,
            detail=(
                f"newest tick is {age:.0f}s old (> {budget:.0f}s budget) — expected "
                f"outside market hours, investigate during a trading session"
            ),
        )
    return ComponentHealth(status=HEALTH_OK, detail=f"newest tick {age:.0f}s ago")


def _check_ml_model(settings: Settings) -> ComponentHealth:
    """Is the shadow artifact loaded? Cheap — inference caches per process."""
    if not settings.SHADOW_MODE_ENABLED:
        return ComponentHealth(status=HEALTH_OK, detail="SHADOW_MODE_ENABLED=false")
    try:
        loaded = inf.load_model(settings)
        return ComponentHealth(
            status=HEALTH_OK, detail=f"model_id={loaded.model_id} promoted={loaded.promoted}"
        )
    except Exception as exc:  # noqa: BLE001 — shadow enabled + no model = not doing its job
        return ComponentHealth(
            status=HEALTH_UNHEALTHY, detail=f"{type(exc).__name__}: {exc}"
        )


def build_health(settings: Settings) -> tuple[HealthResponse, int]:
    """Assemble the health payload and the HTTP status code that goes with it.

    Split out from the route so it is unit-testable without an HTTP client (and
    without disturbing a running instance).
    """
    database = _check_database()
    stream = _check_price_stream(settings)
    model = _check_ml_model(settings)

    statuses = [database.status, stream.status, model.status]
    if HEALTH_UNHEALTHY in statuses:
        overall, code = HEALTH_UNHEALTHY, status.HTTP_503_SERVICE_UNAVAILABLE
    elif HEALTH_DEGRADED in statuses:
        overall, code = HEALTH_DEGRADED, status.HTTP_200_OK
    else:
        overall, code = HEALTH_OK, status.HTTP_200_OK

    return (
        HealthResponse(
            status=overall,
            checked_at=datetime.now(timezone.utc),
            database=database,
            price_stream=stream,
            ml_model=model,
        ),
        code,
    )


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """Readiness probe used by the container HEALTHCHECK and any uptime monitor.

    A static ``{"status": "ok"}`` proved only that uvicorn was accepting sockets —
    it would have returned 200 for a process with a dead database, a frozen price
    stream and no ML artifact, which is precisely the shape of every silent failure
    this deployment is exposed to.
    """
    payload, code = build_health(get_settings())
    response.status_code = code
    return payload
