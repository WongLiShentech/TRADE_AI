import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s — %(message)s")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.brokers.router import get_broker_router
from app.config import get_settings
from app.database import Base, SessionLocal, engine
import app.models  # noqa: F401  — ensure all models register with Base.metadata
from app.api.v1.router import router as v1_router
from app.models.instrument import Instrument as InstrumentModel
from app.services.price_stream import start_stream, stop_stream
from app.services.scheduler import start_scheduler, stop_scheduler

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _ensure_bot_state_row()
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
        logger.info("price stream started for %d instruments", len(symbols))
    else:
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


app = FastAPI(title="Trading Platform", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
