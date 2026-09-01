from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
    # pool_pre_ping: issue a cheap SELECT 1 before handing out a pooled connection.
    # Without it, a Postgres restart (container update, OOM kill, host reboot) leaves
    # every pooled socket dead, and the next scheduled job — which may be hours later
    # and unattended — dies on OperationalError instead of silently reconnecting.
    # pool_recycle: proactively drop connections older than 30 minutes, so an idle
    # connection is never reaped mid-use by a proxy/firewall idle timeout. Both are
    # no-ops for SQLite.
    pool_pre_ping=True,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
