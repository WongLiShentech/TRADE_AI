"""
Wiki promoter — auto-updates frontmatter `status` and `sample_size` on
strategy pages based on actual STRATEGY trade counts in the database.

Reads each `wiki/TRADE_AI/strategies/*.md`, parses its frontmatter
(instrument, session, direction), counts STRATEGY trades in the DB matching
that exact tuple where classification_confidence >= MIN_ML_TRAINING_CONFIDENCE,
and rewrites the frontmatter with the updated count + threshold-mapped
status.

Thresholds (from CLAUDE.md):
  < 10:   preliminary
  10-24:  tentative
  25-49:  confirmed (early)
  50-99:  confirmed
  >= 100: established

NEVER combines BUY+SELL. Each page tracks one direction only.

Scheduled: Sunday 00:00 UTC after the wiki ingester (services/scheduler.py).
Manual trigger: POST /api/v1/wiki/promote.
"""
import logging
import re
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.instrument import Instrument
from app.models.trade import Trade

logger = logging.getLogger(__name__)


def promote_pages(db: Session, settings: Settings) -> dict:
    vault = Path(settings.VAULT_PATH)
    strategies_dir = vault / "wiki" / "TRADE_AI" / "strategies"
    if not strategies_dir.exists():
        return {"promoted": 0, "scanned": 0}

    scanned = 0
    promoted = 0
    for md in strategies_dir.glob("*.md"):
        scanned += 1
        try:
            if _promote_page(md, db, settings):
                promoted += 1
        except Exception as exc:
            logger.warning("promoter skipped %s: %s", md, exc)

    return {"promoted": promoted, "scanned": scanned}


def _promote_page(md_path: Path, db: Session, settings: Settings) -> bool:
    text = md_path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)
    if fm is None:
        return False

    fm_dict = _parse_frontmatter(fm)
    instrument = fm_dict.get("instrument")
    session = fm_dict.get("session")
    direction = fm_dict.get("direction")
    if not (instrument and session and direction):
        return False

    inst_row = db.query(Instrument).filter_by(symbol=instrument).first()
    if inst_row is None:
        return False

    # Count STRATEGY trades for THIS direction only — never combined.
    count = (
        db.query(func.count(Trade.id))
        .filter(
            Trade.instrument_id == inst_row.id,
            Trade.session == session,
            Trade.direction == direction,
            Trade.final_classification == "STRATEGY",
            Trade.classification_confidence >= settings.MIN_ML_TRAINING_CONFIDENCE,
        )
        .scalar()
    ) or 0

    new_status = _status_for(count)
    fm_dict["sample_size"] = str(count)
    fm_dict["status"] = new_status

    new_fm = _serialise_frontmatter(fm_dict)
    md_path.write_text(f"---\n{new_fm}---\n{body}", encoding="utf-8")
    return True


def _status_for(count: int) -> str:
    if count >= 100:
        return "established"
    if count >= 25:
        return "confirmed"
    if count >= 10:
        return "tentative"
    return "preliminary"


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    if not text.startswith("---"):
        return None, text
    end = text.find("\n---", 3)
    if end == -1:
        return None, text
    fm = text[3:end].lstrip("\n")
    body = text[end + 4 :].lstrip("\n")
    return fm, body


_SIMPLE_KV = re.compile(r"^([A-Za-z0-9_]+)\s*:\s*(.*)$")


def _parse_frontmatter(fm: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in fm.splitlines():
        m = _SIMPLE_KV.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _serialise_frontmatter(d: dict[str, str]) -> str:
    # Preserve insertion order — Python dicts do this since 3.7.
    return "".join(f"{k}: {v}\n" for k, v in d.items())
