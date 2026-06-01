from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.services.wiki_filter import WikiFilter
from app.services.wiki_ingestor import ingest_journal
from app.services.wiki_promoter import promote_pages

router = APIRouter()


@router.post("/ingest")
def trigger_ingest(
    week_offset: int = 0,
    settings: Settings = Depends(get_settings),
):
    """Manual server-side wiki ingestion of the most recent (or specified) week."""
    today = datetime.now(timezone.utc).date()
    target = today - timedelta(weeks=week_offset)
    iso_year, iso_week, _ = target.isocalendar()
    week_label = f"{iso_year}-W{iso_week:02d}"
    try:
        return ingest_journal(week_label, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/promote")
def trigger_promote(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Manual run of the wiki promoter (auto-status update by sample size)."""
    try:
        return promote_pages(db, settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/rebuild-filter")
def trigger_rebuild_filter(settings: Settings = Depends(get_settings)):
    """Force a WikiFilter rebuild and return the active rule count."""
    wf = WikiFilter(settings)
    rules = wf.active_rules()
    return {
        "active_rules": len(rules),
        "rules": [
            {
                "instrument": r.instrument,
                "session": r.session,
                "direction": r.direction,
                "action": r.action,
                "reason": r.reason,
            }
            for r in rules
        ],
    }
