from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.schemas.journal import JournalGenerateResponse
from app.services.journal_generator import generate_weekly_journal

router = APIRouter()


@router.post("/generate", response_model=JournalGenerateResponse)
def trigger_journal(
    week_offset: int = 0,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """
    Manual trigger for the weekly journal. week_offset=0 → most recently
    completed week. week_offset=1 → the week before that, etc.
    """
    try:
        result = generate_weekly_journal(db, settings, week_offset=week_offset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result
