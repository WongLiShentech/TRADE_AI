from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.bot_state import BotState

router = APIRouter()


def _get_or_create_state(db: Session) -> BotState:
    state = db.query(BotState).filter_by(id=1).first()
    if state is None:
        state = BotState(id=1, paused=False)
        db.add(state)
        db.commit()
        db.refresh(state)
    return state


@router.get("/state")
def get_state(db: Session = Depends(get_db)):
    state = _get_or_create_state(db)
    return {
        "paused": state.paused,
        "paused_reason": state.paused_reason,
        "paused_at": state.paused_at.isoformat() if state.paused_at else None,
        "last_resumed_at": state.last_resumed_at.isoformat()
        if state.last_resumed_at
        else None,
    }


@router.post("/resume")
def resume(db: Session = Depends(get_db)):
    """Manually clear the circuit breaker. There is no auto-resume."""
    state = _get_or_create_state(db)
    state.paused = False
    state.paused_reason = None
    state.paused_at = None
    state.last_resumed_at = datetime.now(timezone.utc)
    db.commit()
    return {"paused": False, "last_resumed_at": state.last_resumed_at.isoformat()}
