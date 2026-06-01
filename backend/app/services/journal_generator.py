"""
Journal generator — produces a weekly markdown summary of every trade and
writes it to settings.VAULT_PATH/raw/TRADE_AI/journal/{ISO_year}-W{week}.md.

This file is the primary input the wiki ingester reads on Sunday.
Plain-language formatting (Part 4f) — no jargon in body.

Scheduled: Sunday 00:00 UTC by services/scheduler.py.
Manual trigger: POST /api/v1/journal/generate.
"""
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import Settings
from app.models.trade import Trade
from app.models.instrument import Instrument

logger = logging.getLogger(__name__)


def generate_weekly_journal(
    db: Session,
    settings: Settings,
    week_offset: int = 0,
) -> dict:
    """
    Generate the journal for the ISO week ending closest to (today - week_offset weeks).
    week_offset=0 → most recently completed week. Returns {path, total, ...}.
    """
    today = datetime.now(timezone.utc).date()
    target = today - timedelta(weeks=week_offset)
    iso_year, iso_week, _ = target.isocalendar()
    week_label = f"{iso_year}-W{iso_week:02d}"

    week_start = date.fromisocalendar(iso_year, iso_week, 1)
    week_end = week_start + timedelta(days=7)

    trades = (
        db.query(Trade)
        .filter(
            Trade.opened_at >= datetime.combine(week_start, datetime.min.time()),
            Trade.opened_at < datetime.combine(week_end, datetime.min.time()),
        )
        .order_by(Trade.opened_at)
        .all()
    )

    content = _format_journal(week_label, week_start, week_end, trades, db)

    vault = Path(settings.VAULT_PATH)
    journal_dir = vault / "raw" / "TRADE_AI" / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    out_path = journal_dir / f"{week_label}.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("journal written: %s (%d trades)", out_path, len(trades))

    return {
        "path": str(out_path),
        "week_label": week_label,
        "total": len(trades),
    }


def _format_journal(
    week_label: str,
    start: date,
    end: date,
    trades: list[Trade],
    db: Session,
) -> str:
    counts = {"STRATEGY": 0, "NEWS": 0, "MANIPULATION": 0, "MANUAL": 0, "UNCERTAIN": 0}
    strategy_wins = 0
    strategy_total = 0
    net_pnl = 0.0
    for t in trades:
        cls = t.final_classification or t.auto_classification or "UNCERTAIN"
        counts[cls] = counts.get(cls, 0) + 1
        if cls == "STRATEGY":
            strategy_total += 1
            if t.outcome == "win":
                strategy_wins += 1
        if t.rr_actual is not None and t.risk_amount is not None:
            net_pnl += t.rr_actual * t.risk_amount

    win_rate = (strategy_wins / strategy_total * 100) if strategy_total else 0.0

    lines: list[str] = [
        f"# Weekly trade journal — {week_label}",
        "",
        f"_Week of {start.isoformat()} → {(end - timedelta(days=1)).isoformat()}_",
        "",
        "## Summary",
        f"- Total trades: **{len(trades)}**",
        f"- Counted toward strategy: **{counts['STRATEGY']}**",
        f"- Excluded — news event: **{counts['NEWS']}**",
        f"- Excluded — abnormal market: **{counts['MANIPULATION']}**",
        f"- Excluded — closed by hand: **{counts['MANUAL']}**",
        f"- Excluded — could not verify context: **{counts['UNCERTAIN']}**",
        f"- Win rate (strategy trades only): **{win_rate:.1f}%**",
        f"- Net profit / loss: **{net_pnl:+.2f} USD**",
        "",
        "## Trades",
        "",
    ]

    if not trades:
        lines.append("_No trades this week._")
        return "\n".join(lines) + "\n"

    instruments_by_id = {i.id: i.symbol for i in db.query(Instrument).all()}
    for t in trades:
        lines.extend(_format_trade(t, instruments_by_id))
        lines.append("")

    return "\n".join(lines) + "\n"


def _format_trade(t: Trade, instruments_by_id: dict[int, str]) -> list[str]:
    symbol = instruments_by_id.get(t.instrument_id, f"#{t.instrument_id}")
    cls = t.final_classification or t.auto_classification or "UNCERTAIN"
    outcome_label = {
        "win": "won",
        "loss": "lost",
        "breakeven": "broke even",
    }.get(t.outcome or "", "still open")
    return [
        f"### {symbol} {t.direction} — {outcome_label}",
        f"- Opened: {t.opened_at.isoformat() if t.opened_at else 'unknown'}",
        f"- Closed: {t.closed_at.isoformat() if t.closed_at else 'still open'}",
        f"- Session at entry: {t.session or 'unknown'}",
        f"- Why this trade was taken: {_describe_reasoning(t)}",
        f"- Stop method used: {t.stop_method or 'unknown'}",
        f"- Trade classification: **{cls}** "
        f"(confidence: {t.classification_confidence:.0%})"
        if t.classification_confidence is not None
        else f"- Trade classification: **{cls}** (confidence: unknown)",
        f"- Exit reason: {t.exit_reason or 'still open'}",
        f"- News context at entry: "
        f"{len(t.news_events) if t.news_events else 0} relevant events found",
    ]


def _describe_reasoning(t: Trade) -> str:
    if not t.signal_reasoning:
        return "No reasoning recorded."
    parts = []
    score = t.confluence_score
    if score is not None:
        parts.append(f"{score} of 10 conditions aligned")
    if t.signal_source:
        parts.append(f"signal source: {t.signal_source}")
    return "; ".join(parts) or "No reasoning recorded."
