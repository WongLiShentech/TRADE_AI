"""
Server-side wiki ingester. Calls the Anthropic Python SDK to translate a
weekly journal into wiki page updates.

Reads:
- {VAULT_PATH}/raw/TRADE_AI/journal/{week}.md (the journal just written)
- {VAULT_PATH}/CLAUDE.md (wiki rules)
- {VAULT_PATH}/wiki/TRADE_AI/**/*.md (existing pages, filtered to those
  relevant to instruments traded that week)

Writes:
- New / updated wiki pages per Claude's returned JSON ops
- Append entry to {VAULT_PATH}/wiki/TRADE_AI/log.md

Failure mode: any exception is logged and a `failure` entry is appended to
wiki/TRADE_AI/log.md. Trading is unaffected.

This service is wired but not exercised. The Sunday scheduler tick or a
manual POST /api/v1/wiki/ingest invokes it.
"""
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.services.wiki_ingestor_prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

VALID_ACTIONS = {"create", "update", "append"}


def ingest_journal(week_label: str, settings: Settings) -> dict:
    """
    Returns a summary dict. Raises ValueError on configuration problems
    (missing API key) so the caller can surface a clean error to the user.
    """
    if not settings.WIKI_INGEST_ENABLED:
        return {"ingested": False, "reason": "WIKI_INGEST_ENABLED is false"}

    if not settings.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is empty — cannot run server-side wiki ingestion. "
            "Set it in backend/.env or disable WIKI_INGEST_ENABLED."
        )

    vault = Path(settings.VAULT_PATH)
    journal_path = vault / "raw" / "TRADE_AI" / "journal" / f"{week_label}.md"
    if not journal_path.exists():
        raise ValueError(f"journal file not found: {journal_path}")

    journal_md = journal_path.read_text(encoding="utf-8")
    existing_pages = _load_existing_pages(vault)
    today_iso = datetime.now(timezone.utc).date().isoformat()
    user_prompt = build_user_prompt(journal_md, existing_pages, today_iso)

    try:
        ops = _call_anthropic(settings, user_prompt)
    except Exception as exc:
        logger.exception("wiki ingestor failed")
        _append_log(
            vault,
            f"## [{today_iso}] failure | wiki ingest for {week_label} failed: {exc}",
        )
        return {"ingested": False, "reason": str(exc)}

    applied = _apply_ops(vault, ops.get("operations", []))
    summary = ops.get("log_summary", f"Ingested journal {week_label}")
    _append_log(vault, f"## [{today_iso}] ingest | {summary} ({applied} ops applied)")

    return {
        "ingested": True,
        "week_label": week_label,
        "ops_applied": applied,
    }


def _load_existing_pages(vault: Path) -> dict[str, str]:
    """Load all .md pages under wiki/TRADE_AI as {relative_path: content}."""
    base = vault / "wiki" / "TRADE_AI"
    pages: dict[str, str] = {}
    if not base.exists():
        return pages
    for md in base.rglob("*.md"):
        rel = md.relative_to(vault).as_posix()
        try:
            pages[rel] = md.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("could not read %s: %s", md, exc)
    return pages


def _call_anthropic(settings: Settings, user_prompt: str) -> dict:
    """Calls Anthropic with prompt caching on the system prompt."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=settings.WIKI_INGEST_MODEL,
        max_tokens=8000,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    return _extract_json(text)


def _extract_json(text: str) -> dict:
    # Try direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences if present.
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Last resort: find the first { and last } and try.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("could not parse JSON from Anthropic response")


def _apply_ops(vault: Path, operations: list[dict]) -> int:
    """Apply each op safely (path traversal guard) and return count applied."""
    applied = 0
    for op in operations:
        path = op.get("path", "")
        action = op.get("action", "")
        content = op.get("content", "")
        if action not in VALID_ACTIONS or not path:
            logger.warning("skipping invalid op: %s", op)
            continue
        # Path traversal guard — must resolve under vault root.
        target = (vault / path).resolve()
        try:
            target.relative_to(vault.resolve())
        except ValueError:
            logger.warning("rejecting op outside vault: %s", path)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if action in {"create", "update"}:
            target.write_text(content, encoding="utf-8")
        elif action == "append":
            with target.open("a", encoding="utf-8") as f:
                f.write("\n" + content + "\n")
        applied += 1
    return applied


def _append_log(vault: Path, line: str) -> None:
    log_path = vault / "wiki" / "TRADE_AI" / "log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + line + "\n")
