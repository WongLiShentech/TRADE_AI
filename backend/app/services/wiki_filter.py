"""
WikiFilter — direction-aware signal blocker.

Scans every wiki page with `status: established` and a `filter_rule:` block,
builds an in-memory list of (instrument, session, direction, action, reason)
rules, exposes is_blocked(instrument, session, direction).

The signal engine (M5) calls is_blocked before emitting any signal. Filter
rebuilds on every call so newly-promoted pages take effect immediately —
no human action required.

Blocking threshold per CLAUDE.md: a page can only carry an active filter_rule
when status == established (which the wiki_promoter only sets at 100+ STRATEGY
trades for that specific direction).
"""
import logging
import re
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)


class FilterRule:
    __slots__ = ("instrument", "session", "direction", "action", "reason")

    def __init__(
        self,
        instrument: str,
        session: str,
        direction: str,
        action: str,
        reason: str,
    ) -> None:
        self.instrument = instrument
        self.session = session
        self.direction = direction
        self.action = action
        self.reason = reason


class WikiFilter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def is_blocked(
        self,
        instrument: str,
        session: str,
        direction: str,
    ) -> tuple[bool, str | None]:
        for rule in self._load_rules():
            if (
                rule.instrument == instrument
                and rule.session == session
                and rule.direction in (direction, "BOTH")
                and rule.action == "block_signals"
            ):
                return True, rule.reason
        return False, None

    def active_rules(self) -> list[FilterRule]:
        return list(self._load_rules())

    def _load_rules(self) -> list[FilterRule]:
        vault = Path(self._settings.VAULT_PATH)
        strategies = vault / "wiki" / "TRADE_AI" / "strategies"
        if not strategies.exists():
            return []
        rules: list[FilterRule] = []
        for md in strategies.glob("*.md"):
            try:
                rules.extend(self._parse_page(md))
            except Exception as exc:
                logger.warning("WikiFilter: skipping %s: %s", md, exc)
        return rules

    @staticmethod
    def _parse_page(md_path: Path) -> list[FilterRule]:
        text = md_path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return []
        end = text.find("\n---", 3)
        if end == -1:
            return []
        fm = text[3:end]

        # Extract simple top-level key:value pairs.
        flat: dict[str, str] = {}
        for line in fm.splitlines():
            m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
            if m and not line.startswith(" "):
                flat[m.group(1)] = m.group(2).strip()

        if flat.get("status") != "established":
            return []

        # filter_rule is a nested block — parse its indented children.
        if "filter_rule:" not in fm:
            return []
        rule_block = re.search(
            r"filter_rule:\s*\n((?:[ \t]+[A-Za-z0-9_]+\s*:.*\n?)+)", fm
        )
        if not rule_block:
            return []
        nested: dict[str, str] = {}
        for line in rule_block.group(1).splitlines():
            m = re.match(r"^[ \t]+([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
            if m:
                nested[m.group(1)] = m.group(2).strip().strip('"').strip("'")

        instrument = flat.get("instrument", "")
        session = flat.get("session", "")
        direction = nested.get("direction") or flat.get("direction", "")
        action = nested.get("action", "")
        reason = nested.get("reason", "")

        if not (instrument and session and direction and action):
            return []
        return [FilterRule(instrument, session, direction, action, reason)]
