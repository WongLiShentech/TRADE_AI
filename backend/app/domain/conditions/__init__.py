"""The signal-condition registry — one source of truth for every consumer.

Two consumers derive from this table: the signal engine (which conditions to evaluate and
how to combine them) and ``feature_builder`` (which payload keys exist, and what
``confluence_score`` counts). Neither restates the set, so they cannot disagree.

Validated at import, loudly, for the same reason ``feature_builder`` validates its own
contract at import: a registry that is wrong is worse than no registry, because every
consumer now trusts it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.domain.conditions.builtin import BUILTIN
from app.domain.conditions.spec import Condition, ConditionContext, Role

if TYPE_CHECKING:  # pragma: no cover
    from app.config import Settings

__all__ = [
    "Condition", "ConditionContext", "Role",
    "CONDITIONS", "CONDITIONS_BY_KEY", "LEGACY_CONFLUENCE_KEYS",
    "enabled_conditions", "votes", "gates",
]

CONDITIONS: tuple[Condition, ...] = BUILTIN
CONDITIONS_BY_KEY: dict[str, Condition] = {c.key: c for c in CONDITIONS}

# The five that `confluence_score` counts, FOREVER, in declaration order.
#
# `confluence_score` is a model-tier feature. If adding a sixth condition changed what it
# counts, every corpus trained before the addition would describe a different quantity
# under the same name — and nothing downstream compares feature VALUES, only names, so
# nothing would catch it. New conditions land in the payload tier and are promoted
# deliberately, at a schema bump, with evidence.
LEGACY_CONFLUENCE_KEYS: tuple[str, ...] = tuple(c.key for c in CONDITIONS if c.legacy)


def _validate_registry() -> None:
    """Fail at import on a malformed registry, never at the first live signal."""
    from app.config import Settings

    keys = [c.key for c in CONDITIONS]
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"duplicate condition keys in the registry: {keys}")

    payloads = [c.payload_key for c in CONDITIONS]
    if len(set(payloads)) != len(payloads):
        raise RuntimeError(f"duplicate payload keys in the registry: {payloads}")

    fields = set(Settings.model_fields)
    for c in CONDITIONS:
        unknown = [k for k in c.settings_keys if k not in fields]
        if unknown:
            raise RuntimeError(
                f"condition '{c.key}' declares Settings keys that do not exist: {unknown}. "
                f"A typo here reads as 'this condition has no parameters', which silently "
                f"drops it from strategy identity."
            )

    if len(LEGACY_CONFLUENCE_KEYS) != 5:
        raise RuntimeError(
            f"confluence_score counts {len(LEGACY_CONFLUENCE_KEYS)} conditions, not 5 "
            f"({list(LEGACY_CONFLUENCE_KEYS)}). It is a MODEL-TIER feature: changing what "
            f"it counts changes the meaning of a name every existing corpus already uses. "
            f"Add new conditions with legacy=False."
        )


_validate_registry()


def enabled_conditions(settings: "Settings") -> tuple[Condition, ...]:
    """The conditions named in ``SIGNAL_CONDITIONS``, in registry order.

    Registry order rather than config order, so the same set always scores identically
    regardless of how it was spelled.

    Raises:
        RuntimeError: an unknown name. Silently ignoring one would run a strategy that is
            not the strategy anybody configured.
    """
    raw = (getattr(settings, "SIGNAL_CONDITIONS", "") or "").strip()
    if not raw:
        raise RuntimeError(
            "SIGNAL_CONDITIONS is empty — a rule with no conditions would fire on every "
            "bar. Name the conditions explicitly."
        )
    wanted = {name.strip().lower() for name in raw.split(",") if name.strip()}
    unknown = sorted(wanted - set(CONDITIONS_BY_KEY))
    if unknown:
        raise RuntimeError(
            f"SIGNAL_CONDITIONS names unregistered conditions {unknown}. "
            f"Available: {sorted(CONDITIONS_BY_KEY)}"
        )
    return tuple(c for c in CONDITIONS if c.key in wanted)


def votes(settings: "Settings") -> tuple[Condition, ...]:
    """Enabled directional conditions — the ones that contribute to the score."""
    return tuple(c for c in enabled_conditions(settings) if c.role is Role.VOTE)


def gates(settings: "Settings") -> tuple[Condition, ...]:
    """Enabled direction-independent conditions — hard vetoes once Stage 2 lands."""
    return tuple(c for c in enabled_conditions(settings) if c.role is Role.GATE)
