"""Which strategy is this configuration? — one answer, used everywhere.

Why this exists
---------------
Strategy identity was previously defined inside ``scripts/backfill_attribution.py``.
That was fine while only a backfill needed it, and wrong the moment the LIVE path
did: application code importing from ``scripts/`` is backwards, and a second copy of
the identity rule would eventually disagree with the first — at which point the same
configuration would hash to two different strategies and its evidence would silently
split in half.

Identity is the parameter set, not the label
--------------------------------------------
``params_hash`` is a digest over the engine name plus the parameters that affect
SIGNAL GENERATION. Same configuration ⇒ same id, forever; any difference ⇒ a new
strategy. A rename does not create one; changing ``SIGNAL_MIN_CONFLUENCE_SCORE``
does.

``IDENTITY_PARAMS`` is deliberately narrower than the runner's reproducibility
snapshot. That snapshot records everything needed to reproduce a RUN — fold bounds,
promotion thresholds, the execution window. Those describe how a run was EVALUATED,
not what it TRADED, and folding them in would mint a new "strategy" every time a gate
threshold moved. The distinction is the whole reason the two lists differ.

Auto-registration
-----------------
``resolve_active_strategy`` registers the running configuration if it has never been
seen. That is deliberate, and it is not the same as inventing data: a configuration
the system is actually executing IS a strategy, whether or not anyone remembered to
write it down. The alternative — leaving the rows unattributed — is what produced 103
NULL ``strategy_id`` rows in production, each one unusable for training.

Registration is idempotent by construction (the hash is the key), and it logs at
WARNING so an unintended config change surfaces as a new strategy row rather than
quietly contaminating the previous one's evidence.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.config import Settings
from app.models.strategy import Strategy

logger = logging.getLogger(__name__)

ENGINE = "rule_based"

# The parameters that change WHICH SIGNALS FIRE and WHERE THEY EXIT. Adding a key
# here invalidates every existing params_hash, so it is a deliberate act: it means
# "configurations that differ in this respect were never really the same strategy".
IDENTITY_PARAMS: tuple[str, ...] = (
    "SIGNAL_MIN_CONFLUENCE_SCORE",
    "SIGNAL_SESSION_FILTER",
    "SIGNAL_STOP_ATR_MULTIPLIER",
    "SIGNAL_COOLDOWN_BARS_AFTER_CLOSE",
    "SIGNAL_NO_TRADE_HOURS_BEFORE_FRIDAY_CLOSE",
    "MIN_RR_RATIO",
    "ATR_MULTIPLIER_MAX",
    "SIGNAL_MAX_HOLD_BARS",
    "BACKTEST_TRAILING_LOCK_PCT",
    "BACKTEST_TRAILING_DISTANCE_ATR_MULT",
)


def params_hash(engine: str, params: dict) -> str:
    """Stable 12-char digest over engine + identity params. Same input ⇒ same id.

    ``sort_keys`` and ``default=str`` matter: dict ordering and value types must not
    influence the digest, or the same configuration would hash differently depending
    on how it was assembled.
    """
    payload = json.dumps({"engine": engine, "params": params}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def identity_params(settings: Settings) -> dict:
    """Extract the signal-affecting parameters from a Settings object."""
    return {k: getattr(settings, k) for k in IDENTITY_PARAMS if hasattr(settings, k)}


def resolve_active_strategy(
    db: Session,
    settings: Settings,
    *,
    auto_register: bool = True,
) -> Optional[Strategy]:
    """The strategy the running configuration represents.

    Args:
        db: session. A registration commits; a lookup writes nothing.
        settings: the live configuration.
        auto_register: register the configuration if unseen. ``False`` returns None
            instead — used where a write is not wanted (read-only reporting).

    Returns:
        The :class:`Strategy`, or ``None`` when unseen and ``auto_register`` is False.

    Note:
        Callers on the live path must treat a ``None`` as "record the row anyway,
        unattributed" rather than as a reason to skip trading. Attribution is a
        bookkeeping concern; refusing to record a decision because of it would lose
        the very evidence the row exists to capture.
    """
    params = identity_params(settings)
    missing = [k for k in IDENTITY_PARAMS if k not in params]
    if missing:
        logger.error("strategy identity incomplete — settings lack %s", missing)
        return None

    phash = params_hash(ENGINE, params)
    existing = db.query(Strategy).filter_by(params_hash=phash).first()
    if existing is not None:
        return existing
    if not auto_register:
        return None

    name = f"{ENGINE}_auto_{phash}"
    logger.warning(
        "live configuration matches no registered strategy — registering %s (hash=%s). "
        "If this was not an intended config change, the previous strategy's evidence "
        "has just stopped accumulating.",
        name, phash,
    )
    strategy = Strategy(
        name=name,
        engine=ENGINE,
        params_hash=phash,
        params=params,
        status="shadow",
        description=(
            "Auto-registered from the live configuration. Rename and describe it — the "
            "identity (params_hash) is fixed by the config and will not change."
        ),
        notes="Created automatically by strategy_registry.resolve_active_strategy.",
    )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    return strategy
