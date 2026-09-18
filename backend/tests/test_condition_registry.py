"""The condition registry is the single source of truth — these assert it stays one.

The defect this replaces: the condition set was written out twice, in the engine's inline
scoring and in ``feature_builder._CONFLUENCE_TO_PAYLOAD``, with nothing forcing them to
agree. The feature builder iterated its OWN dict rather than the breakdown handed to it,
so a sixth condition would have steered trades while being silently absent from every
feature row — and no test would have failed.
"""
from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.domain.conditions import (
    CONDITIONS,
    CONDITIONS_BY_KEY,
    LEGACY_CONFLUENCE_KEYS,
    Role,
    enabled_conditions,
    gates,
    votes,
)
from app.services.feature_builder import PAYLOAD_KEYS


def test_keys_and_payload_keys_are_unique():
    keys = [c.key for c in CONDITIONS]
    payloads = [c.payload_key for c in CONDITIONS]
    assert len(set(keys)) == len(keys), keys
    assert len(set(payloads)) == len(payloads), payloads


def test_every_declared_settings_key_exists():
    """A typo in ``settings_keys`` reads as 'this condition has no parameters'.

    That is not cosmetic: the list is what strategy identity must cover, so a dropped key
    means two materially different configurations hash to the same strategy.
    """
    fields = set(Settings.model_fields)
    for cond in CONDITIONS:
        unknown = [k for k in cond.settings_keys if k not in fields]
        assert not unknown, f"condition '{cond.key}' declares unknown Settings keys: {unknown}"


def test_every_condition_reaches_the_payload_tier():
    """The anti-`_CONFLUENCE_TO_PAYLOAD` test.

    Every registered condition must have its payload key in the feature contract. If one
    does not, it influences trades while being invisible to every model trained on those
    rows — which is exactly the failure the hand-copied dict made possible.
    """
    missing = [c.payload_key for c in CONDITIONS if c.payload_key not in PAYLOAD_KEYS]
    assert not missing, f"conditions whose payload key is absent from the contract: {missing}"


def test_confluence_score_is_pinned_to_exactly_the_legacy_five():
    """``confluence_score`` is a MODEL-TIER feature and must never change meaning.

    Adding a sixth condition must not move it. If it did, every corpus trained before the
    addition would describe a different quantity under an unchanged name — and nothing
    downstream compares feature values, only names, so nothing would catch it.
    """
    assert LEGACY_CONFLUENCE_KEYS == ("trend", "rsi", "structure", "session", "spread")
    assert len(LEGACY_CONFLUENCE_KEYS) == 5
    assert all(CONDITIONS_BY_KEY[k].legacy for k in LEGACY_CONFLUENCE_KEYS)


def test_roles_are_declared_and_partition_the_enabled_set():
    settings = get_settings()
    enabled = enabled_conditions(settings)
    assert set(votes(settings)) | set(gates(settings)) == set(enabled)
    assert not set(votes(settings)) & set(gates(settings))
    assert {c.key for c in gates(settings)} == {"session", "spread"}, (
        "session and spread are the direction-independent pair; if that changed, the "
        "ambiguity analysis behind Stage 2 needs revisiting"
    )


def test_unknown_condition_names_are_refused():
    """Silently ignoring an unknown name would run a strategy nobody configured."""
    settings = get_settings().model_copy(update={"SIGNAL_CONDITIONS": "trend,not_a_condition"})
    with pytest.raises(RuntimeError, match="unregistered"):
        enabled_conditions(settings)


def test_an_empty_condition_list_is_refused():
    """A rule with no conditions fires on every bar — never a silent default."""
    settings = get_settings().model_copy(update={"SIGNAL_CONDITIONS": "  "})
    with pytest.raises(RuntimeError, match="empty"):
        enabled_conditions(settings)


def test_enabled_conditions_follow_registry_order_not_config_order():
    """The same set must score identically however it was spelled."""
    settings = get_settings()
    forward = enabled_conditions(settings.model_copy(update={"SIGNAL_CONDITIONS": "trend,rsi"}))
    reverse = enabled_conditions(settings.model_copy(update={"SIGNAL_CONDITIONS": "rsi,trend"}))
    assert [c.key for c in forward] == [c.key for c in reverse] == ["trend", "rsi"]


def test_feature_builder_raises_on_an_unregistered_condition(db, settings, instrument):
    """A breakdown key the registry does not know means engine and contract have diverged.

    Loudly, because the quiet version is a condition steering trades that no model can see.
    """
    from datetime import datetime

    from app.services.feature_builder import build_features

    inst = instrument("EUR_USD")
    breakdown = {c.key: True for c in CONDITIONS}
    breakdown["invented_condition"] = True
    with pytest.raises(RuntimeError, match="unregistered"):
        build_features(inst, datetime(2025, 3, 5, 12, 0, 1), "H4", breakdown, db, settings)


def test_gate_callables_ignore_direction():
    """A GATE's answer may not depend on direction — that is what the role asserts.

    Checked structurally here (the signature accepts it and must not branch on it);
    ``test_rule_based_engine`` checks it behaviourally against a real context.
    """
    for cond in CONDITIONS:
        if cond.role is Role.GATE:
            assert cond.evaluate.__code__.co_argcount == 2, (
                f"gate '{cond.key}' does not take (ctx, direction)"
            )
