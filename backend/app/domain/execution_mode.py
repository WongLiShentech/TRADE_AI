"""Execution modes — which venue, if any, a strategy may trade.

Lives in ``domain`` rather than ``brokers`` on purpose. The shadow package carries
a static guarantee, enforced by ``tests/test_shadow_recorder.py``, that it imports
no broker module and mentions no order call — so that observation can never reach a
broker even by accident. But the recorder must still know the mode, because the
mode decides whether a row is labelled ``shadow`` (no order) or ``sandbox`` (an
order follows).

Putting the constants here lets both sides depend on the concept without the
observation path depending on the execution path. ``brokers.base`` re-exports them
for callers that already import from there.

Three states, not a boolean
---------------------------
"No orders", "practice money" and "real money" are genuinely different, and a
boolean can only express two. Collapsing them is exactly how a sandbox flag turns
into a live order.
"""

EXECUTION_MODE_OBSERVE = "observe"   # record decisions only — no order, any broker
EXECUTION_MODE_SANDBOX = "sandbox"   # real order tickets, PRACTICE account only
EXECUTION_MODE_LIVE = "live"         # real money — NOT IMPLEMENTED, blocked in base

EXECUTION_MODES = frozenset(
    {EXECUTION_MODE_OBSERVE, EXECUTION_MODE_SANDBOX, EXECUTION_MODE_LIVE}
)

# Practice and live are distinguished ONLY by the OANDA host — by project rule no
# account-type setting exists. This marker is what makes "sandbox" verifiable
# rather than merely asserted.
PRACTICE_URL_MARKER = "fxpractice"


def normalise(value: object) -> str:
    """Canonical mode string from raw config.

    Case and surrounding whitespace are normalised because a stray capital or space
    in a ``.env`` is a typo, not a different intent, and rejecting it would be an
    availability problem with no safety benefit. A value that means something ELSE
    ('paper', 'true', '') survives normalisation unchanged and is then rejected by
    the caller's membership check — which is where fail-closed belongs.
    """
    return str(value or "").strip().lower()
