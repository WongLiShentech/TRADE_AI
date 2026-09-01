"""M7 backtester package.

Part A ships two pure, side-effect-free building blocks that the Part B runner
orchestrates:

* ``simulator.simulate`` — the triple-barrier / Option-B trailing-stop exit engine
  (one fired signal → one :class:`~app.services.backtester.simulator.ExitResult`).
* ``metrics`` — profit factor, expectancy, drawdown, (probabilistic / deflated)
  Sharpe, outcome breakdown and the promotion gate over a list of trade records.

Neither module writes to the database; the runner persists ``Trade`` /
``BacktestRun`` rows.
"""
from app.services.backtester.metrics import (
    avg_holding_hours,
    deflated_sharpe,
    evaluate_promotion_gate,
    expectancy,
    max_drawdown,
    outcome_breakdown,
    probabilistic_sharpe,
    profit_factor,
    sharpe,
    win_rate,
)
from app.services.backtester.simulator import BidAskBar, ExitResult, simulate

__all__ = [
    "BidAskBar",
    "ExitResult",
    "simulate",
    "profit_factor",
    "expectancy",
    "max_drawdown",
    "sharpe",
    "deflated_sharpe",
    "probabilistic_sharpe",
    "outcome_breakdown",
    "avg_holding_hours",
    "win_rate",
    "evaluate_promotion_gate",
]
