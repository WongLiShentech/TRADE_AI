# TRADE_AI — Interview Brief

*Last verified: 2026-09-04. Every number below is queried from the live system, not estimated.*

---

## The 30-second version

An algorithmic FX trading platform I built end to end: it ingests market and macroeconomic
data, generates trade signals from a rule-based engine, filters them with an XGBoost
classifier, and records every decision for evaluation. It runs 24/7 on AWS.

**It has never placed a real trade — deliberately.** The model beat its baseline on every
metric and still missed one promotion criterion, so I declined to deploy it and built a
shadow-execution mode to gather forward evidence instead. That decision is the part of the
project I'd most want to talk about.

---

## The problem it solves

Most retail trading systems fail for one of three reasons, and the platform is designed
around preventing each:

| Failure | How it happens | What I built against it |
|---|---|---|
| **Lookahead bias** | Using data that wasn't knowable at decision time | Point-in-time feature pipeline over bitemporal macro vintages |
| **Training/serving skew** | Features computed differently in backtest vs live | One feature builder serving both paths |
| **Overfitting to one dataset** | Iterating until the backtest looks good | Walk-forward validation + Deflated Sharpe correction |

---

## Architecture

```
OANDA ─┬─ live tick stream (10 FX pairs, 24/5)
       └─ candles: D1, H4, M1 bid/ask

FRED  ─── macro releases, stored bitemporally
             (reference period AND publication time)
                        │
                        ▼
              ┌─── FEATURE BUILDER ───┐      one chokepoint, used by
              │  19 point-in-time     │      BOTH backtest and live —
              │  features             │      this is what prevents skew
              └───────────┬───────────┘
                          ▼
        GENERATOR ── 5-condition rule engine ──► candidate trades
                          │
                          ▼
        FILTER ────── XGBoost classifier ──────► take / skip
                          │
                          ▼
        RISK ──────── position sizing, correlation limits, hard ceilings
                          │
                          ▼
        EXECUTION ─── observe | sandbox | live   (currently: observe)
```

**Stack:** Python, FastAPI, PostgreSQL, SQLAlchemy/Alembic, Docker, APScheduler, XGBoost,
pandas, SHAP. Deployed on AWS EC2 via Docker Compose.

---

## How the pipeline works

**Every 4 hours**, at each H4 candle close:

1. **Generate** — each of 10 pairs is scored against five conditions: daily trend, H4 RSI
   band, proximity to swing structure, trading session, and current spread. Three of five
   must pass.
2. **Size** — the risk engine derives position size from `risk% ÷ stop distance`. Size is
   never an input; it is always the output of a locked triangle with risk and stop distance.
   It also rejects trades whose reward:risk is under 2:1 or that correlate with an open
   position.
3. **Filter** — the model scores the signal from 19 features and returns P(the trade reaches
   +1R). Above a threshold, it says *take*.
4. **Record** — every decision is written, including skips. Skips are 92% of the output and
   are the majority of the evidence.

**Every hour**, an outcome resolver replays minute-by-minute bid/ask data to determine what
each recorded trade *would* have done — stop hit, target hit, or time exit at 10 bars.

**Skips are resolved too.** That is the point: live trading can never tell you what the
trades you declined would have done, and declining is most of what the model does.

---

## Results — stated honestly

**Validation design:** 3-fold expanding walk-forward, 14-bar embargo between train and test.
Each fold trains a fresh model on its own window and is scored on later, unseen data.

**Corpus:** 7,074 labelled trades, 10 pairs, 2021-06 → 2026-06, resolved against 36.8M
intrabar candles.

| | Trades | Win rate | Profit factor | Expectancy | Max drawdown | Verdict |
|---|---|---|---|---|---|---|
| Rules only | 4,692 OOS | 57.2% | 1.408 | +0.152R | **29.0%** | ❌ failed |
| Rules + ML filter | 1,772 OOS | 59.4% | **1.586** | **+0.207R** | **19.3%** | ❌ failed |

**Both failed the promotion gate.**

- The rule baseline breached the 25% drawdown ceiling.
- The ML filter improved every metric — and missed fold-1 expectancy by **0.003**
  (0.1473 vs a 0.1500 floor).

Per fold, the filter's behaviour is not uniform:

| Fold | Rules PF | +ML PF | |
|---|---|---|---|
| 1 (2023-02 → 2024-10) | 1.430 | 1.381 | filter made it **worse** |
| 2 (2024-10 → 2025-10) | 1.438 | **2.927** | large improvement |
| 3 (2025-10 → 2026-06) | 1.297 | **1.975** | rescued a failing fold |

**I treat this as unresolved rather than as evidence of recent-regime skill.** Folds 2 and 3
survive a permutation test against random selection, but the model's own inputs include
volatility measures, so "it read the regime" and "it tracked volatility" are not separable
from backtest data alone.

**Multiple-testing correction:** 7 research iterations were run against this dataset. The
count is stored with every run and feeds a Deflated Sharpe Ratio, so results are discounted
by how many times the data has been examined.

---

## Current status (2026-09-04)

```
Deployed        AWS EC2, Singapore, 24/7, containerised
Mode            observe — records decisions, places no orders
Shadow rows     80 recorded · 40 cleanly resolved · 0 resolved takes
Model           s1_xgb_v2, tagged NOT_PROMOTED, running for observation only
Tests           275 passing
```

**The model has taken 5 signals in a month and none has cleanly resolved.** So its ability to
*select* trades is currently unmeasured. What the 40 resolved skips show — 29 losses avoided,
11 winners missed, of which only 2 reached +1R — is consistent with a working filter and
equally consistent with a model that is simply very reluctant. **Both readings fit, and I say
so rather than pick the flattering one.**

---

## Design decisions I would defend in an interview

**Point-in-time macro data.** FRED releases are stored with both a reference period and a
publication time. Features gate on *publication* time. Using the reference period would mean
a model in 2023 could see a CPI figure published weeks later — the most common and least
visible form of lookahead in macro-conditioned models.

**One feature builder for backtest and live.** The same code path computes features in both.
Two implementations inevitably diverge, and the divergence shows up as a model that
backtested well and behaves differently in production.

**Fail-closed execution guard.** Order placement is a concrete template method on the broker
base class, not an abstract one. A new broker inherits the guard whether or not its author
knows the flag exists, and a subclass that skips the base constructor refuses every order.
Sandbox mode is additionally cross-checked against the broker URL, because practice and live
accounts differ only by hostname — a mode labelled "sandbox" pointed at a live endpoint would
be real money wearing the wrong label.

**Skips are recorded and resolved.** A filter that avoids every loss *and* every win has no
value. You cannot distinguish protective from timid without grading the declined trades, and
live trading never gives you that.

**Excursion recording.** Each trade stores its maximum favourable and adverse excursion, not
just its result. Two trades that both lose 1R — one that reached +0.9R first and one that
never moved — are different problems with identical rows, and need opposite fixes.

---

## What is deliberately not done

| | Why |
|---|---|
| No live trading | The model failed its gate. The gate is the point. |
| Order path built but disabled | Verified against a practice account; switched off until a model qualifies |
| One strategy only | Multi-strategy schema exists; execution wiring is not built |
| Financing costs not modelled | Positions average ~28h, so nearly every trade crosses an overnight rollover. Backtest results are ~10% optimistic and I say so. |

---

## Questions I'd expect, and my answers

**"Is it profitable?"**
On paper the backtest is, at a 29% drawdown that breaches my own risk limit. That's why it
isn't trading. I'd rather report a system that failed its criteria than move the criteria.

**"Why not just deploy it and see?"**
Because the backtest has been examined seven times, and forward data is the only evidence
that cannot be fished. Shadow mode collects it at zero risk.

**"What would you do differently?"**
Evaluate each of the five conditions' predictive power individually before bundling them into
a vote. I measured whether the bundle makes money, not whether each signal carries
information — so a worthless condition would be invisible inside a 3-of-5 rule.

**"What's the biggest weakness?"**
Concentration. Ten pairs, but seven have USD on one side — I have roughly three or four
independent bets, not ten. That's the mechanical cause of the 29% drawdown, and no amount of
model tuning fixes it.

**"What did you learn?"**
That most of the engineering effort in a trading system goes into not fooling yourself. The
model is a few hundred lines. The leakage firewall, the walk-forward harness, the
multiple-testing correction and the fail-closed execution guard are the rest of it.
