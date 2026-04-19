# betting-backtester

A research-grade backtesting framework for sports-betting exchange strategies on 1X2 (home/draw/away) markets. Built to be honest about what it is: a library for testing quantitative strategies against historical and synthetic data, with the controls necessary to produce results you'd trust in a research context — walk-forward evaluation, bootstrap confidence intervals on yield, structural lookahead safety, and Betfair-style per-market commission.

Not a trading bot. No live venue integration, no slippage modelling, no order-routing layer. Library-only by design.

## What's in it

Ten modules, built module-by-module through design review. Each was designed → reviewed → implemented → reviewed → tested → reviewed → PR'd → merged. The PR history on this repo is the full build provenance.

| Module | What it does |
|---|---|
| Canonical models | Pydantic-validated event types (`Match`, `OddsSnapshot`, `MatchResult`, `SelectionOdds`) and the two events that flow through a backtest (`OddsAvailable`, `MatchSettled`). UTC-only timestamps. `back_price <= lay_price` invariant. |
| SyntheticGenerator | Correctness rig. Fair-odds event streams from a configured outcome distribution. Re-callable; deterministic under seed. |
| NetWinningsCommission | Per-market commission aggregation with winners-only pro-rata attribution. `math.fsum` on every summation. Immutable per-bet breakdown. |
| Backtester | Protocol-based `Strategy` interface, committed-funds accounting, rejection log, bankroll invariants enforced to 1e-6. |
| BacktestResult | Per-match equity curve, summary metrics (9 fields), bootstrap CI on yield with per-MATCH resampling to preserve arb correlation structure. |
| FootballDataLoader | Reads football-data.co.uk CSV files. Pinnacle odds only (PSH/PSD/PSA, PH/PD/PA fallback). Kickoff-5min odds snapshot, kickoff+2h settlement. Exposes a `.matches` directory for strategies that need team identity. |
| FavouriteBacker | Baseline strategy. Stakes the shortest-price selection with HOME>DRAW>AWAY tie-break. |
| WalkForwardEvaluator | Rolling time-based train/test windows. Cohort-based test partitioning (a match is in the test cohort iff *both* its odds and settlement fall in the test window). Chained bankroll across windows. |
| XgPoissonStrategy | Dixon-Coles-lite model: per-team attack/defence ratings plus a global home-advantage parameter, fit by weighted L2-regularised MLE via `scipy.optimize`. Exponential time-decay on training matches. Fractional Kelly sizing for back and lay with per-bet exposure cap. |
| ArbitrageDetector | Back-side three-way arbitrage detector with equal-profit staking. Ships with an `ArbitrageGenerator` data source that injects arbs at known positions (for tests) or at a Bernoulli rate (for realistic streams). |

## Quick start

```python
from datetime import datetime, timezone
from betting_backtester.synthetic import (
    SyntheticGenerator, SyntheticGeneratorConfig, TrueProbabilities,
)
from betting_backtester.strategies.favourite_backer import FavouriteBacker
from betting_backtester.commission import NetWinningsCommission
from betting_backtester.backtester import Backtester
from betting_backtester.backtest_result import BacktestResult
from betting_backtester.reporting import compute_yield_ci

config = SyntheticGeneratorConfig(
    n_matches=500,
    true_probs=TrueProbabilities(home=0.45, draw=0.27, away=0.28),
    seed=42,
    start=datetime(2024, 8, 1, 15, 0, tzinfo=timezone.utc),
)

backtester = Backtester(
    event_source=SyntheticGenerator(config),
    strategy=FavouriteBacker(stake=10.0),
    commission_model=NetWinningsCommission(rate=0.05),
    starting_bankroll=1_000.0,
    seed=0,
)

raw = backtester.run()
result = BacktestResult.from_raw(raw, starting_bankroll=1_000.0, t0=config.start)
ci = compute_yield_ci(result, n_resamples=10_000, seed=0)

print(f"Net P&L: {result.summary_metrics.net_pnl:.2f}")
print(f"Bets placed: {result.summary_metrics.n_bets}")
print(f"Yield: {ci.point:.4f} (95% CI: [{ci.lower:.4f}, {ci.upper:.4f}])")
```

Swap `FavouriteBacker` for `XgPoissonStrategy` (with a `match_directory`) to run a real model. Swap `SyntheticGenerator` for `FootballDataLoader` to run on historical data. Wrap the whole thing in a `WalkForwardEvaluator` to do rolling train/test evaluation.

## Design properties worth calling out

**Lookahead safety is structural.** Strategies receive events via `fit(Iterable[Event])` and `on_odds(OddsSnapshot, PortfolioView)`; they never hold a forward iterator. The walk-forward evaluator enforces cohort-based membership so a match whose odds fall inside a test window but whose settlement falls outside (or vice versa) is explicitly excluded rather than silently leaked. There's an end-to-end test where a team's strength reverses between train and test windows — the strategy must bet according to training-period strength and lose money; a lookahead leak would flip the sign.

**Walk-forward over a single-pass backtest.** Any non-trivial strategy (one that calls `fit`) is evaluated by running independent train/test cycles. Bankroll chains across windows. Aggregate metrics are computed on the concatenated ledger with a consistency check catching cross-window drift.

**Bootstrap CI at the match level.** Yield confidence intervals resample *matches*, not bets. Matches are the iid unit — bets within a match (e.g. the three legs of an arbitrage) are correlated, and bet-level resampling understates variance.

**Per-market commission.** `NetWinningsCommission` aggregates bets on the same match into a single market-winnings figure before applying the commission rate, matching the Betfair model. This is the property that makes arbitrage survive commission as a clean multiplicative scaling rather than a per-bet subtraction.

**Determinism is a tested invariant.** Identical inputs produce byte-identical outputs at the strategy, generator, evaluator, and pipeline levels. Tested explicitly, not assumed.

**Float precision via `math.fsum`.** Every summation over ledger entries, equity curves, or bootstrap resamples uses `math.fsum` to avoid cancellation error drift.

## Tech stack

Python 3.12, [Pydantic](https://docs.pydantic.dev/) for event and config validation, [scipy](https://scipy.org/) for the Dixon-Coles MLE, [numpy](https://numpy.org/) for the numerical kernels. `mypy --strict` and `ruff` clean across the whole project.

No database, no web server, no plotting dependencies. Library-only.

## Install

```bash
git clone https://github.com/<your-username>/betting-backtester.git
cd betting-backtester
uv sync                     # or: pip install -e .
```

## Run the tests

```bash
uv run pytest               # full suite
uv run pytest -q            # terser output
uv run mypy --strict src    # type check
uv run ruff check           # lint
```

~800 tests. Runs in under 10 seconds on a modern laptop.

## Repository layout

```
src/betting_backtester/
├── models.py              # Pydantic event/fixture models
├── event_source.py        # EventSource protocol
├── synthetic.py           # SyntheticGenerator
├── football_data.py       # CSV loader for football-data.co.uk
├── commission.py          # NetWinningsCommission
├── backtester.py          # Backtester core, Strategy protocol, PortfolioView
├── backtest_result.py     # BacktestResult, SummaryMetrics, EquityPoint
├── reporting.py           # Bootstrap CI on yield
├── walk_forward.py        # WalkForwardEvaluator
├── dixon_coles.py         # Dixon-Coles-lite MLE model
├── kelly.py               # Back/lay Kelly utilities
├── arbitrage_generator.py # ArbitrageGenerator + ArbSchedule
└── strategies/
    ├── favourite_backer.py
    ├── xg_poisson.py
    └── arbitrage_detector.py

tests/
└── ...                    # one test file per source module
```

## What this is not

No live-venue integration. No order-routing. No slippage modelling. No partial-fill handling. No latency simulation. No market-impact model. A strategy that backtests profitably here is a strategy with an edge on *clean synthetic or historical* data; closing the gap to a live deployment is a separate engineering project.

The framework is explicit about this on purpose. A backtester that silently promises live-viable results is a worse research tool than one that's honestly scoped.

## Provenance

Each module was specified → designed → reviewed → implemented → reviewed → tested → reviewed → opened as a PR → reviewed again by an AI code reviewer → merged. The PR history is the full build history; specific mathematical decisions (Dixon-Coles gauge fixing, arbitrage P&L formula, commission scaling) are documented in the corresponding PR descriptions. Nothing was merged without the math being worked from scratch in review.

---

Built as the first of four portfolio projects in quantitative software infrastructure. Module roadmap, design decisions, and per-module review notes in `docs/DESIGN.md`.