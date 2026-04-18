# Design

A small, event-driven backtesting framework for betting-exchange strategies. This
document defines the interfaces, invariants, and testing requirements before any code
is written. It is the contract; implementation follows.

## V1 scope

V1 is deliberately small. Anything not listed here is out of scope.

- **Market:** 1X2 only (home / draw / away).
- **Sides:** back and lay both supported.
- **Currency:** single-currency, GBP. No FX.
- **Stakes:** decimal units of bankroll.
- **Odds granularity:** at most **one `OddsAvailable` event per match**. The data we
  load (football-data.co.uk closing odds) gives us exactly one snapshot per match, so
  v1 assumes that model. In-play or multi-snapshot data is explicitly deferred.
- **Bet adjustment:** not supported. A bet, once placed, is held to settlement. No
  cancel, no re-stake, no hedging of an existing position by editing it (a strategy
  may of course place a new, independent bet on the other side, but that is a second
  `BetOrder`, not an adjustment).
- **Commission:** applied at settlement. Betfair-style (percentage of net market
  winnings) is the default; other models plug in behind a small interface.
- **Persistence:** in-memory `BacktestResult`, with `.to_dataframe()` and
  `.to_parquet(path)` helpers for export. No database, no live writes during a run.

## Data layer

The data layer's job is to produce a **single, canonical, time-ordered stream of
events** that the backtester consumes. Source-specific quirks stay behind the loader.

### Canonical models (Pydantic)

- **`Match`** — fixture identity: `match_id`, `kickoff` (UTC), `league`, `season`,
  `home`, `away`.
- **`OddsSnapshot`** — `match_id`, `timestamp`, and, per selection (home/draw/away),
  a `back_price` and a `lay_price`. For v1 the back and lay prices will typically be
  equal (football-data.co.uk gives a single closing price), but the shape supports a
  real exchange book later without an interface change.
- **`MatchResult`** — `match_id`, `timestamp` (settlement time), final score, derived
  outcome.

### Event stream

Two event types flow through the system:

- `OddsAvailable(snapshot)` — a new odds snapshot is observable.
- `MatchSettled(result)` — a match has finalised.

Both carry an explicit `timestamp`. The stream is a pull iterator of events in
strict non-decreasing timestamp order. Ties (same timestamp) are broken
deterministically by event type (settlement before new-odds) and then by `match_id`,
so runs are reproducible across machines.

### Loaders

Two implementations of an `EventSource` protocol:

1. **`FootballDataLoader`** — reads football-data.co.uk CSVs, normalises column names
   across seasons/leagues, emits one `OddsAvailable` just before kickoff and one
   `MatchSettled` at the match end.
2. **`SyntheticGenerator`** — samples fixtures and outcomes from a configurable true
   probability distribution. Takes an explicit seed. Used to verify the backtester
   recovers expected P&L when fed a strategy with a known edge.

Strategy-specific auxiliary data (e.g. xG ratings for the Poisson strategy) is **not**
part of this stream. It is loaded separately and injected into the strategy at
construction. The canonical stream stays minimal.

## Lookahead safety (invariant + test)

This is the single most important correctness property of a backtester. It is stated
explicitly and enforced by API shape, not by convention.

**Invariant.** At the moment a strategy is invoked to act on an event with timestamp
`t`, it may observe only events with timestamp `≤ t`. Every feature, prediction, or
probability the strategy uses at time `t` must be derivable from the subset of the
stream with timestamp `≤ t`.

**Enforcement.**

- The strategy does **not** hold a reference to the event iterator. It cannot pull
  "the next event" or scan ahead. The backtester owns the iterator and pushes events
  in one at a time via callbacks (`on_odds`, `on_settled`).
- The `OddsSnapshot` passed to `on_odds` is the strategy's only view of the market at
  that instant. The `PortfolioView` passed alongside it is pinned to `t`.
- Strategies that need fitted models (xG/Poisson) receive a `fit(history)` call with
  events from the **training window only**, before the evaluation window begins. No
  refitting from live events in v1 — if we add it later it will pass only the subset
  with timestamp `≤ t`.

**Tests.**

- A dedicated test constructs a stream where a post-kickoff event carries information
  that would "leak" a correct prediction. The strategy under test must not achieve a
  hit-rate above the prior, confirming it cannot see the leaking event.
- A property test asserts the event stream is emitted in strict non-decreasing
  timestamp order (with deterministic tie-breaks).
- An API-shape test confirms `Strategy` has no method that receives the raw event
  iterator and that `on_odds` / `on_settled` are the only input channels.

## Strategy interface

A strategy is a small stateful object. It does not own cash, commission, or
settlement — the backtester does. The strategy's only job is: given market state at
time `t`, emit zero or more bet orders.

Protocol (plain English):

- `fit(history)` — optional. Called once before the evaluation window. History is a
  sequence of past events. Trivial strategies leave it as a no-op.
- `on_odds(snapshot, portfolio) -> list[BetOrder]` — the decision point.
- `on_settled(result, portfolio) -> None` — notification hook for strategies that
  track their own statistics. May not emit orders.

`PortfolioView` is a read-only snapshot of bankroll, open bets, and realised P&L so
far. Strategies can do Kelly sizing or limit exposure without managing cash directly.

`BetOrder` is a Pydantic model: `match_id`, `selection` (home/draw/away), `side`
(back/lay), `price`, `stake`. An order is validated against the current snapshot when
it arrives at the backtester. An invalid order (e.g. price that wasn't on offer,
non-positive stake, stake exceeding available bankroll net of reserved liability) is
**not** dropped silently and is **not** added to the bet ledger. It is appended to a
separate **rejections log** on the `BacktestResult` with the original order, the
snapshot timestamp, and a machine-readable rejection reason. Keeping rejections out
of the ledger means ROI, yield, hit rate, and the bootstrap CI all operate on
actually-placed bets; rejections remain inspectable for debugging strategies but
never contaminate performance metrics.

## Backtester

**Inputs**

- a `Strategy`
- an `EventSource` (the canonical stream)
- a `CommissionModel`
- initial bankroll (GBP)
- an explicit random seed

**Loop.** For each event in order: if `OddsAvailable`, call `strategy.on_odds(...)`,
validate each returned `BetOrder` against the snapshot, and record accepted orders as
open bets with their liability reserved from bankroll. If `MatchSettled`, resolve
every open bet on that match, apply commission, release liability, realise P&L, update
bankroll, append to the ledger, call `strategy.on_settled(...)`.

**Output: `BacktestResult`.** An immutable record of one run.

- **Bet ledger** — one row per *placed* bet: match_id, timestamp, selection, side,
  price, stake, liability, outcome, gross P&L, commission, net P&L, bankroll after.
- **Rejections log** — one row per rejected order: match_id, timestamp, selection,
  side, price, stake, rejection reason. Kept separate from the ledger; never enters
  P&L.
- **Equity curve** — timestamp, bankroll.
- **Summary metrics** — number of bets, hit rate, turnover, ROI on turnover, max
  drawdown, and the bootstrap CI on yield described below.
- **Helpers** — `.to_dataframe()` returns the ledger as a pandas DataFrame;
  `.to_parquet(path)` writes it. The equity curve and rejections log are exposed the
  same way.

### Commission aggregation

Betfair-style commission is a percentage of **net market winnings per customer per
market**, not per bet. In v1 terms: for each match, sum the gross P&L of every bet
the strategy placed on that match (back and lay, any selection). If the sum is
positive, commission is `rate × sum`; if zero or negative, commission is zero. That
aggregated commission is then attributed back to the contributing bets
pro-rata-on-gross-P&L for ledger-row accounting, so per-row net P&L still sums to the
correct per-match total. This matters for the arbitrage strategy, which by construction
places bets on multiple outcomes of the same match; computing commission per bet in
isolation would overstate costs. Other `CommissionModel` implementations (flat pct
per stake, zero-commission) are free to ignore aggregation.

## Confidence intervals on yield

V1 uses a **non-parametric bootstrap CI on yield** (net P&L divided by total
turnover), not a t-test.

**Procedure.** Resample the bet ledger with replacement `N` times (default
`N = 10_000`), compute yield on each resample, report the mean and the 2.5 / 97.5
percentiles as the 95% CI. The bootstrap RNG takes its seed from the backtester's
seed, so the reported CI is reproducible.

**Why bootstrap and not a t-test.**

- Per-bet returns are heavily right-skewed: mostly small losses, occasional large
  wins at long odds. Normality is a poor assumption for any sample size we realistically
  produce (hundreds, maybe low thousands of bets).
- Stakes are variable (Kelly or capped-fraction sizing), which breaks the iid
  assumption a t-test on per-bet returns would rely on.
- The metric we care about is yield, a ratio — bootstrapping gives a CI directly on
  that statistic without needing to derive its sampling distribution analytically.
- When the shipped example strategies lose money after commission (the expected
  outcome), a CI that overlaps zero is the honest way to report it. Bootstrap makes
  that visible without pretending we have enough data to run a parametric test.

## Walk-forward evaluation

A separate layer on top of the single-pass backtester, not baked into it.

`WalkForwardEvaluator.run(strategy_factory, events, window_spec, seed)` splits the
event stream into rolling (train, test) windows, constructs a fresh strategy from the
factory for each test window, calls `fit(train_events)`, runs the backtester on the
test window, and returns a list of per-window `BacktestResult`s plus an aggregate
view. The aggregate concatenates ledgers and recomputes summary metrics (including a
fresh bootstrap CI) over the full out-of-sample history.

The single-pass backtester stays ignorant of windowing.

## Determinism

**Requirement.** Identical inputs and identical seed produce a byte-identical
`BacktestResult` (same bet ledger row order and contents, same equity curve, same
summary metrics including bootstrap CI bounds).

**What this forces on the implementation.**

- Every randomised component — `SyntheticGenerator`, the bootstrap summariser, any
  stochastic sizing rule — receives its RNG from a single seed plumbed in via the
  backtester or evaluator. No implicit global seeds, no `random.seed(...)` at module
  import, no `numpy.random` default RNG usage.
- Deterministic tie-breaking in the event stream (see data layer).
- Stable iteration order in all internal collections (Python dicts are insertion-ordered;
  no reliance on set iteration).
- No wall-clock time in results. Timestamps come from the event stream; durations are
  not recorded.

**Tests.**

- **Same-seed determinism:** run the backtester twice with identical inputs and seed;
  assert the two `BacktestResult`s are equal field-for-field.
- **Seed sensitivity:** run with two different seeds on a synthetic stream with
  stochastic sizing; assert the results differ. This catches the trivial "determinism
  by ignoring the seed" bug.
- **Walk-forward determinism:** same-seed equality over the aggregated result of a
  multi-window walk-forward run.

## Testing posture

- Every non-trivial module ships with pytest tests before the next module begins.
- Synthetic data generator + a strategy with a known edge is the primary end-to-end
  check: the backtester must recover expected P&L within the bootstrap CI.
- Lookahead, determinism, and commission-correctness are called out as standalone
  test files, not folded into general integration tests, so a regression in any of
  them is obvious from the failure name.

## Module build order (proposed)

Not part of the contract, but the order I will propose when we move to
implementation:

1. Canonical models + event types.
2. Synthetic generator (needed for testing everything downstream).
3. Commission model.
4. Backtester core (single-pass).
5. Reporting + bootstrap CI.
6. Football-data.co.uk loader.
7. Trivial strategy (back-the-favourite) — validates the whole pipeline end-to-end.
8. Walk-forward evaluator.
9. xG/Poisson strategy.
10. Arbitrage detector strategy.

We confirm this order together before I start on module 1.
