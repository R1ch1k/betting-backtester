"""Module 8: walk-forward evaluator.

A thin layer on top of the single-pass
:class:`~betting_backtester.backtester.Backtester` that splits a long
event stream into rolling (train, test) time windows, fits a fresh
strategy per window, runs the backtester on each window's test cohort,
and aggregates the per-window results. The single-pass backtester stays
ignorant of windowing: each window is driven by a fresh
:class:`~betting_backtester.backtester.Backtester` instance.

Window shape
------------

Rolling, time-based windows only in v1. ``WindowSpec.train_duration``
and ``WindowSpec.test_duration`` are both :class:`~datetime.timedelta`
values; windows step forward by ``test_duration``, producing adjacent
non-overlapping test windows. Expanding and count-based shapes are
deferred.

Event partitioning
------------------

Half-open intervals throughout: ``[train_start, train_end)`` and
``[test_start, test_end)``, with ``train_end == test_start`` exactly.

* **Training partition** -- all events (both ``OddsAvailable`` and
  ``MatchSettled``) whose timestamp lies in ``[train_start, train_end)``,
  in stream order. A match whose ``OddsAvailable`` falls in the train
  window while its ``MatchSettled`` falls in the test window will only
  have its odds event visible to ``fit``; strategies that need paired
  training data must filter for matches with both events present.
* **Test partition** -- cohort-based, not timestamp-based. A match is
  in test window ``k`` iff **both** its ``OddsAvailable`` and
  ``MatchSettled`` timestamps fall in ``[test_start_k, test_end_k)``.
  Matches whose odds and settlement straddle a window boundary are
  excluded from *both* windows. This produces symmetric, quantifiable
  data loss rather than the silently biased subsample that dropping
  trailing unsettled bets would produce.
  :attr:`WalkForwardResult.n_matches_boundary_excluded` reports this
  count for transparency.

Strategy lifecycle
------------------

``strategy_factory`` is called exactly once per test window. The
returned instance is handed ``fit(train_events)`` and then drives the
per-window backtester. The factory contract is "return a fresh instance
each call"; returning a singleton leaks state between windows. This is
documented, not runtime-enforced.

Seeding
-------

The evaluator takes one ``seed: int``. Per-window backtesters receive
``seed + window_index`` so they do not all share an identical RNG
stream (matters only for strategies that use RNG; none of our v1
strategies do, but the derivation is future-proofing). The aggregate
bootstrap CI uses the evaluator's ``seed`` directly, not a derived one,
so the reported aggregate yield CI is reproducible from the evaluator's
construction parameters alone.

Aggregate construction
----------------------

The aggregate ledger, rejections, equity curve, and summary metrics are
built by feeding the concatenated per-window ledger into
:meth:`~betting_backtester.backtest_result.BacktestResult.from_raw`
with ``starting_bankroll`` set to the overall-run start and ``t0`` set
to the first test window's ``test_start``. This re-uses module 4b's
discipline for equity-curve derivation and summary metrics, so
aggregate ``realised_pnl`` and ``cumulative_turnover`` on each
:class:`~betting_backtester.backtest_result.EquityPoint` are cumulative
over the full run, not per-window values. The aggregate yield CI is
computed by :func:`~betting_backtester.reporting.compute_yield_ci` on
that wrapped aggregate.

Chained bankroll
----------------

Window ``N+1``'s backtester is constructed with
``starting_bankroll = window_N.result.equity_curve[-1].bankroll``, so
each window's P&L builds on the previous window's ending state. The
aggregate equity curve derived from the concatenated ledger is correct
by construction: each ledger row's ``bankroll_after`` is already
expressed in overall-run terms (``overall_starting_bankroll +
cumulative_net_pnl``), because each per-window backtester was
constructed with a starting bankroll that itself encoded prior windows'
cumulative P&L.

Memory
------

The evaluator materialises the full event stream into an internal
``tuple[Event, ...]`` on ``run()`` so windows can index into it with
``bisect``. For v1 research-scale streams (tens of thousands of events)
this fits comfortably in memory; streaming-only walk-forward is a v2
concern.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, model_validator

from betting_backtester.backtest_result import (
    BacktestResult,
    EquityPoint,
    SummaryMetrics,
)
from betting_backtester.backtester import (
    Backtester,
    RawBacktestOutput,
    RejectedOrder,
    SettledBet,
    Strategy,
)
from betting_backtester.commission import CommissionModel
from betting_backtester.event_source import EventSource
from betting_backtester.models import Event, MatchSettled, OddsAvailable
from betting_backtester.reporting import YieldCI, compute_yield_ci


def _require_utc(name: str, ts: datetime) -> None:
    if ts.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC (offset 0)")


def _match_id_of(event: Event) -> str:
    """Return the ``match_id`` an event pertains to, regardless of type.

    Mirrors the defensive dispatch in
    :meth:`~betting_backtester.backtester.Backtester.run`: both
    concrete ``Event`` subtypes are handled explicitly and any other
    runtime type raises :class:`TypeError`. The ``Event`` union is
    closed today, but this belt-and-braces guard means a future
    addition (e.g. an in-play snapshot event) surfaces as a clear
    error here instead of an ``AttributeError`` on a missing
    ``.result`` or ``.snapshot``.
    """
    if isinstance(event, OddsAvailable):
        return event.snapshot.match_id
    if isinstance(event, MatchSettled):
        return event.result.match_id
    raise TypeError(f"unknown Event subtype: {type(event).__name__}")


@dataclass(frozen=True)
class _FixedEventSource:
    """Tiny :class:`EventSource` over a pre-captured tuple of events.

    Scoped to this module; not exported. Each walk-forward window
    constructs one of these over its cohort-filtered test events and
    hands it to a fresh
    :class:`~betting_backtester.backtester.Backtester`. The
    ``EventSource`` contract ("``events()`` may be called multiple
    times, each call returns a fresh iterator") is satisfied trivially:
    ``iter(self.events_tuple)`` returns an independent iterator per
    call.
    """

    events_tuple: tuple[Event, ...]

    def events(self) -> Iterator[Event]:
        return iter(self.events_tuple)


class WindowSpec(BaseModel):
    """Rolling, time-based walk-forward window specification.

    Windows step forward by ``test_duration``, producing adjacent
    non-overlapping test windows. Expanding windows and count-based
    windows are intentional v2 additions; the v1 shape is deliberately
    minimal.
    """

    model_config = ConfigDict(frozen=True)

    train_duration: timedelta = Field(
        description="Length of the training window. Strict positive."
    )
    test_duration: timedelta = Field(
        description=(
            "Length of each test window, and the step size between "
            "consecutive test windows. Strict positive."
        )
    )

    @model_validator(mode="after")
    def _validate(self) -> WindowSpec:
        if self.train_duration <= timedelta(0):
            raise ValueError(
                f"train_duration must be positive, got {self.train_duration}"
            )
        if self.test_duration <= timedelta(0):
            raise ValueError(
                f"test_duration must be positive, got {self.test_duration}"
            )
        return self

    def iter_windows(
        self, stream_start: datetime, stream_end: datetime
    ) -> Iterator[tuple[datetime, datetime, datetime, datetime]]:
        """Yield ``(train_start, train_end, test_start, test_end)`` per window.

        Pure arithmetic over ``self``, ``stream_start`` and
        ``stream_end`` -- no event-stream awareness. Yields zero
        windows if the stream is too short to fit one full
        ``train + test`` span; the walk-forward evaluator raises
        :class:`ValueError` in that case. Trailing events that cannot
        fill a whole test window are implicitly dropped (the
        ``<= stream_end`` check rejects any test window whose right
        edge would sit past the last event).
        """
        _require_utc("stream_start", stream_start)
        _require_utc("stream_end", stream_end)
        test_start = stream_start + self.train_duration
        while test_start + self.test_duration <= stream_end:
            test_end = test_start + self.test_duration
            train_end = test_start
            train_start = test_start - self.train_duration
            yield (train_start, train_end, test_start, test_end)
            test_start = test_end


class WindowResult(BaseModel):
    """Per-window slice of a walk-forward run.

    ``result`` carries the per-window
    :class:`~betting_backtester.backtest_result.BacktestResult`. The
    four timestamps let a consumer know exactly which time range the
    window covered without having to re-inspect events.
    ``n_train_events`` is recorded so a reader can cheaply check "did
    this window have training data?" without re-materialising the
    stream.
    """

    model_config = ConfigDict(frozen=True)

    train_start: datetime = Field(
        description="Inclusive start of the training window, UTC."
    )
    train_end: datetime = Field(
        description=(
            "Exclusive end of the training window, UTC. Equals "
            "``test_start`` exactly; windows are adjacent by construction."
        )
    )
    test_start: datetime = Field(description="Inclusive start of the test window, UTC.")
    test_end: datetime = Field(description="Exclusive end of the test window, UTC.")
    n_train_events: int = Field(
        ge=0,
        description=(
            "Number of events (both ``OddsAvailable`` and ``MatchSettled``) "
            "passed to ``strategy.fit`` for this window."
        ),
    )
    result: BacktestResult = Field(
        description=("Module 4b wrapper over this window's single-pass backtest run.")
    )

    @model_validator(mode="after")
    def _validate(self) -> WindowResult:
        _require_utc("train_start", self.train_start)
        _require_utc("train_end", self.train_end)
        _require_utc("test_start", self.test_start)
        _require_utc("test_end", self.test_end)
        if self.train_start >= self.train_end:
            raise ValueError(
                f"train_start ({self.train_start}) must precede "
                f"train_end ({self.train_end})"
            )
        if self.test_start >= self.test_end:
            raise ValueError(
                f"test_start ({self.test_start}) must precede "
                f"test_end ({self.test_end})"
            )
        if self.train_end != self.test_start:
            raise ValueError(
                f"train_end ({self.train_end}) must equal test_start "
                f"({self.test_start}); windows are adjacent by construction."
            )
        return self


class WalkForwardResult(BaseModel):
    """Immutable, flat record of one walk-forward run.

    The ``aggregate_*`` fields are the primary analysis entry points;
    ``per_window`` is retained for drill-down. Sole construction path
    is :meth:`WalkForwardEvaluator.run`; callers do not build this
    directly.
    """

    model_config = ConfigDict(frozen=True)

    per_window: tuple[WindowResult, ...] = Field(
        min_length=1,
        description=("One entry per completed test window, in chronological order."),
    )
    aggregate_ledger: tuple[SettledBet, ...] = Field(
        description=("Concatenation of every window's ledger, in chronological order.")
    )
    aggregate_rejections: tuple[RejectedOrder, ...] = Field(
        description=("Concatenation of every window's rejections log, in order.")
    )
    aggregate_equity_curve: tuple[EquityPoint, ...] = Field(
        description=(
            "Single curve over the full run. ``realised_pnl`` and "
            "``cumulative_turnover`` are cumulative from the overall-run "
            "start, not per-window values."
        ),
    )
    aggregate_summary: SummaryMetrics = Field(
        description=("Summary metrics recomputed on the concatenated ledger.")
    )
    aggregate_yield_ci: YieldCI = Field(
        description=(
            "Bootstrap yield CI on the concatenated ledger, computed with "
            "the evaluator's ``seed`` (not a window-derived seed)."
        ),
    )
    window_spec: WindowSpec = Field(description="Echoed for provenance.")
    starting_bankroll: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="The overall run's starting bankroll.",
    )
    seed: int = Field(description="The evaluator's seed, echoed for provenance.")
    n_matches_boundary_excluded: int = Field(
        ge=0,
        description=(
            "Count of matches whose ``OddsAvailable`` fell inside some test "
            "window but whose ``MatchSettled`` fell outside the same window. "
            "These matches are excluded from every window's test cohort to "
            "avoid biased subsampling; the count is surfaced for "
            "transparency."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> WalkForwardResult:
        if not self.aggregate_equity_curve:
            raise ValueError("aggregate_equity_curve must have at least one point")
        first = self.aggregate_equity_curve[0]
        if first.bankroll != self.starting_bankroll:
            raise ValueError(
                f"aggregate_equity_curve[0].bankroll ({first.bankroll}) "
                f"must equal starting_bankroll ({self.starting_bankroll})"
            )
        first_test_start = self.per_window[0].test_start
        if first.timestamp != first_test_start:
            raise ValueError(
                f"aggregate_equity_curve[0].timestamp ({first.timestamp}) "
                f"must equal the first window's test_start "
                f"({first_test_start})"
            )
        return self


class WalkForwardEvaluator:
    """Run a walk-forward evaluation over an event stream.

    Construction is all-keyword and validates inputs eagerly
    (``starting_bankroll`` finite and positive, bootstrap parameters in
    range). :meth:`run` is deterministic: identical ``(event_source,
    strategy_factory, commission_model, window_spec, starting_bankroll,
    seed, n_resamples, confidence)`` produce a byte-identical
    :class:`WalkForwardResult`, provided ``strategy_factory`` itself is
    deterministic.

    The strategy-factory contract is "must return a fresh instance each
    call; returning a singleton causes state leakage between windows".
    This is documented, not runtime-enforced.

    Note on ``fit`` history. The history passed to ``strategy.fit``
    contains both ``OddsAvailable`` and ``MatchSettled`` events in
    stream order, filtered by timestamp to ``[train_start, train_end)``.
    A match whose odds event falls in the train window but whose
    settlement falls past ``train_end`` (settlement lands ~2h after
    kickoff for both shipped loaders) will only have its odds visible
    to ``fit``; strategies that need paired training data must filter
    for matches with both events present.
    """

    def __init__(
        self,
        *,
        event_source: EventSource,
        strategy_factory: Callable[[], Strategy],
        commission_model: CommissionModel,
        window_spec: WindowSpec,
        starting_bankroll: float,
        seed: int,
        n_resamples: int = 10_000,
        confidence: float = 0.95,
    ) -> None:
        if not math.isfinite(starting_bankroll):
            raise ValueError(
                f"starting_bankroll must be finite, got {starting_bankroll!r}"
            )
        if starting_bankroll <= 0.0:
            raise ValueError(
                f"starting_bankroll must be positive, got {starting_bankroll}"
            )
        if n_resamples < 1:
            raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

        self._event_source: EventSource = event_source
        self._strategy_factory: Callable[[], Strategy] = strategy_factory
        self._commission_model: CommissionModel = commission_model
        self._window_spec: WindowSpec = window_spec
        self._starting_bankroll: float = float(starting_bankroll)
        self._seed: int = seed
        self._n_resamples: int = n_resamples
        self._confidence: float = confidence

    @property
    def seed(self) -> int:
        """The evaluator's seed, as passed at construction."""
        return self._seed

    def run(self) -> WalkForwardResult:
        """Execute the walk-forward run and return the aggregated result.

        Raises
        ------
        ValueError
            If the event stream is empty; if the stream's total
            duration is less than ``train_duration + test_duration``
            (no full window can run); if the stream contains multiple
            ``OddsAvailable`` or multiple ``MatchSettled`` events for
            the same ``match_id``; or if the concatenated ledger is
            empty (no bets placed in any window), which bubbles up
            from the aggregate bootstrap step.
        """
        events: tuple[Event, ...] = tuple(self._event_source.events())
        if not events:
            raise ValueError("event stream is empty; cannot run walk-forward.")

        stream_start = events[0].timestamp
        stream_end = events[-1].timestamp
        stream_duration = stream_end - stream_start
        min_required = (
            self._window_spec.train_duration + self._window_spec.test_duration
        )
        if stream_duration < min_required:
            raise ValueError(
                f"stream duration ({stream_duration}) is less than "
                f"train_duration + test_duration ({min_required}); no full "
                "walk-forward window can run. Either supply a longer "
                "stream or shorten the WindowSpec."
            )

        match_timestamps = self._index_match_timestamps(events)
        event_timestamps = [e.timestamp for e in events]

        per_window: list[WindowResult] = []
        aggregate_ledger: list[SettledBet] = []
        aggregate_rejections: list[RejectedOrder] = []
        n_boundary_excluded = 0
        window_starting_bankroll = self._starting_bankroll

        for window_index, (
            train_start,
            train_end,
            test_start,
            test_end,
        ) in enumerate(self._window_spec.iter_windows(stream_start, stream_end)):
            train_events = self._slice_by_timestamp(
                events, event_timestamps, train_start, train_end
            )
            candidate_test_events = self._slice_by_timestamp(
                events, event_timestamps, test_start, test_end
            )
            cohort, n_odds_in_window = self._cohort_for_window(
                candidate_test_events, match_timestamps, test_start, test_end
            )
            n_boundary_excluded += n_odds_in_window - len(cohort)
            test_events = tuple(
                e for e in candidate_test_events if _match_id_of(e) in cohort
            )

            strategy = self._strategy_factory()
            strategy.fit(train_events)

            backtester = Backtester(
                event_source=_FixedEventSource(test_events),
                strategy=strategy,
                commission_model=self._commission_model,
                starting_bankroll=window_starting_bankroll,
                seed=self._seed + window_index,
                strict_settlement=True,
            )
            raw = backtester.run()
            window_backtest = BacktestResult.from_raw(
                raw,
                starting_bankroll=window_starting_bankroll,
                t0=test_start,
            )
            per_window.append(
                WindowResult(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    n_train_events=len(train_events),
                    result=window_backtest,
                )
            )
            aggregate_ledger.extend(window_backtest.ledger)
            aggregate_rejections.extend(window_backtest.rejections)
            window_starting_bankroll = window_backtest.equity_curve[-1].bankroll

        aggregate_raw = RawBacktestOutput(
            ledger=tuple(aggregate_ledger),
            rejections=tuple(aggregate_rejections),
        )
        aggregate_result = BacktestResult.from_raw(
            aggregate_raw,
            starting_bankroll=self._starting_bankroll,
            t0=per_window[0].test_start,
        )
        aggregate_yield_ci = compute_yield_ci(
            aggregate_result,
            seed=self._seed,
            n_resamples=self._n_resamples,
            confidence=self._confidence,
        )

        return WalkForwardResult(
            per_window=tuple(per_window),
            aggregate_ledger=aggregate_result.ledger,
            aggregate_rejections=aggregate_result.rejections,
            aggregate_equity_curve=aggregate_result.equity_curve,
            aggregate_summary=aggregate_result.summary_metrics,
            aggregate_yield_ci=aggregate_yield_ci,
            window_spec=self._window_spec,
            starting_bankroll=self._starting_bankroll,
            seed=self._seed,
            n_matches_boundary_excluded=n_boundary_excluded,
        )

    # ----- helpers ------------------------------------------------------------

    @staticmethod
    def _index_match_timestamps(
        events: tuple[Event, ...],
    ) -> dict[str, tuple[datetime | None, datetime | None]]:
        """Build ``{match_id: (odds_ts, settled_ts)}`` for cohort filtering.

        Each match may contribute at most one ``OddsAvailable`` (v1
        contract) and at most one ``MatchSettled``; a duplicate of
        either type for the same ``match_id`` is treated as a
        data-integrity error and raises :class:`ValueError` -- no
        silent last-write-wins.
        """
        indexed: dict[str, tuple[datetime | None, datetime | None]] = {}
        for event in events:
            mid = _match_id_of(event)
            odds_ts, settled_ts = indexed.get(mid, (None, None))
            if isinstance(event, OddsAvailable):
                if odds_ts is not None:
                    raise ValueError(
                        f"multiple OddsAvailable events for match_id "
                        f"{mid!r}; v1 assumes at most one snapshot per match."
                    )
                odds_ts = event.timestamp
            else:
                if settled_ts is not None:
                    raise ValueError(
                        f"multiple MatchSettled events for match_id {mid!r}."
                    )
                settled_ts = event.timestamp
            indexed[mid] = (odds_ts, settled_ts)
        return indexed

    @staticmethod
    def _slice_by_timestamp(
        events: tuple[Event, ...],
        event_timestamps: list[datetime],
        lo: datetime,
        hi: datetime,
    ) -> tuple[Event, ...]:
        """Return events with ``lo <= timestamp < hi``, preserving order.

        Uses ``bisect_left`` on a parallel list of timestamps so window
        setup is O(log N) rather than O(N) per window. The half-open
        interval matches the design contract (an event at exactly
        ``hi`` belongs to the next window, not this one).
        """
        lo_idx = bisect.bisect_left(event_timestamps, lo)
        hi_idx = bisect.bisect_left(event_timestamps, hi)
        return events[lo_idx:hi_idx]

    @staticmethod
    def _cohort_for_window(
        candidate_events: tuple[Event, ...],
        match_timestamps: dict[str, tuple[datetime | None, datetime | None]],
        test_start: datetime,
        test_end: datetime,
    ) -> tuple[set[str], int]:
        """Compute the cohort of matches fully inside ``[test_start, test_end)``.

        Returns a ``(cohort, n_odds_in_window)`` pair:

        * ``cohort`` -- ``match_id``\\ s whose ``OddsAvailable`` and
          ``MatchSettled`` both fall in the window.
        * ``n_odds_in_window`` -- count of ``match_id``\\ s whose
          ``OddsAvailable`` falls in the window, whether or not their
          settlement also does. The caller subtracts ``len(cohort)``
          to tally boundary-excluded matches.

        Since ``candidate_events`` is already sliced to
        ``[test_start, test_end)``, every ``OddsAvailable`` observed
        here is in-window; we only need to check the corresponding
        ``MatchSettled`` against the same interval.
        """
        odds_in_window: set[str] = set()
        cohort: set[str] = set()
        for event in candidate_events:
            if not isinstance(event, OddsAvailable):
                continue
            mid = event.snapshot.match_id
            odds_in_window.add(mid)
            settled_ts = match_timestamps[mid][1]
            if settled_ts is not None and test_start <= settled_ts < test_end:
                cohort.add(mid)
        return cohort, len(odds_in_window)


__all__ = [
    "WalkForwardEvaluator",
    "WalkForwardResult",
    "WindowResult",
    "WindowSpec",
]
