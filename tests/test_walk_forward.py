"""Tests for module 8: walk-forward evaluator.

Organisation:

* ``TestWindowSpec`` / ``TestWindowResult`` -- Pydantic validator surface.
* ``TestWindowIteration`` -- pure arithmetic on :meth:`WindowSpec.iter_windows`
  (one full window, two adjacent, trailing-partial dropped, too-short).
* ``TestEventPartition`` -- half-open interval semantics via the
  ``_slice_by_timestamp`` helper.
* ``TestEvaluatorInit`` -- constructor validation
  (bankroll/n_resamples/confidence).
* ``TestEmptyStream`` / ``TestStreamTooShort`` -- ``ValueError`` at
  :meth:`WalkForwardEvaluator.run` for degenerate streams.
* ``TestEndToEndSynthetic`` -- primary correctness gate:
  :class:`~betting_backtester.synthetic.SyntheticGenerator` with
  :class:`~betting_backtester.strategies.favourite_backer.FavouriteBacker`
  and :class:`~betting_backtester.commission.NetWinningsCommission`.
* ``TestEquityChaining`` -- per-window starting bankrolls chain; the
  aggregate equity curve matches ``BacktestResult.from_raw`` over the
  concatenated ledger.
* ``TestDeterminism`` -- same inputs and seed yield byte-identical
  :class:`WalkForwardResult` objects. Seed-sensitivity is deferred to
  module 9: v1 strategies are deterministic, so a meaningful
  seed-sensitivity test requires a strategy with RNG (xG/Poisson
  introduces one). A mock-based variant here would test the mock more
  than the evaluator.
* ``TestSeedDerivation`` -- per-window backtesters receive
  ``seed + window_index``.
* ``TestFactoryFreshness`` -- ``strategy_factory`` called once per
  window, producing a fresh instance each time.
* ``TestFit`` -- ``fit`` receives events in
  ``[train_start, train_end)`` in stream order, both event types
  included.
* ``TestTrailingPartial`` -- events past the last full test window are
  unused.
* ``TestBoundaryExclusion`` -- a match whose odds fall in one window
  and settlement in the next is excluded from both and counted in
  ``n_matches_boundary_excluded``.
* ``TestDuplicateEvents`` -- multiple ``OddsAvailable`` or
  ``MatchSettled`` events for the same ``match_id`` raise.
* ``TestYieldCI`` -- the aggregate CI equals
  ``compute_yield_ci`` applied externally to the wrapped aggregate
  result with the evaluator's seed.
* ``TestCommissionShared`` -- every per-window backtester receives the
  same :class:`CommissionModel` instance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from betting_backtester._event_ordering import stream_sort_key
from betting_backtester.backtest_result import BacktestResult
from betting_backtester.backtester import (
    Backtester,
    BetOrder,
    PortfolioView,
    RawBacktestOutput,
    Side,
    Strategy,
)
from betting_backtester.commission import CommissionModel, NetWinningsCommission
from betting_backtester.event_source import EventSource
from betting_backtester.models import (
    Event,
    MatchResult,
    MatchSettled,
    OddsAvailable,
    OddsSnapshot,
    Selection,
    SelectionOdds,
)
from betting_backtester.reporting import compute_yield_ci
from betting_backtester.strategies.favourite_backer import FavouriteBacker
from betting_backtester.synthetic import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
    TrueProbabilities,
)
from betting_backtester.walk_forward import (
    WalkForwardEvaluator,
    WindowResult,
    WindowSpec,
)

UTC = timezone.utc


# ---------- shared test helpers -------------------------------------------


def _odds(match_id: str, ts: datetime) -> OddsAvailable:
    """Build an ``OddsAvailable`` with HOME as the market favourite."""
    return OddsAvailable(
        snapshot=OddsSnapshot(
            match_id=match_id,
            timestamp=ts,
            home=SelectionOdds(back_price=2.0, lay_price=2.0),
            draw=SelectionOdds(back_price=3.5, lay_price=3.5),
            away=SelectionOdds(back_price=4.0, lay_price=4.0),
        )
    )


def _settled(
    match_id: str,
    ts: datetime,
    home_goals: int = 1,
    away_goals: int = 0,
) -> MatchSettled:
    """Build a ``MatchSettled`` with HOME winning by default."""
    return MatchSettled(
        result=MatchResult(
            match_id=match_id,
            timestamp=ts,
            home_goals=home_goals,
            away_goals=away_goals,
        )
    )


@dataclass(frozen=True)
class _ListSource:
    """Test-only :class:`EventSource` over a pre-sorted tuple of events."""

    events_tuple: tuple[Event, ...]

    def events(self) -> Iterator[Event]:
        return iter(self.events_tuple)


def _sorted_source(events: list[Event]) -> _ListSource:
    return _ListSource(events_tuple=tuple(sorted(events, key=stream_sort_key)))


def _synthetic_source(
    n_matches: int = 40,
    seed: int = 7,
    start: datetime = datetime(2024, 1, 1, 15, 0, tzinfo=UTC),
) -> SyntheticGenerator:
    return SyntheticGenerator(
        SyntheticGeneratorConfig(
            n_matches=n_matches,
            true_probs=TrueProbabilities(home=0.5, draw=0.25, away=0.25),
            seed=seed,
            start=start,
        )
    )


def _minimal_backtest_result(
    t0: datetime, starting_bankroll: float = 1000.0
) -> BacktestResult:
    """Construct a :class:`BacktestResult` with an empty ledger.

    Convenient for building :class:`WindowResult` instances in
    validator tests where the inner result is incidental.
    """
    return BacktestResult.from_raw(
        RawBacktestOutput(ledger=(), rejections=()),
        starting_bankroll=starting_bankroll,
        t0=t0,
    )


# ---------- mock strategies -----------------------------------------------


class _HomeBacker:
    """Backs the HOME selection on every snapshot.

    Used across tests that need a deterministic, non-empty ledger and
    that also want to inspect ``fit`` history and invocation counts.
    """

    def __init__(self, stake: float = 1.0) -> None:
        self._stake = stake
        self.fit_calls = 0
        self.fit_history: list[Event] = []
        self.on_odds_calls = 0
        self.on_settled_calls = 0

    def fit(self, history: Iterable[Event]) -> None:
        self.fit_calls += 1
        self.fit_history = list(history)

    def on_odds(
        self, snapshot: OddsSnapshot, portfolio: PortfolioView
    ) -> list[BetOrder]:
        self.on_odds_calls += 1
        return [
            BetOrder(
                match_id=snapshot.match_id,
                selection=Selection.HOME,
                side=Side.BACK,
                price=snapshot.home.back_price,
                stake=self._stake,
            )
        ]

    def on_settled(self, result: MatchResult, portfolio: PortfolioView) -> None:
        self.on_settled_calls += 1


def _home_backer_factory(
    instances: list[_HomeBacker] | None = None,
    stake: float = 1.0,
) -> tuple[Callable[[], _HomeBacker], list[_HomeBacker]]:
    """Factory helper that optionally records every instance it produces.

    Returned as ``(factory_callable, instances_list)``; the caller can
    inspect per-window state by reading the list after ``run()``.
    """
    captured: list[_HomeBacker] = [] if instances is None else instances

    def _factory() -> _HomeBacker:
        inst = _HomeBacker(stake=stake)
        captured.append(inst)
        return inst

    return _factory, captured


# ==========================================================================
# WindowSpec validation
# ==========================================================================


class TestWindowSpec:
    def test_valid_spec(self) -> None:
        spec = WindowSpec(
            train_duration=timedelta(days=5),
            test_duration=timedelta(days=3),
        )
        assert spec.train_duration == timedelta(days=5)
        assert spec.test_duration == timedelta(days=3)

    def test_equal_train_and_test_allowed(self) -> None:
        WindowSpec(
            train_duration=timedelta(days=7),
            test_duration=timedelta(days=7),
        )

    def test_zero_train_duration_rejected(self) -> None:
        with pytest.raises(ValidationError, match="train_duration"):
            WindowSpec(
                train_duration=timedelta(0),
                test_duration=timedelta(days=1),
            )

    def test_zero_test_duration_rejected(self) -> None:
        with pytest.raises(ValidationError, match="test_duration"):
            WindowSpec(
                train_duration=timedelta(days=1),
                test_duration=timedelta(0),
            )

    def test_negative_train_rejected(self) -> None:
        with pytest.raises(ValidationError, match="train_duration"):
            WindowSpec(
                train_duration=timedelta(days=-1),
                test_duration=timedelta(days=1),
            )

    def test_negative_test_rejected(self) -> None:
        with pytest.raises(ValidationError, match="test_duration"):
            WindowSpec(
                train_duration=timedelta(days=1),
                test_duration=timedelta(days=-1),
            )

    def test_frozen(self) -> None:
        spec = WindowSpec(
            train_duration=timedelta(days=5),
            test_duration=timedelta(days=3),
        )
        with pytest.raises(ValidationError):
            spec.train_duration = timedelta(days=99)


# ==========================================================================
# WindowResult validation
# ==========================================================================


class TestWindowResult:
    def test_valid_instance(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        test_start = t0 + timedelta(days=5)
        test_end = test_start + timedelta(days=3)
        bt = _minimal_backtest_result(test_start)
        wr = WindowResult(
            train_start=t0,
            train_end=test_start,
            test_start=test_start,
            test_end=test_end,
            n_train_events=0,
            result=bt,
        )
        assert wr.train_end == wr.test_start

    def test_train_end_must_equal_test_start(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        test_start = t0 + timedelta(days=5)
        bt = _minimal_backtest_result(test_start)
        with pytest.raises(ValidationError, match="train_end"):
            WindowResult(
                train_start=t0,
                train_end=test_start - timedelta(seconds=1),
                test_start=test_start,
                test_end=test_start + timedelta(days=3),
                n_train_events=0,
                result=bt,
            )

    def test_train_start_before_train_end(self) -> None:
        t = datetime(2024, 1, 5, tzinfo=UTC)
        bt = _minimal_backtest_result(t)
        with pytest.raises(ValidationError, match="train_start"):
            WindowResult(
                train_start=t,
                train_end=t,
                test_start=t,
                test_end=t + timedelta(days=3),
                n_train_events=0,
                result=bt,
            )

    def test_test_start_before_test_end(self) -> None:
        t = datetime(2024, 1, 5, tzinfo=UTC)
        bt = _minimal_backtest_result(t)
        with pytest.raises(ValidationError, match="test_start"):
            WindowResult(
                train_start=t - timedelta(days=1),
                train_end=t,
                test_start=t,
                test_end=t,
                n_train_events=0,
                result=bt,
            )

    def test_utc_required(self) -> None:
        non_utc = timezone(timedelta(hours=-5))
        t_non_utc = datetime(2024, 1, 1, tzinfo=non_utc)
        t_utc = t_non_utc.astimezone(UTC)
        # Build a valid-geometry instance with one non-UTC timestamp.
        test_start = t_utc + timedelta(days=5)
        bt = _minimal_backtest_result(test_start)
        with pytest.raises(ValidationError, match="UTC"):
            WindowResult(
                train_start=t_non_utc,
                train_end=test_start,
                test_start=test_start,
                test_end=test_start + timedelta(days=3),
                n_train_events=0,
                result=bt,
            )


# ==========================================================================
# Window iteration arithmetic
# ==========================================================================


class TestWindowIteration:
    def _spec(self) -> WindowSpec:
        return WindowSpec(
            train_duration=timedelta(days=5),
            test_duration=timedelta(days=3),
        )

    def test_exactly_one_full_window(self) -> None:
        spec = self._spec()
        stream_start = datetime(2024, 1, 1, tzinfo=UTC)
        stream_end = stream_start + timedelta(days=8)  # train + test
        windows = list(spec.iter_windows(stream_start, stream_end))
        assert len(windows) == 1
        train_start, train_end, test_start, test_end = windows[0]
        assert train_start == stream_start
        assert train_end == stream_start + timedelta(days=5)
        assert test_start == train_end
        assert test_end == stream_end

    def test_two_adjacent_windows(self) -> None:
        spec = self._spec()
        stream_start = datetime(2024, 1, 1, tzinfo=UTC)
        stream_end = stream_start + timedelta(days=11)  # train + 2*test
        windows = list(spec.iter_windows(stream_start, stream_end))
        assert len(windows) == 2
        # adjacency: test_end of window 0 equals test_start of window 1.
        assert windows[0][3] == windows[1][2]
        # rolling training: train_start rolls forward by test_duration.
        assert windows[1][0] == windows[0][0] + spec.test_duration

    def test_trailing_partial_dropped(self) -> None:
        spec = self._spec()
        stream_start = datetime(2024, 1, 1, tzinfo=UTC)
        # train + 2.5 * test: 5 + 7.5 = 12.5 days. Third window would
        # require test_end at day 5 + 9 = 14, past the 12.5 mark.
        stream_end = stream_start + timedelta(days=12, hours=12)
        windows = list(spec.iter_windows(stream_start, stream_end))
        assert len(windows) == 2
        assert windows[-1][3] == stream_start + timedelta(days=11)

    def test_stream_too_short_yields_no_windows(self) -> None:
        spec = self._spec()
        stream_start = datetime(2024, 1, 1, tzinfo=UTC)
        # One second shy of train + test.
        stream_end = stream_start + timedelta(days=8) - timedelta(seconds=1)
        assert list(spec.iter_windows(stream_start, stream_end)) == []

    def test_boundary_exact_end_is_kept(self) -> None:
        spec = self._spec()
        stream_start = datetime(2024, 1, 1, tzinfo=UTC)
        # test_end lands exactly on stream_end -> accepted (<= check).
        stream_end = stream_start + timedelta(days=8)
        windows = list(spec.iter_windows(stream_start, stream_end))
        assert len(windows) == 1

    def test_train_windows_roll_not_expand(self) -> None:
        spec = self._spec()
        stream_start = datetime(2024, 1, 1, tzinfo=UTC)
        stream_end = stream_start + timedelta(days=14)  # 4 windows fit
        windows = list(spec.iter_windows(stream_start, stream_end))
        # Each window's train length is fixed; no expansion.
        for tr_s, tr_e, _, _ in windows:
            assert tr_e - tr_s == spec.train_duration


# ==========================================================================
# Event partition semantics (half-open interval)
# ==========================================================================


class TestEventPartition:
    def _events(self) -> tuple[tuple[Event, ...], list[datetime]]:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        events = tuple(_odds(f"M{i}", base + timedelta(days=i)) for i in range(5))
        timestamps = [e.timestamp for e in events]
        return events, timestamps

    def test_half_open_includes_lo_excludes_hi(self) -> None:
        events, timestamps = self._events()
        base = events[0].timestamp
        result = WalkForwardEvaluator._slice_by_timestamp(
            events, timestamps, base + timedelta(days=1), base + timedelta(days=3)
        )
        ids = [e.snapshot.match_id for e in result if isinstance(e, OddsAvailable)]
        assert ids == ["M1", "M2"]

    def test_empty_slice_when_range_between_events(self) -> None:
        events, timestamps = self._events()
        base = events[0].timestamp
        result = WalkForwardEvaluator._slice_by_timestamp(
            events,
            timestamps,
            base + timedelta(days=1, hours=1),
            base + timedelta(days=1, hours=23),
        )
        assert result == ()

    def test_entire_range(self) -> None:
        events, timestamps = self._events()
        base = events[0].timestamp
        result = WalkForwardEvaluator._slice_by_timestamp(
            events, timestamps, base, base + timedelta(days=10)
        )
        assert len(result) == len(events)


# ==========================================================================
# Evaluator constructor validation
# ==========================================================================


class TestEvaluatorInit:
    def _defaults(self) -> dict[str, object]:
        return {
            "event_source": _sorted_source([]),
            "strategy_factory": _HomeBacker,
            "commission_model": NetWinningsCommission(rate=0.0),
            "window_spec": WindowSpec(
                train_duration=timedelta(days=1),
                test_duration=timedelta(days=1),
            ),
            "starting_bankroll": 1000.0,
            "seed": 0,
        }

    def test_rejects_non_finite_bankroll(self) -> None:
        kwargs = self._defaults() | {"starting_bankroll": math.inf}
        with pytest.raises(ValueError, match="finite"):
            WalkForwardEvaluator(**kwargs)  # type: ignore[arg-type]

    def test_rejects_zero_bankroll(self) -> None:
        kwargs = self._defaults() | {"starting_bankroll": 0.0}
        with pytest.raises(ValueError, match="positive"):
            WalkForwardEvaluator(**kwargs)  # type: ignore[arg-type]

    def test_rejects_negative_bankroll(self) -> None:
        kwargs = self._defaults() | {"starting_bankroll": -1.0}
        with pytest.raises(ValueError, match="positive"):
            WalkForwardEvaluator(**kwargs)  # type: ignore[arg-type]

    def test_rejects_zero_n_resamples(self) -> None:
        kwargs = self._defaults() | {"n_resamples": 0}
        with pytest.raises(ValueError, match="n_resamples"):
            WalkForwardEvaluator(**kwargs)  # type: ignore[arg-type]

    def test_rejects_confidence_zero(self) -> None:
        kwargs = self._defaults() | {"confidence": 0.0}
        with pytest.raises(ValueError, match="confidence"):
            WalkForwardEvaluator(**kwargs)  # type: ignore[arg-type]

    def test_rejects_confidence_one(self) -> None:
        kwargs = self._defaults() | {"confidence": 1.0}
        with pytest.raises(ValueError, match="confidence"):
            WalkForwardEvaluator(**kwargs)  # type: ignore[arg-type]


# ==========================================================================
# Degenerate-stream guards
# ==========================================================================


class TestEmptyStream:
    def test_empty_stream_raises(self) -> None:
        evaluator = WalkForwardEvaluator(
            event_source=_sorted_source([]),
            strategy_factory=_HomeBacker,
            commission_model=NetWinningsCommission(rate=0.0),
            window_spec=WindowSpec(
                train_duration=timedelta(days=1),
                test_duration=timedelta(days=1),
            ),
            starting_bankroll=1000.0,
            seed=0,
        )
        with pytest.raises(ValueError, match="empty"):
            evaluator.run()


class TestStreamTooShort:
    def test_duration_below_train_plus_test_raises(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        events: list[Event] = [
            _odds("A", base),
            _settled("A", base + timedelta(days=1)),
        ]
        evaluator = WalkForwardEvaluator(
            event_source=_sorted_source(events),
            strategy_factory=_HomeBacker,
            commission_model=NetWinningsCommission(rate=0.0),
            window_spec=WindowSpec(
                train_duration=timedelta(days=5),
                test_duration=timedelta(days=5),
            ),
            starting_bankroll=1000.0,
            seed=0,
        )
        with pytest.raises(ValueError, match="no full"):
            evaluator.run()


# ==========================================================================
# End-to-end against SyntheticGenerator
# ==========================================================================


class TestEndToEndSynthetic:
    """40 matches at 1-day spacing, train=10d/test=10d, expect 2 windows
    with roughly 10 bets each."""

    def _evaluator(self, seed: int = 42) -> WalkForwardEvaluator:
        return WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=40, seed=7),
            strategy_factory=lambda: FavouriteBacker(stake=1.0),
            commission_model=NetWinningsCommission(rate=0.05),
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=seed,
        )

    def test_window_count(self) -> None:
        result = self._evaluator().run()
        assert len(result.per_window) == 2

    def test_each_window_has_bets(self) -> None:
        result = self._evaluator().run()
        for window in result.per_window:
            assert window.result.summary_metrics.n_bets > 0

    def test_aggregate_ledger_concatenates_per_window(self) -> None:
        result = self._evaluator().run()
        expected: list[object] = []
        for window in result.per_window:
            expected.extend(window.result.ledger)
        assert list(result.aggregate_ledger) == expected

    def test_aggregate_summary_matches_from_raw_over_concatenated(self) -> None:
        result = self._evaluator().run()
        reconstructed = BacktestResult.from_raw(
            RawBacktestOutput(
                ledger=result.aggregate_ledger,
                rejections=result.aggregate_rejections,
            ),
            starting_bankroll=result.starting_bankroll,
            t0=result.per_window[0].test_start,
        )
        assert result.aggregate_summary == reconstructed.summary_metrics
        assert result.aggregate_equity_curve == reconstructed.equity_curve

    def test_window_provenance_echoed(self) -> None:
        result = self._evaluator(seed=17).run()
        assert result.seed == 17
        assert result.window_spec.train_duration == timedelta(days=10)
        assert result.window_spec.test_duration == timedelta(days=10)
        assert result.starting_bankroll == 1000.0


# ==========================================================================
# Equity chaining
# ==========================================================================


class TestEquityChaining:
    def _evaluator(self) -> WalkForwardEvaluator:
        return WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=40),
            strategy_factory=lambda: FavouriteBacker(stake=1.0),
            commission_model=NetWinningsCommission(rate=0.05),
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=3,
        )

    def test_window_bankrolls_chain(self) -> None:
        result = self._evaluator().run()
        for prev, nxt in zip(result.per_window[:-1], result.per_window[1:]):
            # Each window's t0 baseline carries the prior window's last bankroll.
            assert nxt.result.starting_bankroll == prev.result.equity_curve[-1].bankroll

    def test_aggregate_curve_starts_at_starting_bankroll(self) -> None:
        result = self._evaluator().run()
        first = result.aggregate_equity_curve[0]
        assert first.bankroll == result.starting_bankroll
        assert first.realised_pnl == 0.0
        assert first.cumulative_turnover == 0.0

    def test_aggregate_curve_first_ts_is_first_window_test_start(self) -> None:
        result = self._evaluator().run()
        assert (
            result.aggregate_equity_curve[0].timestamp
            == result.per_window[0].test_start
        )

    def test_aggregate_curve_cumulative_turnover_nondecreasing(self) -> None:
        result = self._evaluator().run()
        turnovers = [p.cumulative_turnover for p in result.aggregate_equity_curve]
        for prev, nxt in zip(turnovers[:-1], turnovers[1:]):
            assert nxt >= prev


# ==========================================================================
# Determinism (same seed -> byte-identical)
# ==========================================================================


class TestDeterminism:
    """Seed-sensitivity (different seed -> different result) is deferred
    to module 9. v1 strategies have no RNG, so a meaningful test needs
    an RNG-using strategy (xG/Poisson); writing a mock for it here would
    test the mock rather than the evaluator."""

    def _evaluator(self) -> WalkForwardEvaluator:
        return WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=40),
            strategy_factory=lambda: FavouriteBacker(stake=1.0),
            commission_model=NetWinningsCommission(rate=0.05),
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=12345,
        )

    def test_same_inputs_byte_identical(self) -> None:
        r1 = self._evaluator().run()
        r2 = self._evaluator().run()
        assert r1 == r2

    def test_repeated_run_on_same_evaluator(self) -> None:
        evaluator = self._evaluator()
        r1 = evaluator.run()
        r2 = evaluator.run()
        assert r1 == r2


# ==========================================================================
# Per-window seed derivation
# ==========================================================================


class TestSeedDerivation:
    def test_per_window_seed_is_evaluator_seed_plus_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from betting_backtester import walk_forward as wf_module

        captured: list[int] = []

        class RecordingBacktester(Backtester):
            def __init__(
                self,
                *,
                event_source: EventSource,
                strategy: Strategy,
                commission_model: CommissionModel,
                starting_bankroll: float,
                seed: int,
                strict_settlement: bool = True,
            ) -> None:
                captured.append(seed)
                super().__init__(
                    event_source=event_source,
                    strategy=strategy,
                    commission_model=commission_model,
                    starting_bankroll=starting_bankroll,
                    seed=seed,
                    strict_settlement=strict_settlement,
                )

        monkeypatch.setattr(wf_module, "Backtester", RecordingBacktester)

        evaluator = WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=40),
            strategy_factory=lambda: FavouriteBacker(stake=1.0),
            commission_model=NetWinningsCommission(rate=0.05),
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=100,
        )
        result = evaluator.run()

        assert captured == [100 + i for i in range(len(result.per_window))]


# ==========================================================================
# Factory freshness
# ==========================================================================


class TestFactoryFreshness:
    def test_one_fresh_instance_per_window(self) -> None:
        factory, instances = _home_backer_factory()
        evaluator = WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=40),
            strategy_factory=factory,
            commission_model=NetWinningsCommission(rate=0.0),
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=1,
        )
        result = evaluator.run()

        assert len(instances) == len(result.per_window)
        # Each instance is distinct.
        assert len({id(inst) for inst in instances}) == len(instances)
        # Each instance had fit called exactly once.
        for inst in instances:
            assert inst.fit_calls == 1


# ==========================================================================
# Fit history semantics
# ==========================================================================


class TestFit:
    def test_fit_history_confined_to_train_window(self) -> None:
        factory, instances = _home_backer_factory()
        evaluator = WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=40),
            strategy_factory=factory,
            commission_model=NetWinningsCommission(rate=0.0),
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=2,
        )
        result = evaluator.run()

        for window, strategy in zip(result.per_window, instances):
            history = strategy.fit_history
            assert len(history) == window.n_train_events
            for event in history:
                assert window.train_start <= event.timestamp < window.train_end
            # stream-order preservation.
            for prev, nxt in zip(history[:-1], history[1:]):
                assert stream_sort_key(prev) <= stream_sort_key(nxt)

    def test_fit_history_includes_both_event_types(self) -> None:
        factory, instances = _home_backer_factory()
        evaluator = WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=40),
            strategy_factory=factory,
            commission_model=NetWinningsCommission(rate=0.0),
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=2,
        )
        evaluator.run()

        # Window 1's training window is fully populated with both odds
        # and settlements from window 0's test range.
        window_1_history = instances[1].fit_history
        has_odds = any(isinstance(e, OddsAvailable) for e in window_1_history)
        has_settled = any(isinstance(e, MatchSettled) for e in window_1_history)
        assert has_odds and has_settled


# ==========================================================================
# Trailing-partial drop
# ==========================================================================


class TestTrailingPartial:
    def test_trailing_events_absent_from_aggregate(self) -> None:
        # 45 matches -> stream ~44 days. train=10d, test=10d.
        # first_test_start at day ~10, last test_end at day 40.
        # Matches 40..44 fall past the last full test window.
        factory, _ = _home_backer_factory()
        evaluator = WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=45),
            strategy_factory=factory,
            commission_model=NetWinningsCommission(rate=0.0),
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=5,
        )
        result = evaluator.run()

        last_test_end = result.per_window[-1].test_end
        for bet in result.aggregate_ledger:
            assert bet.settled_at < last_test_end
            assert bet.placed_at < last_test_end


# ==========================================================================
# Boundary exclusion (match straddles adjacent windows)
# ==========================================================================


class TestBoundaryExclusion:
    def test_straddling_match_excluded_from_both_windows(self) -> None:
        # Windows: train=5d, test=5d, first_test_start = stream_start + 5d.
        # Stream starts at 2024-01-01 00:00; W0 = [2024-01-06, 2024-01-11),
        # W1 = [2024-01-11, 2024-01-16).
        #
        # Match A: both events in W0 (included).
        # Match B: odds in W0, settlement in W1 (STRADDLER, excluded).
        # Match C: both events in W1 (included).
        # Match T: both events in training period (not in any cohort).
        # Match Z: extends stream_end past W1 (not in any cohort).
        train = _odds("T", datetime(2024, 1, 1, tzinfo=UTC))
        train_settled = _settled("T", datetime(2024, 1, 1, 2, tzinfo=UTC))
        a_odds = _odds("A", datetime(2024, 1, 7, tzinfo=UTC))
        a_settled = _settled("A", datetime(2024, 1, 7, 2, tzinfo=UTC))
        b_odds = _odds("B", datetime(2024, 1, 10, 23, tzinfo=UTC))
        b_settled = _settled("B", datetime(2024, 1, 11, 1, tzinfo=UTC))
        c_odds = _odds("C", datetime(2024, 1, 12, tzinfo=UTC))
        c_settled = _settled("C", datetime(2024, 1, 12, 2, tzinfo=UTC))
        z_odds = _odds("Z", datetime(2024, 1, 16, 0, 30, tzinfo=UTC))
        z_settled = _settled("Z", datetime(2024, 1, 16, 2, tzinfo=UTC))

        source = _sorted_source(
            [
                train,
                train_settled,
                a_odds,
                a_settled,
                b_odds,
                b_settled,
                c_odds,
                c_settled,
                z_odds,
                z_settled,
            ]
        )

        evaluator = WalkForwardEvaluator(
            event_source=source,
            strategy_factory=lambda: _HomeBacker(stake=1.0),
            commission_model=NetWinningsCommission(rate=0.0),
            window_spec=WindowSpec(
                train_duration=timedelta(days=5),
                test_duration=timedelta(days=5),
            ),
            starting_bankroll=1000.0,
            seed=0,
        )
        result = evaluator.run()

        assert len(result.per_window) == 2
        assert result.n_matches_boundary_excluded == 1

        ledger_match_ids = {bet.match_id for bet in result.aggregate_ledger}
        assert ledger_match_ids == {"A", "C"}
        assert "B" not in ledger_match_ids
        assert "T" not in ledger_match_ids
        assert "Z" not in ledger_match_ids

        # Per-window breakdown.
        w0_ids = {bet.match_id for bet in result.per_window[0].result.ledger}
        w1_ids = {bet.match_id for bet in result.per_window[1].result.ledger}
        assert w0_ids == {"A"}
        assert w1_ids == {"C"}

    def test_zero_boundary_excluded_when_no_straddlers(self) -> None:
        result = WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=40),
            strategy_factory=lambda: FavouriteBacker(stake=1.0),
            commission_model=NetWinningsCommission(rate=0.05),
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=0,
        ).run()
        # Synthetic fixtures are 1-day apart with 2h settlement; windows
        # align to minutes, so no match straddles.
        assert result.n_matches_boundary_excluded == 0


# ==========================================================================
# Duplicate events
# ==========================================================================


class TestDuplicateEvents:
    def _window_spec(self) -> WindowSpec:
        return WindowSpec(
            train_duration=timedelta(days=3),
            test_duration=timedelta(days=3),
        )

    def test_duplicate_odds_raises(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        events: list[Event] = [
            _odds("A", base),
            _odds("A", base + timedelta(hours=1)),
            _settled("A", base + timedelta(days=10)),
        ]
        evaluator = WalkForwardEvaluator(
            event_source=_sorted_source(events),
            strategy_factory=lambda: _HomeBacker(),
            commission_model=NetWinningsCommission(rate=0.0),
            window_spec=self._window_spec(),
            starting_bankroll=1000.0,
            seed=0,
        )
        with pytest.raises(ValueError, match="multiple OddsAvailable"):
            evaluator.run()

    def test_duplicate_settled_raises(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        events: list[Event] = [
            _odds("A", base),
            _settled("A", base + timedelta(hours=1)),
            _settled("A", base + timedelta(days=10)),
        ]
        evaluator = WalkForwardEvaluator(
            event_source=_sorted_source(events),
            strategy_factory=lambda: _HomeBacker(),
            commission_model=NetWinningsCommission(rate=0.0),
            window_spec=self._window_spec(),
            starting_bankroll=1000.0,
            seed=0,
        )
        with pytest.raises(ValueError, match="multiple MatchSettled"):
            evaluator.run()


# ==========================================================================
# Aggregate YieldCI reproducibility
# ==========================================================================


class TestYieldCI:
    def test_aggregate_ci_matches_external_recomputation(self) -> None:
        evaluator = WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=40),
            strategy_factory=lambda: FavouriteBacker(stake=1.0),
            commission_model=NetWinningsCommission(rate=0.05),
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=9876,
            n_resamples=2_000,
            confidence=0.9,
        )
        result = evaluator.run()

        aggregate_result = BacktestResult.from_raw(
            RawBacktestOutput(
                ledger=result.aggregate_ledger,
                rejections=result.aggregate_rejections,
            ),
            starting_bankroll=result.starting_bankroll,
            t0=result.per_window[0].test_start,
        )
        expected = compute_yield_ci(
            aggregate_result,
            seed=evaluator.seed,
            n_resamples=2_000,
            confidence=0.9,
        )
        assert result.aggregate_yield_ci == expected


# ==========================================================================
# Commission instance sharing
# ==========================================================================


class TestCommissionShared:
    def test_same_commission_instance_across_windows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from betting_backtester import walk_forward as wf_module

        captured: list[CommissionModel] = []

        class RecordingBacktester(Backtester):
            def __init__(
                self,
                *,
                event_source: EventSource,
                strategy: Strategy,
                commission_model: CommissionModel,
                starting_bankroll: float,
                seed: int,
                strict_settlement: bool = True,
            ) -> None:
                captured.append(commission_model)
                super().__init__(
                    event_source=event_source,
                    strategy=strategy,
                    commission_model=commission_model,
                    starting_bankroll=starting_bankroll,
                    seed=seed,
                    strict_settlement=strict_settlement,
                )

        monkeypatch.setattr(wf_module, "Backtester", RecordingBacktester)

        commission = NetWinningsCommission(rate=0.05)
        evaluator = WalkForwardEvaluator(
            event_source=_synthetic_source(n_matches=40),
            strategy_factory=lambda: FavouriteBacker(stake=1.0),
            commission_model=commission,
            window_spec=WindowSpec(
                train_duration=timedelta(days=10),
                test_duration=timedelta(days=10),
            ),
            starting_bankroll=1000.0,
            seed=0,
        )
        evaluator.run()

        assert len(captured) >= 2
        for instance in captured:
            assert instance is commission
