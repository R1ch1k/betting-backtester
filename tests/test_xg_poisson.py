"""Tests for :class:`betting_backtester.strategies.xg_poisson.XgPoissonStrategy`.

Organisation:

* ``TestConstruction`` -- match_directory and numeric parameter validation,
  bankroll_basis label guard, initial state.
* ``TestReadOnlyProperties`` -- provenance properties echo the
  constructor args.
* ``TestStrategyProtocol`` -- runtime-checkable protocol conformance
  and ``__slots__`` enforcement.
* ``TestEmptyHistoryDegradation`` -- fit on empty or settled-free
  history sets model to None; subsequent on_odds returns [].
* ``TestFitProcessing`` -- OddsAvailable events ignored; unseen-in-
  directory settled events dropped from training; unseen_skips reset.
* ``TestOnOddsBackRule`` -- scenario where our prob exceeds market
  implied by more than the edge threshold -> BACK.
* ``TestOnOddsLayRule`` -- scenario where our prob undercuts the market
  implied -> LAY (on the opposing selection of the same match).
* ``TestOnOddsNoBet`` -- high edge_threshold suppresses all orders.
* ``TestUnseenSkipping`` -- snapshot with unknown match_id or unseen
  teams -> [] and unseen_skips += 1; fit-time drops do NOT count.
* ``TestBankrollBasis`` -- realised_wealth vs available_cash basis
  produce different stakes when the two bankroll figures diverge.
* ``TestDeterminism`` -- on_odds is byte-deterministic across runs.
* ``TestSyntheticIntegration`` -- end-to-end with a recurring-teams
  stream + the real :class:`Backtester`.
* ``TestWalkForwardLookahead`` -- under a regime change between train
  and test windows, the strategy's bets reflect the *training*
  strengths, not leaked test truth.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from betting_backtester._event_ordering import stream_sort_key
from betting_backtester.backtest_result import BacktestResult
from betting_backtester.backtester import (
    Backtester,
    BetOrder,
    PortfolioView,
    Side,
    Strategy,
)
from betting_backtester.commission import NetWinningsCommission
from betting_backtester.models import (
    Event,
    Match,
    MatchResult,
    MatchSettled,
    OddsAvailable,
    OddsSnapshot,
    Selection,
    SelectionOdds,
)
from betting_backtester.strategies.xg_poisson import XgPoissonStrategy
from betting_backtester.walk_forward import (
    WalkForwardEvaluator,
    WindowSpec,
)

UTC = timezone.utc


# ---------- event/fixture helpers -----------------------------------------


@dataclass(frozen=True)
class _ListSource:
    """Tiny test-only ``EventSource`` over a pre-sorted tuple of events."""

    events_tuple: tuple[Event, ...]

    def events(self) -> Iterator[Event]:
        return iter(self.events_tuple)


def _build_stream(
    fixtures: list[tuple[str, str, int, int, int]],
    *,
    t0: datetime = datetime(2024, 1, 1, tzinfo=UTC),
    odds: tuple[float, float, float] = (3.0, 3.0, 3.0),
    id_prefix: str = "M",
) -> tuple[tuple[Event, ...], dict[str, Match]]:
    """Materialise a synthetic event stream + directory.

    Each fixture tuple is ``(home, away, home_goals, away_goals,
    day_offset)``. Odds are constant across every match for easy
    edge-threshold reasoning.
    """
    events: list[Event] = []
    directory: dict[str, Match] = {}
    for idx, (home, away, hg, ag, day) in enumerate(fixtures):
        kickoff = t0 + timedelta(days=day, hours=15)
        match_id = f"{id_prefix}-{idx:04d}"
        match = Match(
            match_id=match_id,
            kickoff=kickoff,
            league="TST",
            season="2024-25",
            home=home,
            away=away,
        )
        directory[match_id] = match
        events.append(
            OddsAvailable(
                snapshot=OddsSnapshot(
                    match_id=match_id,
                    timestamp=kickoff - timedelta(minutes=5),
                    home=SelectionOdds(back_price=odds[0], lay_price=odds[0]),
                    draw=SelectionOdds(back_price=odds[1], lay_price=odds[1]),
                    away=SelectionOdds(back_price=odds[2], lay_price=odds[2]),
                )
            )
        )
        events.append(
            MatchSettled(
                result=MatchResult(
                    match_id=match_id,
                    timestamp=kickoff + timedelta(hours=2),
                    home_goals=hg,
                    away_goals=ag,
                )
            )
        )
    events.sort(key=stream_sort_key)
    return tuple(events), directory


def _dominator_training_fixtures(
    *,
    n_rounds: int = 10,
    opponents: tuple[str, ...] = ("O0", "O1", "O2", "O3"),
    dom_wins: bool = True,
) -> list[tuple[str, str, int, int, int]]:
    """DOMINATOR vs each opponent, alternating home and away, one per day.

    ``dom_wins`` controls the outcome: 3-0 in DOMINATOR's favour when
    True, 0-3 against when False.
    """
    fixtures: list[tuple[str, str, int, int, int]] = []
    day = 0
    for r in range(n_rounds):
        for opp in opponents:
            dom_home = (r + opponents.index(opp)) % 2 == 0
            if dom_home:
                home, away = "DOM", opp
                hg, ag = (3, 0) if dom_wins else (0, 3)
            else:
                home, away = opp, "DOM"
                hg, ag = (0, 3) if dom_wins else (3, 0)
            fixtures.append((home, away, hg, ag, day))
            day += 1
    return fixtures


def _portfolio(
    *,
    available: float = 1000.0,
    starting: float = 1000.0,
    realised: float = 0.0,
    open_bets: int = 0,
) -> PortfolioView:
    return PortfolioView(
        available_bankroll=available,
        starting_bankroll=starting,
        open_bets_count=open_bets,
        realised_pnl=realised,
    )


def _snapshot(
    *,
    match_id: str,
    timestamp: datetime,
    odds: tuple[float, float, float] = (3.0, 3.0, 3.0),
) -> OddsSnapshot:
    return OddsSnapshot(
        match_id=match_id,
        timestamp=timestamp,
        home=SelectionOdds(back_price=odds[0], lay_price=odds[0]),
        draw=SelectionOdds(back_price=odds[1], lay_price=odds[1]),
        away=SelectionOdds(back_price=odds[2], lay_price=odds[2]),
    )


def _minimal_directory() -> dict[str, Match]:
    """A one-entry directory for tests that just need construction to succeed."""
    return {
        "M-0000": Match(
            match_id="M-0000",
            kickoff=datetime(2024, 1, 1, 15, tzinfo=UTC),
            league="TST",
            season="2024-25",
            home="A",
            away="B",
        )
    }


# ---------- construction --------------------------------------------------


class TestConstruction:
    def test_valid_construction_initialises_unfitted(self) -> None:
        s = XgPoissonStrategy(match_directory=_minimal_directory())
        assert s.model is None
        assert s.unseen_skips == 0

    def test_non_mapping_directory_raises(self) -> None:
        with pytest.raises(TypeError, match="Mapping"):
            XgPoissonStrategy(match_directory=[])  # type: ignore[arg-type]

    def test_empty_directory_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            XgPoissonStrategy(match_directory={})

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, math.nan, math.inf])
    def test_invalid_edge_threshold_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="edge_threshold"):
            XgPoissonStrategy(
                match_directory=_minimal_directory(), edge_threshold=bad
            )

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.1, math.nan, math.inf])
    def test_invalid_kelly_fraction_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="kelly_fraction"):
            XgPoissonStrategy(
                match_directory=_minimal_directory(), kelly_fraction=bad
            )

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.1, math.nan, math.inf])
    def test_invalid_max_exposure_fraction_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="max_exposure_fraction"):
            XgPoissonStrategy(
                match_directory=_minimal_directory(),
                max_exposure_fraction=bad,
            )

    @pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
    def test_invalid_l2_reg_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="l2_reg"):
            XgPoissonStrategy(match_directory=_minimal_directory(), l2_reg=bad)

    @pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
    def test_invalid_decay_rate_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="decay_rate"):
            XgPoissonStrategy(
                match_directory=_minimal_directory(), decay_rate=bad
            )

    def test_invalid_bankroll_basis_raises(self) -> None:
        with pytest.raises(ValueError, match="bankroll_basis"):
            XgPoissonStrategy(
                match_directory=_minimal_directory(),
                bankroll_basis="wealth",  # type: ignore[arg-type]
            )


# ---------- read-only properties ------------------------------------------


class TestReadOnlyProperties:
    def test_properties_echo_constructor_args(self) -> None:
        s = XgPoissonStrategy(
            match_directory=_minimal_directory(),
            edge_threshold=0.05,
            kelly_fraction=0.5,
            max_exposure_fraction=0.1,
            l2_reg=0.01,
            decay_rate=0.003,
            bankroll_basis="available_cash",
        )
        assert s.edge_threshold == 0.05
        assert s.kelly_fraction == 0.5
        assert s.max_exposure_fraction == 0.1
        assert s.l2_reg == 0.01
        assert s.decay_rate == 0.003
        assert s.bankroll_basis == "available_cash"

    def test_model_property_is_none_before_fit(self) -> None:
        s = XgPoissonStrategy(match_directory=_minimal_directory())
        assert s.model is None


# ---------- protocol conformance ------------------------------------------


class TestStrategyProtocol:
    def test_satisfies_runtime_checkable_strategy(self) -> None:
        s = XgPoissonStrategy(match_directory=_minimal_directory())
        assert isinstance(s, Strategy)

    def test_slots_reject_arbitrary_attributes(self) -> None:
        s = XgPoissonStrategy(match_directory=_minimal_directory())
        with pytest.raises(AttributeError):
            s.extra = 1  # type: ignore[attr-defined]


# ---------- empty-history graceful degradation ----------------------------


class TestEmptyHistoryDegradation:
    def test_fit_empty_history_sets_model_to_none(self) -> None:
        s = XgPoissonStrategy(match_directory=_minimal_directory())
        s.fit([])
        assert s.model is None

    def test_fit_only_odds_events_sets_model_to_none(self) -> None:
        events, directory = _build_stream([("A", "B", 1, 0, 0)])
        odds_only = tuple(e for e in events if isinstance(e, OddsAvailable))
        s = XgPoissonStrategy(match_directory=directory)
        s.fit(odds_only)
        assert s.model is None

    def test_on_odds_after_empty_fit_returns_empty_list(self) -> None:
        events, directory = _build_stream([("A", "B", 1, 0, 0)])
        s = XgPoissonStrategy(match_directory=directory)
        s.fit([])
        snap = _snapshot(match_id="M-0000", timestamp=datetime(2024, 1, 2, tzinfo=UTC))
        assert s.on_odds(snap, _portfolio()) == []

    def test_successful_refit_after_empty_fit_recovers_model(self) -> None:
        events, directory = _build_stream(_dominator_training_fixtures(n_rounds=5))
        s = XgPoissonStrategy(match_directory=directory)
        s.fit([])
        assert s.model is None
        s.fit(events)
        assert s.model is not None
        assert s.model.is_fitted is True


# ---------- fit processing ------------------------------------------------


class TestFitProcessing:
    def test_odds_events_ignored_during_fit(self) -> None:
        # Fit on full stream vs settled-only stream -> same model
        # parameters. Tests that OddsAvailable events are ignored.
        events, directory = _build_stream(_dominator_training_fixtures(n_rounds=5))
        settled_only = tuple(e for e in events if isinstance(e, MatchSettled))

        s_full = XgPoissonStrategy(match_directory=directory)
        s_full.fit(events)
        s_settled = XgPoissonStrategy(match_directory=directory)
        s_settled.fit(settled_only)

        # anchor_time differs between runs (max timestamp in each
        # filtered stream), so the decay weights differ minutely. With
        # decay_rate set to 0, the two fits become identical.
        s_full2 = XgPoissonStrategy(match_directory=directory, decay_rate=0.0)
        s_full2.fit(events)
        s_settled2 = XgPoissonStrategy(
            match_directory=directory, decay_rate=0.0
        )
        s_settled2.fit(settled_only)
        m_full = s_full2.model
        m_settled = s_settled2.model
        assert m_full is not None and m_settled is not None
        assert m_full.known_teams() == m_settled.known_teams()
        assert m_full.predict("DOM", "O0") == m_settled.predict("DOM", "O0")

    def test_settled_match_missing_from_directory_is_dropped(self) -> None:
        # Two matches: one whose match_id is in the directory, one
        # whose match_id is not. The missing one is dropped from the
        # training set; unseen_skips stays 0 (it only counts inference).
        events, directory = _build_stream(
            [("A", "B", 1, 0, 0), ("C", "D", 2, 1, 1)]
        )
        # Remove the second match from the directory only.
        trimmed: dict[str, Match] = {
            k: v for k, v in directory.items() if k != "M-0001"
        }
        s = XgPoissonStrategy(match_directory=trimmed)
        s.fit(events)
        # Model was still fit on the remaining match.
        model = s.model
        assert model is not None
        assert model.known_teams() == frozenset({"A", "B"})
        # Per the docstring contract: training-time drops do not bump
        # the inference-time counter.
        assert s.unseen_skips == 0

    def test_fit_resets_unseen_skips(self) -> None:
        events, directory = _build_stream(_dominator_training_fixtures(n_rounds=5))
        s = XgPoissonStrategy(match_directory=directory)
        s.fit(events)
        # Trigger an inference-time unseen skip.
        mystery = _snapshot(
            match_id="not-in-directory",
            timestamp=datetime(2024, 2, 1, tzinfo=UTC),
        )
        s.on_odds(mystery, _portfolio())
        assert s.unseen_skips == 1
        # Re-fit resets the counter.
        s.fit(events)
        assert s.unseen_skips == 0


# ---------- on_odds: back rule -------------------------------------------


class TestOnOddsBackRule:
    def test_back_order_emitted_on_strong_favourite(self) -> None:
        # Train: DOM wins 3-0 every match. Model will give DOM home a
        # probability well above the neutral-market-implied 0.33.
        events, directory = _build_stream(
            _dominator_training_fixtures(n_rounds=10)
        )
        s = XgPoissonStrategy(match_directory=directory)
        s.fit(events)

        # Test snapshot: DOM at home vs O0, odds (3.0, 3.0, 3.0).
        kickoff = datetime(2024, 6, 1, 15, tzinfo=UTC)
        # Directory must contain the test match.
        directory["TEST-dom-home"] = Match(
            match_id="TEST-dom-home",
            kickoff=kickoff,
            league="TST",
            season="2024-25",
            home="DOM",
            away="O0",
        )
        snap = _snapshot(
            match_id="TEST-dom-home",
            timestamp=kickoff - timedelta(minutes=5),
        )
        orders = s.on_odds(snap, _portfolio(available=1000.0))

        backs_on_home = [
            o
            for o in orders
            if o.side is Side.BACK and o.selection is Selection.HOME
        ]
        assert len(backs_on_home) == 1
        back = backs_on_home[0]
        assert back.match_id == "TEST-dom-home"
        assert back.price == 3.0
        assert 0.0 < back.stake <= 0.05 * 1000.0 + 1e-9  # exposure cap


# ---------- on_odds: lay rule --------------------------------------------


class TestOnOddsLayRule:
    def test_lay_order_emitted_on_strong_underdog(self) -> None:
        # Train: DOM wins 3-0 every match. Opponent is a strong under-
        # dog. Snapshot with DOM *away*: the opponent sits on HOME,
        # and we expect a LAY on HOME (since our_prob(HOME) undercuts
        # the market-implied 0.33 by more than the 0.02 edge threshold).
        events, directory = _build_stream(
            _dominator_training_fixtures(n_rounds=10)
        )
        s = XgPoissonStrategy(match_directory=directory)
        s.fit(events)

        kickoff = datetime(2024, 6, 1, 15, tzinfo=UTC)
        directory["TEST-dom-away"] = Match(
            match_id="TEST-dom-away",
            kickoff=kickoff,
            league="TST",
            season="2024-25",
            home="O0",
            away="DOM",
        )
        snap = _snapshot(
            match_id="TEST-dom-away",
            timestamp=kickoff - timedelta(minutes=5),
        )
        orders = s.on_odds(snap, _portfolio(available=1000.0))

        lays_on_home = [
            o
            for o in orders
            if o.side is Side.LAY and o.selection is Selection.HOME
        ]
        assert len(lays_on_home) == 1
        lay = lays_on_home[0]
        assert lay.price == 3.0
        assert lay.stake > 0.0


# ---------- on_odds: no-bet ----------------------------------------------


class TestOnOddsNoBet:
    def test_high_edge_threshold_suppresses_all_orders(self) -> None:
        events, directory = _build_stream(
            _dominator_training_fixtures(n_rounds=10)
        )
        s = XgPoissonStrategy(
            match_directory=directory, edge_threshold=0.95
        )
        s.fit(events)

        kickoff = datetime(2024, 6, 1, 15, tzinfo=UTC)
        directory["TEST-nobet"] = Match(
            match_id="TEST-nobet",
            kickoff=kickoff,
            league="TST",
            season="2024-25",
            home="DOM",
            away="O0",
        )
        snap = _snapshot(
            match_id="TEST-nobet",
            timestamp=kickoff - timedelta(minutes=5),
        )
        assert s.on_odds(snap, _portfolio()) == []

    def test_non_positive_bankroll_yields_no_orders(self) -> None:
        events, directory = _build_stream(
            _dominator_training_fixtures(n_rounds=10)
        )
        s = XgPoissonStrategy(match_directory=directory)
        s.fit(events)

        kickoff = datetime(2024, 6, 1, 15, tzinfo=UTC)
        directory["TEST-broke"] = Match(
            match_id="TEST-broke",
            kickoff=kickoff,
            league="TST",
            season="2024-25",
            home="DOM",
            away="O0",
        )
        snap = _snapshot(
            match_id="TEST-broke",
            timestamp=kickoff - timedelta(minutes=5),
        )
        # realised_wealth = starting + realised = 1000 - 1000 = 0 -> no bets.
        blown = _portfolio(available=0.0, starting=1000.0, realised=-1000.0)
        assert s.on_odds(snap, blown) == []


# ---------- unseen skipping ----------------------------------------------


class TestUnseenSkipping:
    def test_snapshot_with_unknown_match_id_increments_and_returns_empty(
        self,
    ) -> None:
        events, directory = _build_stream(
            _dominator_training_fixtures(n_rounds=5)
        )
        s = XgPoissonStrategy(match_directory=directory)
        s.fit(events)

        mystery = _snapshot(
            match_id="not-in-directory",
            timestamp=datetime(2024, 6, 1, 15, tzinfo=UTC),
        )
        assert s.on_odds(mystery, _portfolio()) == []
        assert s.unseen_skips == 1

    def test_snapshot_with_unseen_team_increments_and_returns_empty(self) -> None:
        # Directory contains a match whose teams do NOT appear in the
        # fitted training set -> model has no ratings for them.
        events, directory = _build_stream(
            _dominator_training_fixtures(n_rounds=5)
        )
        kickoff = datetime(2024, 6, 1, 15, tzinfo=UTC)
        directory["TEST-unseen-team"] = Match(
            match_id="TEST-unseen-team",
            kickoff=kickoff,
            league="TST",
            season="2024-25",
            home="MYSTERY_A",
            away="MYSTERY_B",
        )
        s = XgPoissonStrategy(match_directory=directory)
        s.fit(events)

        snap = _snapshot(
            match_id="TEST-unseen-team",
            timestamp=kickoff - timedelta(minutes=5),
        )
        assert s.on_odds(snap, _portfolio()) == []
        assert s.unseen_skips == 1

    def test_multiple_skips_accumulate(self) -> None:
        events, directory = _build_stream(
            _dominator_training_fixtures(n_rounds=5)
        )
        s = XgPoissonStrategy(match_directory=directory)
        s.fit(events)

        for i in range(3):
            snap = _snapshot(
                match_id=f"missing-{i}",
                timestamp=datetime(2024, 6, i + 1, 15, tzinfo=UTC),
            )
            s.on_odds(snap, _portfolio())
        assert s.unseen_skips == 3


# ---------- bankroll basis -----------------------------------------------


class TestBankrollBasis:
    def test_realised_wealth_vs_available_cash_produce_different_stakes(
        self,
    ) -> None:
        # Portfolio with available_cash far below realised_wealth:
        # starting 1000, realised +500 (so realised_wealth=1500) but
        # available_cash=200 (900 locked up in open commitments,
        # construct that picture).
        events, directory = _build_stream(
            _dominator_training_fixtures(n_rounds=10)
        )
        kickoff = datetime(2024, 6, 1, 15, tzinfo=UTC)
        directory["TEST-basis"] = Match(
            match_id="TEST-basis",
            kickoff=kickoff,
            league="TST",
            season="2024-25",
            home="DOM",
            away="O0",
        )
        snap = _snapshot(
            match_id="TEST-basis",
            timestamp=kickoff - timedelta(minutes=5),
        )
        port = _portfolio(
            available=200.0, starting=1000.0, realised=500.0, open_bets=3
        )

        s_wealth = XgPoissonStrategy(
            match_directory=directory, bankroll_basis="realised_wealth"
        )
        s_cash = XgPoissonStrategy(
            match_directory=directory, bankroll_basis="available_cash"
        )
        s_wealth.fit(events)
        s_cash.fit(events)

        orders_wealth = s_wealth.on_odds(snap, port)
        orders_cash = s_cash.on_odds(snap, port)

        # Both fire on the same selections (same predictions, same
        # odds); only the stakes differ, scaled by the bankroll base.
        assert len(orders_wealth) == len(orders_cash) > 0
        # realised_wealth (1500) is larger -> stakes are strictly
        # larger than those on available_cash (200).
        for w, c in zip(
            sorted(orders_wealth, key=lambda o: (o.selection, o.side)),
            sorted(orders_cash, key=lambda o: (o.selection, o.side)),
            strict=True,
        ):
            assert w.selection == c.selection
            assert w.side == c.side
            assert w.stake > c.stake


# ---------- determinism --------------------------------------------------


class TestDeterminism:
    def test_two_runs_emit_identical_orders(self) -> None:
        events, directory = _build_stream(
            _dominator_training_fixtures(n_rounds=10)
        )
        kickoff = datetime(2024, 6, 1, 15, tzinfo=UTC)
        directory["TEST-det"] = Match(
            match_id="TEST-det",
            kickoff=kickoff,
            league="TST",
            season="2024-25",
            home="DOM",
            away="O0",
        )
        snap = _snapshot(
            match_id="TEST-det",
            timestamp=kickoff - timedelta(minutes=5),
        )
        port = _portfolio()

        def run_once() -> list[BetOrder]:
            s = XgPoissonStrategy(match_directory=directory)
            s.fit(events)
            return s.on_odds(snap, port)

        assert run_once() == run_once()


# ---------- end-to-end with Backtester on synthetic recurring-teams ------


class TestSyntheticIntegration:
    def test_strategy_runs_through_backtester_and_places_bets(self) -> None:
        """End-to-end compositional smoke test: the strategy emits BetOrders
        that the Backtester accepts, settles, and records in a ledger.

        Does NOT test walk-forward training/test discipline -- that is
        covered separately in :class:`TestWalkForwardLookahead`.
        """
        # Recurring-teams stream so the strategy actually has known
        # teams by the time on_odds fires. Split into train (first 40
        # matches) and test (last 20 matches). Fit on train, run
        # backtester on the whole stream; bets only emit on matches
        # whose teams are known (the last 20).
        train_fixtures = _dominator_training_fixtures(n_rounds=10)
        # 10 additional test matches beyond the training window,
        # keeping the same opponents -> known teams.
        test_day_start = max(day for _, _, _, _, day in train_fixtures) + 1
        test_fixtures = [
            (
                "DOM" if i % 2 == 0 else f"O{i % 4}",
                f"O{i % 4}" if i % 2 == 0 else "DOM",
                1,
                2,
                test_day_start + i,
            )
            for i in range(10)
        ]
        all_fixtures = train_fixtures + test_fixtures
        events, directory = _build_stream(all_fixtures)

        train_cutoff = len(train_fixtures) * 2  # two events per fixture
        strategy = XgPoissonStrategy(match_directory=directory)
        strategy.fit(events[:train_cutoff])

        backtester = Backtester(
            event_source=_ListSource(events_tuple=events),
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=0.05),
            starting_bankroll=1000.0,
            seed=0,
        )
        raw = backtester.run()
        # Bets placed during the test portion (test matches have all
        # known teams and DOM predictions misalign with neutral odds).
        assert len(raw.ledger) > 0
        result = BacktestResult.from_raw(
            raw,
            starting_bankroll=1000.0,
            t0=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert result.summary_metrics.n_bets == len(raw.ledger)


# ---------- walk-forward lookahead composition ---------------------------


class TestWalkForwardLookahead:
    def test_predictions_use_training_strengths_not_test_truth(self) -> None:
        """Composition gate for the walk-forward evaluator.

        Scenario: a regime change between train and test windows.
        Training period says DOM is strong (wins 3-0 every match);
        test period says DOM is weak (loses 0-3 every match). The
        market prices every match at neutral odds (3.0 / 3.0 / 3.0).

        If the strategy composes correctly with the walk-forward
        evaluator, during on_odds calls in the test window it must
        see only training events (fit window ends before test_start),
        and its predictions therefore reflect DOM-is-strong. Every
        BACK order should land on the DOMINATOR side and the
        aggregate P&L should be negative (bets lose because the test
        truth diverges). A lookahead leak would produce the opposite:
        backs on the opponent side and positive P&L.
        """
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        opponents = ("O0", "O1", "O2", "O3", "O4")

        # Training window: days 0-199. 200 DOM matches, DOM wins 3-0.
        train_fixtures: list[tuple[str, str, int, int, int]] = []
        for day in range(200):
            opp = opponents[day % len(opponents)]
            dom_home = day % 2 == 0
            if dom_home:
                train_fixtures.append(("DOM", opp, 3, 0, day))
            else:
                train_fixtures.append((opp, "DOM", 0, 3, day))

        # Test window: days 200-224. 25 DOM matches, DOM loses 0-3.
        test_fixtures: list[tuple[str, str, int, int, int]] = []
        for i, day in enumerate(range(200, 225)):
            opp = opponents[i % len(opponents)]
            dom_home = i % 2 == 0
            if dom_home:
                test_fixtures.append(("DOM", opp, 0, 3, day))
            else:
                test_fixtures.append((opp, "DOM", 3, 0, day))

        # Sentinel match on day 230 between opponents, so stream_end
        # lies past test_end = day 230 and iter_windows yields the
        # window. Not in the test cohort (its kickoff is at day 230
        # 15:00, after test_end at day 230 00:00).
        sentinel = [("O0", "O1", 1, 1, 230)]

        all_fixtures = train_fixtures + test_fixtures + sentinel
        events, directory = _build_stream(all_fixtures, t0=t0, id_prefix="R")

        def factory() -> XgPoissonStrategy:
            return XgPoissonStrategy(match_directory=directory)

        evaluator = WalkForwardEvaluator(
            event_source=_ListSource(events_tuple=events),
            strategy_factory=factory,
            commission_model=NetWinningsCommission(rate=0.05),
            window_spec=WindowSpec(
                train_duration=timedelta(days=200),
                test_duration=timedelta(days=30),
            ),
            starting_bankroll=10_000.0,
            seed=0,
            n_resamples=200,  # keep the bootstrap cheap in a test
        )
        result = evaluator.run()

        # The walk-forward yields exactly one window covering the test
        # phase (days 200-230), with 25 matches in cohort.
        assert len(result.per_window) == 1
        assert result.aggregate_summary.n_bets > 0

        # Primary assertion: every BACK order sits on the DOMINATOR
        # side of its match. A lookahead leak would flip this.
        back_orders = [
            b for b in result.aggregate_ledger if b.side is Side.BACK
        ]
        assert len(back_orders) > 0
        for back in back_orders:
            match = directory[back.match_id]
            dominator_is_home = match.home == "DOM"
            bet_on_home = back.selection is Selection.HOME
            bet_on_away = back.selection is Selection.AWAY
            if dominator_is_home:
                assert bet_on_home, (
                    f"expected BACK on HOME (DOM side) for match "
                    f"{match.match_id}; got {back.selection}"
                )
            else:
                # DOM is away in this match.
                assert bet_on_away, (
                    f"expected BACK on AWAY (DOM side) for match "
                    f"{match.match_id}; got {back.selection}"
                )

        # Economic half of the claim: betting DOM under a regime
        # reversal loses money.
        assert result.aggregate_summary.net_pnl < 0.0, (
            f"expected negative net P&L under regime reversal; got "
            f"{result.aggregate_summary.net_pnl}"
        )
