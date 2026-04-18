"""Tests for the backtester core (module 4a).

Organisation:

* Type-level tests (``TestSide``, ``TestBetOrder``, ``TestPendingBet``, ...)
  exercise the Pydantic models and helpers in isolation.
* ``TestStrategyProtocol`` pins the protocol surface.
* ``TestBacktesterConstruction`` covers constructor validation.
* ``TestAccountingEndToEnd`` and ``TestArbitrageAccounting`` are the primary
  correctness gates: hand-computed expected ledgers, byte-equal asserts.
* ``TestRejections`` covers every rejection reason.
* ``TestLifecycle`` pins the call pattern (on_odds/on_settled ordering,
  ``fit`` not invoked, ``strict_settlement`` behaviour).
* ``TestDeterminism`` locks same-seed repeatability.
* ``TestConvergence`` is the statistical sanity check: zero-EV random walk
  stays inside a wide sigma bound.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from betting_backtester.backtester import (
    Backtester,
    BetOrder,
    PendingBet,
    PortfolioView,
    RawBacktestOutput,
    RejectedOrder,
    RejectionReason,
    SettledBet,
    Side,
    Strategy,
    committed_funds,
)
from betting_backtester.commission import NetWinningsCommission
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
from betting_backtester.synthetic import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
    TrueProbabilities,
)

UTC = timezone.utc


# ---------- shared test doubles --------------------------------------------


class ListEventSource:
    """Minimal :class:`EventSource` over a pre-built list. Re-iterable."""

    def __init__(self, events: list[Event]) -> None:
        self._events = list(events)

    def events(self) -> Iterator[Event]:
        return iter(self._events)


class ScriptedStrategy:
    """Records every callback and emits orders from a pre-built script.

    The script maps ``match_id -> list[BetOrder]`` so a test can stage exactly
    the orders it cares about per match without writing a bespoke strategy.
    """

    def __init__(
        self, orders_by_match: dict[str, list[BetOrder]] | None = None
    ) -> None:
        self._orders = orders_by_match or {}
        self.fit_calls: list[list[Event]] = []
        self.on_odds_calls: list[tuple[OddsSnapshot, PortfolioView]] = []
        self.on_settled_calls: list[tuple[MatchResult, PortfolioView]] = []

    def fit(self, history: Iterable[Event]) -> None:
        self.fit_calls.append(list(history))

    def on_odds(
        self, snapshot: OddsSnapshot, portfolio: PortfolioView
    ) -> list[BetOrder]:
        self.on_odds_calls.append((snapshot, portfolio))
        return list(self._orders.get(snapshot.match_id, []))

    def on_settled(self, result: MatchResult, portfolio: PortfolioView) -> None:
        self.on_settled_calls.append((result, portfolio))


def _uniform_snapshot(
    match_id: str, timestamp: datetime, back: float, lay: float | None = None
) -> OddsSnapshot:
    """A snapshot with the same back/lay on every selection. Convenience
    factory for tests that don't care about per-selection odds shape."""
    lay_price = lay if lay is not None else back
    odds = SelectionOdds(back_price=back, lay_price=lay_price)
    return OddsSnapshot(
        match_id=match_id,
        timestamp=timestamp,
        home=odds,
        draw=odds,
        away=odds,
    )


def _result(
    match_id: str, timestamp: datetime, outcome: Selection
) -> MatchResult:
    scores = {
        Selection.HOME: (1, 0),
        Selection.DRAW: (1, 1),
        Selection.AWAY: (0, 1),
    }
    home_goals, away_goals = scores[outcome]
    return MatchResult(
        match_id=match_id,
        timestamp=timestamp,
        home_goals=home_goals,
        away_goals=away_goals,
    )


# ---------- type-level tests -----------------------------------------------


class TestSide:
    def test_values(self) -> None:
        assert Side.BACK.value == "back"
        assert Side.LAY.value == "lay"

    def test_is_str_enum(self) -> None:
        # StrEnum membership lets BetOrder accept either the enum or its str
        # value, which keeps Pydantic construction ergonomic in tests and
        # notebooks.
        assert isinstance(Side.BACK, str)


class TestCommittedFunds:
    def test_back_returns_stake(self) -> None:
        assert committed_funds(Side.BACK, price=2.5, stake=10.0) == 10.0

    def test_lay_returns_liability(self) -> None:
        # Laying at 3.0 for 10 stakes reserves (3-1)*10 = 20 in liability.
        assert committed_funds(Side.LAY, price=3.0, stake=10.0) == 20.0

    @pytest.mark.parametrize("price", [1.01, 2.0, 10.0, 100.0])
    def test_back_invariant_to_price(self, price: float) -> None:
        assert committed_funds(Side.BACK, price=price, stake=5.0) == 5.0

    def test_lay_at_minimum_price_near_zero_liability(self) -> None:
        # Laying at a price just above 1 reserves almost nothing; exchange
        # math, not a bug.
        assert committed_funds(Side.LAY, price=1.01, stake=100.0) == pytest.approx(1.0)


class TestBetOrder:
    def _valid(self, **overrides: object) -> BetOrder:
        kwargs: dict[str, object] = {
            "match_id": "M1",
            "selection": Selection.HOME,
            "side": Side.BACK,
            "price": 2.0,
            "stake": 10.0,
        }
        kwargs.update(overrides)
        return BetOrder(**kwargs)  # type: ignore[arg-type]

    def test_valid_construction(self) -> None:
        order = self._valid()
        assert order.match_id == "M1"
        assert order.side is Side.BACK

    def test_rejects_empty_match_id(self) -> None:
        with pytest.raises(ValidationError):
            self._valid(match_id="")

    @pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -1.0])
    def test_rejects_price_at_or_below_one(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            self._valid(price=bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_price(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            self._valid(price=bad)

    @pytest.mark.parametrize("bad", [0.0, -0.001, -100.0])
    def test_rejects_non_positive_stake(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            self._valid(stake=bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_stake(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            self._valid(stake=bad)

    def test_is_frozen(self) -> None:
        order = self._valid()
        with pytest.raises(ValidationError):
            order.stake = 999.0


class TestPendingBet:
    def _valid(self, **overrides: object) -> PendingBet:
        kwargs: dict[str, object] = {
            "bet_id": "b0",
            "match_id": "M1",
            "selection": Selection.HOME,
            "side": Side.BACK,
            "price": 2.0,
            "stake": 10.0,
            "placed_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            "committed_funds": 10.0,
        }
        kwargs.update(overrides)
        return PendingBet(**kwargs)  # type: ignore[arg-type]

    def test_valid_construction(self) -> None:
        self._valid()

    def test_rejects_naive_placed_at(self) -> None:
        naive = datetime(2024, 1, 1, 12, 0)
        with pytest.raises(ValidationError):
            self._valid(placed_at=naive)

    def test_rejects_non_utc_placed_at(self) -> None:
        non_utc = datetime(2024, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=3)))
        with pytest.raises(ValidationError, match="UTC"):
            self._valid(placed_at=non_utc)

    def test_rejects_non_positive_committed(self) -> None:
        with pytest.raises(ValidationError):
            self._valid(committed_funds=0.0)

    def test_is_frozen(self) -> None:
        pending = self._valid()
        with pytest.raises(ValidationError):
            pending.price = 3.0


class TestSettledBet:
    def _valid(self, **overrides: object) -> SettledBet:
        kwargs: dict[str, object] = {
            "bet_id": "b0",
            "match_id": "M1",
            "selection": Selection.HOME,
            "side": Side.BACK,
            "price": 2.0,
            "stake": 10.0,
            "placed_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            "committed_funds": 10.0,
            "settled_at": datetime(2024, 1, 1, 14, 0, tzinfo=UTC),
            "outcome": Selection.HOME,
            "gross_pnl": 10.0,
            "commission": 0.5,
            "net_pnl": 9.5,
            "bankroll_after": 109.5,
        }
        kwargs.update(overrides)
        return SettledBet(**kwargs)  # type: ignore[arg-type]

    def test_valid_construction(self) -> None:
        self._valid()

    def test_net_pnl_matches_gross_minus_commission(self) -> None:
        # Tolerance of 1e-8 is the documented invariant; 1e-10 drift is fine.
        self._valid(gross_pnl=10.0, commission=0.5, net_pnl=9.5 + 1e-10)

    def test_rejects_net_pnl_drift_beyond_tolerance(self) -> None:
        with pytest.raises(ValidationError, match="net_pnl"):
            self._valid(gross_pnl=10.0, commission=0.5, net_pnl=9.5 + 1e-6)

    def test_rejects_settled_before_placed(self) -> None:
        with pytest.raises(ValidationError, match="settled_at"):
            self._valid(
                placed_at=datetime(2024, 1, 1, 14, 0, tzinfo=UTC),
                settled_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            )

    def test_rejects_negative_commission(self) -> None:
        with pytest.raises(ValidationError):
            self._valid(commission=-0.1)

    def test_is_frozen(self) -> None:
        settled = self._valid()
        with pytest.raises(ValidationError):
            settled.net_pnl = 0.0


class TestRejectedOrder:
    def _order(self) -> BetOrder:
        return BetOrder(
            match_id="M1",
            selection=Selection.HOME,
            side=Side.BACK,
            price=2.0,
            stake=10.0,
        )

    def test_valid_construction(self) -> None:
        rej = RejectedOrder(
            order=self._order(),
            rejected_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            reason=RejectionReason.OFF_SNAPSHOT,
            detail="price too high",
        )
        assert rej.reason is RejectionReason.OFF_SNAPSHOT

    def test_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            RejectedOrder(
                order=self._order(),
                rejected_at=datetime(2024, 1, 1, 12, 0),
                reason=RejectionReason.OFF_SNAPSHOT,
                detail="x",
            )

    def test_rejects_empty_detail(self) -> None:
        with pytest.raises(ValidationError):
            RejectedOrder(
                order=self._order(),
                rejected_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                reason=RejectionReason.OFF_SNAPSHOT,
                detail="",
            )

    def test_is_frozen(self) -> None:
        rej = RejectedOrder(
            order=self._order(),
            rejected_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            reason=RejectionReason.OFF_SNAPSHOT,
            detail="x",
        )
        with pytest.raises(ValidationError):
            rej.reason = RejectionReason.MISSING_MATCH


class TestPortfolioView:
    def test_valid_construction(self) -> None:
        view = PortfolioView(
            available_bankroll=100.0,
            starting_bankroll=100.0,
            open_bets_count=0,
            realised_pnl=0.0,
        )
        assert view.open_bets_count == 0

    def test_is_frozen(self) -> None:
        view = PortfolioView(
            available_bankroll=100.0,
            starting_bankroll=100.0,
            open_bets_count=0,
            realised_pnl=0.0,
        )
        with pytest.raises(ValidationError):
            view.available_bankroll = 0.0

    def test_rejects_non_positive_starting(self) -> None:
        with pytest.raises(ValidationError):
            PortfolioView(
                available_bankroll=100.0,
                starting_bankroll=0.0,
                open_bets_count=0,
                realised_pnl=0.0,
            )

    def test_rejects_negative_open_bets_count(self) -> None:
        with pytest.raises(ValidationError):
            PortfolioView(
                available_bankroll=100.0,
                starting_bankroll=100.0,
                open_bets_count=-1,
                realised_pnl=0.0,
            )


class TestRawBacktestOutput:
    def test_accepts_empty_run(self) -> None:
        output = RawBacktestOutput(ledger=(), rejections=())
        assert output.ledger == ()
        assert output.rejections == ()

    def test_coerces_list_inputs_to_tuples(self) -> None:
        # Pydantic validates tuple[...] fields by coercing sequences; this
        # check documents that the stored form is always a tuple regardless
        # of how the backtester populates it internally.
        output = RawBacktestOutput.model_validate({"ledger": [], "rejections": []})
        assert isinstance(output.ledger, tuple)
        assert isinstance(output.rejections, tuple)

    def test_is_frozen(self) -> None:
        output = RawBacktestOutput(ledger=(), rejections=())
        with pytest.raises(ValidationError):
            output.ledger = ()


# ---------- strategy protocol ----------------------------------------------


class TestStrategyProtocol:
    def test_scripted_strategy_is_strategy(self) -> None:
        # Structural check via runtime_checkable Protocol, parallels the
        # EventSource and CommissionModel conformance tests in modules 2/3.
        assert isinstance(ScriptedStrategy(), Strategy)

    def test_non_strategy_is_not(self) -> None:
        class NoHooks:
            pass

        assert not isinstance(NoHooks(), Strategy)

    def test_partial_strategy_is_not(self) -> None:
        class NoSettled:
            def fit(self, history: Iterable[Event]) -> None:
                pass

            def on_odds(
                self, snapshot: OddsSnapshot, portfolio: PortfolioView
            ) -> list[BetOrder]:
                return []

        assert not isinstance(NoSettled(), Strategy)


# ---------- backtester construction ----------------------------------------


class TestBacktesterConstruction:
    def _kwargs(self, **overrides: object) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "event_source": ListEventSource([]),
            "strategy": ScriptedStrategy(),
            "commission_model": NetWinningsCommission(),
            "starting_bankroll": 100.0,
            "seed": 0,
        }
        kwargs.update(overrides)
        return kwargs

    def test_valid_construction(self) -> None:
        bt = Backtester(**self._kwargs())  # type: ignore[arg-type]
        assert bt.seed == 0

    @pytest.mark.parametrize("bad", [0.0, -0.1, -1.0])
    def test_rejects_non_positive_starting_bankroll(self, bad: float) -> None:
        with pytest.raises(ValueError, match="positive"):
            Backtester(**self._kwargs(starting_bankroll=bad))  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_starting_bankroll(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            Backtester(**self._kwargs(starting_bankroll=bad))  # type: ignore[arg-type]

    def test_seed_is_stored_and_readable(self) -> None:
        bt = Backtester(**self._kwargs(seed=4242))  # type: ignore[arg-type]
        assert bt.seed == 4242


# ---------- accounting end-to-end ------------------------------------------


class TestAccountingEndToEnd:
    """Hand-computed exact-ledger test against the deterministic synthetic stream.

    The edge case the framing of decision-tree discussion flagged: the
    SyntheticGenerator emits fair odds, so no statistical edge exists to
    "recover" — but the accounting pipeline (order validation, bet placement,
    settlement, commission attribution, bankroll bookkeeping) is fully
    exercised by running it against a known-seed outcome sequence and
    comparing the produced ledger to a hand-computation that never runs the
    backtester.
    """

    def _scripted_back_home(self, match_ids: list[str], stake: float) -> ScriptedStrategy:
        orders: dict[str, list[BetOrder]] = {}
        for mid in match_ids:
            orders[mid] = [
                BetOrder(
                    match_id=mid,
                    selection=Selection.HOME,
                    side=Side.BACK,
                    price=2.0,  # fair back at p_home = 0.5
                    stake=stake,
                )
            ]
        return ScriptedStrategy(orders)

    def test_exact_ledger_from_synthetic(self) -> None:
        probs = TrueProbabilities(home=0.5, draw=0.3, away=0.2)
        cfg = SyntheticGeneratorConfig(
            n_matches=10,
            true_probs=probs,
            seed=42,
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        generator = SyntheticGenerator(cfg)

        # Discover the exact outcome sequence the seed will produce. The
        # generator is re-iterable and seed-deterministic, so the second
        # iteration in Backtester.run() replays the same events.
        odds_by_match: dict[str, OddsAvailable] = {}
        results_in_order: list[MatchResult] = []
        for event in generator.events():
            if isinstance(event, OddsAvailable):
                odds_by_match[event.snapshot.match_id] = event
            else:
                assert isinstance(event, MatchSettled)
                results_in_order.append(event.result)

        match_ids = [m.match_id for m in generator.matches]
        stake = 1.0
        rate = 0.05
        starting = 100.0

        strategy = self._scripted_back_home(match_ids, stake=stake)
        backtester = Backtester(
            event_source=generator,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=rate),
            starting_bankroll=starting,
            seed=7,
        )
        output = backtester.run()

        assert len(output.rejections) == 0
        assert len(output.ledger) == len(results_in_order)

        price = 2.0
        running_realised = 0.0
        # Match-settled events fire in kickoff order; one bet per match, so
        # ledger order matches results_in_order.
        for row, result in zip(output.ledger, results_in_order, strict=True):
            odds_event = odds_by_match[result.match_id]
            expected_bet_id = f"{result.match_id}#0000"
            if result.outcome is Selection.HOME:
                gross = stake * (price - 1.0)
                commission = rate * gross  # single-bet market, all attributed
            else:
                gross = -stake
                commission = 0.0
            net = gross - commission
            running_realised += net

            assert row.bet_id == expected_bet_id
            assert row.match_id == result.match_id
            assert row.selection is Selection.HOME
            assert row.side is Side.BACK
            assert row.price == pytest.approx(price)
            assert row.stake == pytest.approx(stake)
            assert row.placed_at == odds_event.snapshot.timestamp
            assert row.committed_funds == pytest.approx(stake)
            assert row.settled_at == result.timestamp
            assert row.outcome is result.outcome
            assert row.gross_pnl == pytest.approx(gross)
            assert row.commission == pytest.approx(commission)
            assert row.net_pnl == pytest.approx(net)
            assert row.bankroll_after == pytest.approx(starting + running_realised)

    def test_empty_strategy_leaves_ledger_empty(self) -> None:
        # A strategy that emits no orders still produces valid state and
        # visits on_settled once per match.
        probs = TrueProbabilities(home=0.5, draw=0.3, away=0.2)
        cfg = SyntheticGeneratorConfig(
            n_matches=3,
            true_probs=probs,
            seed=1,
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        generator = SyntheticGenerator(cfg)
        strategy = ScriptedStrategy()  # no orders
        backtester = Backtester(
            event_source=generator,
            strategy=strategy,
            commission_model=NetWinningsCommission(),
            starting_bankroll=50.0,
            seed=0,
        )
        output = backtester.run()
        assert output.ledger == ()
        assert output.rejections == ()
        assert len(strategy.on_odds_calls) == 3
        assert len(strategy.on_settled_calls) == 3


# ---------- arbitrage accounting ------------------------------------------


class TestArbitrageAccounting:
    """End-to-end proof of decisions 8 + 9: gross P&L per bet, commission
    aggregated per market, attribution to winners only."""

    def test_back_plus_lay_same_market_home_wins(self) -> None:
        match_id = "M1"
        t_odds = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 6, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot(match_id, t_odds, back=2.0, lay=2.0)
        result = _result(match_id, t_settle, Selection.HOME)

        orders = [
            BetOrder(
                match_id=match_id,
                selection=Selection.HOME,
                side=Side.BACK,
                price=2.0,
                stake=100.0,
            ),
            BetOrder(
                match_id=match_id,
                selection=Selection.HOME,
                side=Side.LAY,
                price=2.0,
                stake=40.0,
            ),
        ]
        strategy = ScriptedStrategy({match_id: orders})
        source = ListEventSource(
            [OddsAvailable(snapshot=snapshot), MatchSettled(result=result)]
        )
        rate = 0.05
        starting = 200.0
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=rate),
            starting_bankroll=starting,
            seed=0,
        )

        output = backtester.run()

        assert len(output.rejections) == 0
        assert len(output.ledger) == 2
        back_row, lay_row = output.ledger

        # Back: stake * (price - 1) = 100 * 1 = 100.
        # Lay loses: -stake * (price - 1) = -40.
        # Market net = 60. Commission = 0.05 * 60 = 3.0, attributed to the
        # winning back bet only.
        assert back_row.side is Side.BACK
        assert back_row.gross_pnl == pytest.approx(100.0)
        assert back_row.commission == pytest.approx(3.0)
        assert back_row.net_pnl == pytest.approx(97.0)
        # bankroll_after uses running realised, which after the back row is
        # +97. Note this temporarily reads higher than the machine cash at
        # that instant because the lay bet's committed funds have not been
        # released yet — documented in the module docstring.
        assert back_row.bankroll_after == pytest.approx(starting + 97.0)

        assert lay_row.side is Side.LAY
        assert lay_row.gross_pnl == pytest.approx(-40.0)
        assert lay_row.commission == 0.0
        assert lay_row.net_pnl == pytest.approx(-40.0)
        assert lay_row.bankroll_after == pytest.approx(starting + 57.0)

    def test_back_plus_lay_same_market_home_loses(self) -> None:
        # Mirror of the above: back loses, lay wins, market net is negative
        # so commission is zero.
        match_id = "M1"
        t_odds = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 6, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot(match_id, t_odds, back=2.0, lay=2.0)
        result = _result(match_id, t_settle, Selection.AWAY)

        orders = [
            BetOrder(
                match_id=match_id,
                selection=Selection.HOME,
                side=Side.BACK,
                price=2.0,
                stake=100.0,
            ),
            BetOrder(
                match_id=match_id,
                selection=Selection.HOME,
                side=Side.LAY,
                price=2.0,
                stake=40.0,
            ),
        ]
        strategy = ScriptedStrategy({match_id: orders})
        source = ListEventSource(
            [OddsAvailable(snapshot=snapshot), MatchSettled(result=result)]
        )
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=0.05),
            starting_bankroll=200.0,
            seed=0,
        )

        output = backtester.run()

        assert len(output.ledger) == 2
        back_row, lay_row = output.ledger
        assert back_row.gross_pnl == pytest.approx(-100.0)
        assert back_row.commission == 0.0
        assert lay_row.gross_pnl == pytest.approx(40.0)
        assert lay_row.commission == 0.0  # market net = -60, so no commission
        # Final realised = -100 + 40 = -60.
        assert lay_row.bankroll_after == pytest.approx(200.0 - 60.0)


# ---------- rejections -----------------------------------------------------


class TestRejections:
    def _run(
        self,
        orders: list[BetOrder],
        snapshot: OddsSnapshot,
        result: MatchResult,
        starting_bankroll: float = 100.0,
    ) -> RawBacktestOutput:
        strategy = ScriptedStrategy({snapshot.match_id: orders})
        source = ListEventSource(
            [OddsAvailable(snapshot=snapshot), MatchSettled(result=result)]
        )
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(),
            starting_bankroll=starting_bankroll,
            seed=0,
        )
        return backtester.run()

    def test_off_snapshot_back_price_above_market(self) -> None:
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=2.0, lay=2.0)
        result = _result("M1", t_settle, Selection.HOME)
        order = BetOrder(
            match_id="M1",
            selection=Selection.HOME,
            side=Side.BACK,
            price=2.05,
            stake=10.0,
        )
        output = self._run([order], snapshot, result)
        assert output.ledger == ()
        assert len(output.rejections) == 1
        rej = output.rejections[0]
        assert rej.reason is RejectionReason.OFF_SNAPSHOT
        assert rej.rejected_at == t_odds
        assert rej.order is order

    def test_off_snapshot_lay_price_below_market(self) -> None:
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=2.0, lay=2.0)
        result = _result("M1", t_settle, Selection.HOME)
        order = BetOrder(
            match_id="M1",
            selection=Selection.HOME,
            side=Side.LAY,
            price=1.95,
            stake=10.0,
        )
        output = self._run([order], snapshot, result)
        assert output.ledger == ()
        assert output.rejections[0].reason is RejectionReason.OFF_SNAPSHOT

    def test_missing_match_on_different_match_id(self) -> None:
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=2.0, lay=2.0)
        result = _result("M1", t_settle, Selection.HOME)
        order = BetOrder(
            match_id="WRONG",
            selection=Selection.HOME,
            side=Side.BACK,
            price=2.0,
            stake=10.0,
        )
        output = self._run([order], snapshot, result)
        assert output.ledger == ()
        assert output.rejections[0].reason is RejectionReason.MISSING_MATCH

    def test_insufficient_bankroll_single_order(self) -> None:
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=2.0, lay=2.0)
        result = _result("M1", t_settle, Selection.HOME)
        order = BetOrder(
            match_id="M1",
            selection=Selection.HOME,
            side=Side.BACK,
            price=2.0,
            stake=200.0,  # exceeds starting bankroll of 100
        )
        output = self._run([order], snapshot, result, starting_bankroll=100.0)
        assert output.ledger == ()
        assert output.rejections[0].reason is RejectionReason.INSUFFICIENT_BANKROLL

    def test_insufficient_bankroll_cascades_within_one_on_odds(self) -> None:
        # Two orders in a single on_odds call; the first consumes cash and
        # the second no longer fits. Documents that acceptance state between
        # orders is visible to the rejection check.
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=2.0, lay=2.0)
        result = _result("M1", t_settle, Selection.HOME)
        orders = [
            BetOrder(
                match_id="M1",
                selection=Selection.HOME,
                side=Side.BACK,
                price=2.0,
                stake=80.0,
            ),
            BetOrder(
                match_id="M1",
                selection=Selection.AWAY,
                side=Side.BACK,
                price=2.0,
                stake=30.0,  # only 20 cash left after first order
            ),
        ]
        output = self._run(orders, snapshot, result, starting_bankroll=100.0)
        assert len(output.ledger) == 1  # first accepted and settled
        assert len(output.rejections) == 1
        assert output.rejections[0].reason is RejectionReason.INSUFFICIENT_BANKROLL

    def test_lay_liability_checked_against_cash(self) -> None:
        # Lay at price 3.0 stake 60 reserves (3-1)*60 = 120, exceeding a 100
        # bankroll.
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=3.0, lay=3.0)
        result = _result("M1", t_settle, Selection.HOME)
        order = BetOrder(
            match_id="M1",
            selection=Selection.HOME,
            side=Side.LAY,
            price=3.0,
            stake=60.0,
        )
        output = self._run([order], snapshot, result, starting_bankroll=100.0)
        assert output.rejections[0].reason is RejectionReason.INSUFFICIENT_BANKROLL

    def test_accepted_back_at_price_equal_to_back_price(self) -> None:
        # Boundary: price == snapshot.back_price is marketable.
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=2.0, lay=2.0)
        result = _result("M1", t_settle, Selection.HOME)
        order = BetOrder(
            match_id="M1",
            selection=Selection.HOME,
            side=Side.BACK,
            price=2.0,
            stake=10.0,
        )
        output = self._run([order], snapshot, result)
        assert len(output.ledger) == 1
        assert output.rejections == ()

    def test_accepted_back_at_price_below_back_price(self) -> None:
        # Willing to take a worse price than offered — accepted.
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=2.0, lay=2.0)
        result = _result("M1", t_settle, Selection.HOME)
        order = BetOrder(
            match_id="M1",
            selection=Selection.HOME,
            side=Side.BACK,
            price=1.8,
            stake=10.0,
        )
        output = self._run([order], snapshot, result)
        assert len(output.ledger) == 1
        assert output.ledger[0].price == pytest.approx(1.8)


# ---------- lifecycle ------------------------------------------------------


class TestLifecycle:
    def _single_match_source(
        self, outcome: Selection = Selection.HOME
    ) -> tuple[ListEventSource, OddsSnapshot, MatchResult]:
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        t_settle = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=2.0, lay=2.0)
        result = _result("M1", t_settle, outcome)
        source = ListEventSource(
            [OddsAvailable(snapshot=snapshot), MatchSettled(result=result)]
        )
        return source, snapshot, result

    def test_fit_is_not_called_by_backtester(self) -> None:
        source, _, _ = self._single_match_source()
        strategy = ScriptedStrategy()
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(),
            starting_bankroll=100.0,
            seed=0,
        )
        backtester.run()
        assert strategy.fit_calls == []

    def test_on_odds_and_on_settled_called_exactly_once_each(self) -> None:
        source, _, _ = self._single_match_source()
        strategy = ScriptedStrategy()
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(),
            starting_bankroll=100.0,
            seed=0,
        )
        backtester.run()
        assert len(strategy.on_odds_calls) == 1
        assert len(strategy.on_settled_calls) == 1

    def test_on_settled_fires_when_no_pending_bets(self) -> None:
        # Match with no bets still gets notified — documented behaviour.
        source, _, _ = self._single_match_source()
        strategy = ScriptedStrategy()
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(),
            starting_bankroll=100.0,
            seed=0,
        )
        backtester.run()
        assert len(strategy.on_settled_calls) == 1

    def test_portfolio_view_at_on_settled_reflects_realised(self) -> None:
        source, snapshot, _ = self._single_match_source()
        order = BetOrder(
            match_id=snapshot.match_id,
            selection=Selection.HOME,
            side=Side.BACK,
            price=2.0,
            stake=10.0,
        )
        strategy = ScriptedStrategy({snapshot.match_id: [order]})
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=0.05),
            starting_bankroll=100.0,
            seed=0,
        )
        backtester.run()

        _, view_at_odds = strategy.on_odds_calls[0]
        assert view_at_odds.realised_pnl == 0.0
        assert view_at_odds.open_bets_count == 0

        _, view_at_settled = strategy.on_settled_calls[0]
        # Back at 2.0 for 10: gross +10, commission 0.5, net +9.5.
        assert view_at_settled.realised_pnl == pytest.approx(9.5)
        assert view_at_settled.open_bets_count == 0

    def test_portfolio_view_is_frozen(self) -> None:
        source, _, _ = self._single_match_source()
        strategy = ScriptedStrategy()
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(),
            starting_bankroll=100.0,
            seed=0,
        )
        backtester.run()
        _, view = strategy.on_odds_calls[0]
        with pytest.raises(ValidationError):
            view.available_bankroll = 0.0

    def test_strict_settlement_raises_on_open_bets(self) -> None:
        # OddsAvailable without the matching MatchSettled triggers the
        # default strict-settlement raise.
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=2.0, lay=2.0)
        order = BetOrder(
            match_id="M1",
            selection=Selection.HOME,
            side=Side.BACK,
            price=2.0,
            stake=10.0,
        )
        strategy = ScriptedStrategy({"M1": [order]})
        source = ListEventSource([OddsAvailable(snapshot=snapshot)])
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(),
            starting_bankroll=100.0,
            seed=0,
        )
        with pytest.raises(RuntimeError, match="unsettled"):
            backtester.run()

    def test_non_strict_settlement_drops_with_warning_and_refunds_cash(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # White-box test on private state: the contract is that non-strict
        # end-of-stream drops leave the backtester's internal bankroll
        # state consistent, so the bets-placed-but-not-settled cash is
        # refunded to ``_cash`` and ``_open_bets`` is emptied. Stake is
        # set to the full starting bankroll so any missing refund would
        # leave ``_cash`` at 0 and be obvious.
        starting = 100.0
        stake = starting  # whole bankroll committed to the unsettled bet
        t_odds = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        snapshot = _uniform_snapshot("M1", t_odds, back=2.0, lay=2.0)
        order = BetOrder(
            match_id="M1",
            selection=Selection.HOME,
            side=Side.BACK,
            price=2.0,
            stake=stake,
        )
        strategy = ScriptedStrategy({"M1": [order]})
        source = ListEventSource([OddsAvailable(snapshot=snapshot)])
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(),
            starting_bankroll=starting,
            seed=0,
            strict_settlement=False,
        )
        with caplog.at_level(logging.WARNING, logger="betting_backtester.backtester"):
            output = backtester.run()

        assert output.ledger == ()
        assert output.rejections == ()

        messages = [record.getMessage() for record in caplog.records]
        assert any("unsettled" in m for m in messages)
        # Log wording must truthfully describe the refund behaviour.
        assert any("refunding" in m for m in messages)

        # Committed funds refunded back to cash; open-bets map cleared.
        assert backtester._cash == pytest.approx(starting)
        assert backtester._open_bets == {}

    def test_strict_settlement_passes_on_clean_run(self) -> None:
        source, _, _ = self._single_match_source()
        strategy = ScriptedStrategy()
        backtester = Backtester(
            event_source=source,
            strategy=strategy,
            commission_model=NetWinningsCommission(),
            starting_bankroll=100.0,
            seed=0,
        )
        # No raise expected.
        backtester.run()

    def test_unknown_event_type_raises(self) -> None:
        class BogusEvent:
            @property
            def timestamp(self) -> datetime:
                return datetime(2024, 1, 1, tzinfo=UTC)

        class BogusSource:
            def events(self) -> Iterator[Event]:
                yield BogusEvent()  # type: ignore[misc]

        backtester = Backtester(
            event_source=BogusSource(),
            strategy=ScriptedStrategy(),
            commission_model=NetWinningsCommission(),
            starting_bankroll=100.0,
            seed=0,
        )
        with pytest.raises(TypeError, match="unexpected event type"):
            backtester.run()

    def test_source_conformance(self) -> None:
        # ListEventSource satisfies the EventSource protocol.
        assert isinstance(ListEventSource([]), EventSource)


# ---------- determinism ----------------------------------------------------


class TestDeterminism:
    def _build(self, gen_seed: int, bt_seed: int) -> RawBacktestOutput:
        probs = TrueProbabilities(home=0.5, draw=0.3, away=0.2)
        cfg = SyntheticGeneratorConfig(
            n_matches=20,
            true_probs=probs,
            seed=gen_seed,
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        generator = SyntheticGenerator(cfg)

        # Back HOME every match.
        orders = {
            m.match_id: [
                BetOrder(
                    match_id=m.match_id,
                    selection=Selection.HOME,
                    side=Side.BACK,
                    price=2.0,
                    stake=1.0,
                )
            ]
            for m in generator.matches
        }
        strategy = ScriptedStrategy(orders)
        backtester = Backtester(
            event_source=generator,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=0.05),
            starting_bankroll=100.0,
            seed=bt_seed,
        )
        return backtester.run()

    def test_same_seed_yields_byte_identical_output(self) -> None:
        a = self._build(gen_seed=123, bt_seed=1)
        b = self._build(gen_seed=123, bt_seed=1)
        assert a == b

    def test_reusing_one_backtester_twice_yields_identical_output(self) -> None:
        # ``run()`` resets state, so a single instance is re-runnable.
        probs = TrueProbabilities(home=0.5, draw=0.3, away=0.2)
        cfg = SyntheticGeneratorConfig(
            n_matches=10,
            true_probs=probs,
            seed=55,
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        generator = SyntheticGenerator(cfg)
        orders = {
            m.match_id: [
                BetOrder(
                    match_id=m.match_id,
                    selection=Selection.HOME,
                    side=Side.BACK,
                    price=2.0,
                    stake=1.0,
                )
            ]
            for m in generator.matches
        }
        strategy = ScriptedStrategy(orders)
        backtester = Backtester(
            event_source=generator,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=0.05),
            starting_bankroll=100.0,
            seed=0,
        )
        assert backtester.run() == backtester.run()

    def test_different_generator_seed_yields_different_output(self) -> None:
        # Guards against "determinism by ignoring the seed" in the synthetic
        # stream: two distinct generator seeds must produce distinct
        # ledgers under the same strategy.
        a = self._build(gen_seed=1, bt_seed=0)
        b = self._build(gen_seed=2, bt_seed=0)
        assert a != b

    def test_backtester_seed_not_used_in_core_loop(self) -> None:
        # 4a documents that ``seed`` is stored for 4b's bootstrap but does
        # not affect the core loop. Two runs differing only in bt seed must
        # produce identical outputs.
        a = self._build(gen_seed=1, bt_seed=111)
        b = self._build(gen_seed=1, bt_seed=222)
        assert a == b


# ---------- convergence / statistical sanity -------------------------------


class TestConvergence:
    def test_zero_ev_drift_bounded(self) -> None:
        # With fair odds and zero commission, backing HOME every match has
        # exact per-bet EV of 0. Variance per bet with true_prob p and
        # stake s is s^2 * (1 - p) / p; summed over N independent bets the
        # total standard deviation is s * sqrt(N * (1 - p) / p). Asserting
        # |realised| < 5 sigma catches any systematic drift introduced by
        # a broken gross-P&L formula without being fragile on lucky seeds.
        n = 5_000
        p = 0.5
        stake = 1.0
        probs = TrueProbabilities(home=p, draw=0.25, away=0.25)
        cfg = SyntheticGeneratorConfig(
            n_matches=n,
            true_probs=probs,
            seed=2024,
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        generator = SyntheticGenerator(cfg)
        orders = {
            m.match_id: [
                BetOrder(
                    match_id=m.match_id,
                    selection=Selection.HOME,
                    side=Side.BACK,
                    price=2.0,
                    stake=stake,
                )
            ]
            for m in generator.matches
        }
        strategy = ScriptedStrategy(orders)
        backtester = Backtester(
            event_source=generator,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=0.0),
            starting_bankroll=10_000.0,
            seed=0,
        )
        output = backtester.run()

        realised = sum(row.net_pnl for row in output.ledger)
        sigma_total = stake * math.sqrt(n * (1.0 - p) / p)
        assert abs(realised) < 5.0 * sigma_total

    def test_commission_creates_exact_expected_drag_on_winners(self) -> None:
        # Sanity check on commission accounting: sum of commission across
        # the ledger equals rate * sum of positive gross_pnl (since every
        # settled market is a single bet, so winner sets the attribution).
        n = 200
        probs = TrueProbabilities(home=0.5, draw=0.25, away=0.25)
        cfg = SyntheticGeneratorConfig(
            n_matches=n,
            true_probs=probs,
            seed=7,
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        generator = SyntheticGenerator(cfg)
        orders = {
            m.match_id: [
                BetOrder(
                    match_id=m.match_id,
                    selection=Selection.HOME,
                    side=Side.BACK,
                    price=2.0,
                    stake=1.0,
                )
            ]
            for m in generator.matches
        }
        strategy = ScriptedStrategy(orders)
        rate = 0.05
        backtester = Backtester(
            event_source=generator,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=rate),
            starting_bankroll=100.0,
            seed=0,
        )
        output = backtester.run()

        total_commission = sum(row.commission for row in output.ledger)
        total_winnings = sum(
            row.gross_pnl for row in output.ledger if row.gross_pnl > 0.0
        )
        assert total_commission == pytest.approx(rate * total_winnings)


# ---------- bankroll invariant ---------------------------------------------


class TestBankrollInvariant:
    def test_bankroll_after_equals_starting_plus_running_realised(self) -> None:
        # Every ledger row's bankroll_after should equal starting_bankroll
        # plus the running sum of net_pnl up to and including that row.
        probs = TrueProbabilities(home=0.5, draw=0.25, away=0.25)
        cfg = SyntheticGeneratorConfig(
            n_matches=25,
            true_probs=probs,
            seed=99,
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        generator = SyntheticGenerator(cfg)
        orders = {
            m.match_id: [
                BetOrder(
                    match_id=m.match_id,
                    selection=Selection.HOME,
                    side=Side.BACK,
                    price=2.0,
                    stake=1.0,
                )
            ]
            for m in generator.matches
        }
        strategy = ScriptedStrategy(orders)
        starting = 100.0
        backtester = Backtester(
            event_source=generator,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=0.05),
            starting_bankroll=starting,
            seed=0,
        )
        output = backtester.run()

        running = 0.0
        for row in output.ledger:
            running += row.net_pnl
            assert row.bankroll_after == pytest.approx(starting + running)
