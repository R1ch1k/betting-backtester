"""Tests for :class:`betting_backtester.strategies.arbitrage_detector.ArbitrageDetector`.

Organisation:

* ``TestConstruction`` -- numeric parameter validation and bankroll_basis
  label guard.
* ``TestReadOnlyProperties`` -- provenance properties echo the
  constructor args and resist mutation.
* ``TestStrategyProtocol`` -- runtime-checkable protocol conformance
  and ``__slots__`` enforcement.
* ``TestStatelessness`` -- ``fit`` on arbitrary histories leaves behaviour
  unchanged; ``on_settled`` is observably a no-op.
* ``TestNoArbReturnsEmpty`` -- overround and fair books emit no orders.
* ``TestArbEmitsThreeBackOrders`` -- shape, prices, stake sums, equal
  profit across outcomes.
* ``TestMinMarginFilter`` -- margin threshold discriminates tight vs
  loose arbs on a mixed stream.
* ``TestBankrollGuards`` -- non-positive bankroll and
  available-bankroll-below-required both return [].
* ``TestBankrollBasis`` -- the two bases pick different bankroll
  figures when they diverge.
* ``TestDeterminism`` -- same inputs -> byte-identical orders.
* ``TestPositionalArbs`` -- a :class:`FixedArbSchedule` produces bets
  on exactly its flagged match indices.
* ``TestEndToEndPnl`` -- full pipeline (ArbitrageGenerator ->
  Backtester -> BacktestResult) recovers the analytically iterated
  expected net P&L within float tolerance.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import pytest

from betting_backtester.arbitrage_generator import (
    ArbitrageGenerator,
    ArbitrageGeneratorConfig,
    FixedArbSchedule,
)
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
    MatchResult,
    MatchSettled,
    OddsSnapshot,
    Selection,
    SelectionOdds,
)
from betting_backtester.strategies.arbitrage_detector import ArbitrageDetector
from betting_backtester.synthetic import TrueProbabilities

UTC = timezone.utc
UTC_START = datetime(2024, 8, 1, 15, 0, tzinfo=UTC)


# ---------- helpers --------------------------------------------------------


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


def _snapshot_from_back_prices(
    *,
    match_id: str = "M-0000",
    timestamp: datetime = datetime(2024, 1, 1, 14, 55, tzinfo=UTC),
    back_prices: tuple[float, float, float],
) -> OddsSnapshot:
    """Build a snapshot with ``back == lay`` so arb math stays clean.

    ArbitrageDetector reads only the back prices, so the lay values
    here are irrelevant for the strategy's decision -- we set
    ``lay == back`` to keep the SelectionOdds validator happy
    (``back <= lay``) without layering a spread that the test would
    then have to reason about.
    """
    b_home, b_draw, b_away = back_prices
    return OddsSnapshot(
        match_id=match_id,
        timestamp=timestamp,
        home=SelectionOdds(back_price=b_home, lay_price=b_home),
        draw=SelectionOdds(back_price=b_draw, lay_price=b_draw),
        away=SelectionOdds(back_price=b_away, lay_price=b_away),
    )


def _arb_prices(
    probs: tuple[float, float, float], arb_margin: float
) -> tuple[float, float, float]:
    """Back prices that realise ``sum(1/b_i) == 1 - arb_margin``."""
    scale = 1.0 - arb_margin
    return (1.0 / (probs[0] * scale), 1.0 / (probs[1] * scale), 1.0 / (probs[2] * scale))


def _expected_iterative_net_pnl(
    *,
    starting_bankroll: float,
    fraction: float,
    arb_margin: float,
    commission_rate: float,
    n_arbs: int,
) -> float:
    """Iteratively compound the per-arb net P&L.

    Per arb: profit = bankroll * fraction * arb_margin / (1 - arb_margin)
    * (1 - commission_rate); bankroll += profit. Returns the sum over
    ``n_arbs`` iterations -- equal to aggregate ``net_pnl`` in
    ``BacktestResult.summary_metrics``.
    """
    bankroll = starting_bankroll
    total = 0.0
    per_arb_factor = fraction * arb_margin / (1.0 - arb_margin) * (1.0 - commission_rate)
    for _ in range(n_arbs):
        profit = bankroll * per_arb_factor
        total += profit
        bankroll += profit
    return total


# ---------- construction --------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        s = ArbitrageDetector()
        assert s.total_stake_fraction == 0.5
        assert s.min_margin == 0.0
        assert s.bankroll_basis == "available_cash"

    def test_explicit_values(self) -> None:
        s = ArbitrageDetector(
            total_stake_fraction=0.25,
            min_margin=0.005,
            bankroll_basis="realised_wealth",
        )
        assert s.total_stake_fraction == 0.25
        assert s.min_margin == 0.005
        assert s.bankroll_basis == "realised_wealth"

    @pytest.mark.parametrize(
        "bad", [0.0, -0.01, -1.0, 1.01, 2.0, math.nan, math.inf, -math.inf]
    )
    def test_invalid_total_stake_fraction_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="total_stake_fraction"):
            ArbitrageDetector(total_stake_fraction=bad)

    def test_total_stake_fraction_one_is_allowed(self) -> None:
        # (0, 1] is the documented range; the upper bound must be inclusive.
        ArbitrageDetector(total_stake_fraction=1.0)

    @pytest.mark.parametrize(
        "bad", [-0.01, 1.0, 1.5, math.nan, math.inf, -math.inf]
    )
    def test_invalid_min_margin_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="min_margin"):
            ArbitrageDetector(min_margin=bad)

    def test_min_margin_zero_is_allowed(self) -> None:
        # [0, 1) is the documented range; the lower bound must be inclusive.
        ArbitrageDetector(min_margin=0.0)

    def test_invalid_bankroll_basis_raises(self) -> None:
        with pytest.raises(ValueError, match="bankroll_basis"):
            ArbitrageDetector(
                bankroll_basis="starting_bankroll",  # type: ignore[arg-type]
            )


class TestReadOnlyProperties:
    def test_slots_prevent_new_attributes(self) -> None:
        s = ArbitrageDetector()
        with pytest.raises(AttributeError):
            s.surprise_attribute = 42  # type: ignore[attr-defined]

    def test_private_fields_are_immutable_via_properties(self) -> None:
        s = ArbitrageDetector(total_stake_fraction=0.25)
        assert s.total_stake_fraction == 0.25
        # Properties expose a read-only view -- attempting to assign
        # to the property raises AttributeError (no setter).
        with pytest.raises(AttributeError):
            s.total_stake_fraction = 0.5  # type: ignore[misc]


class TestStrategyProtocol:
    def test_is_strategy_instance(self) -> None:
        assert isinstance(ArbitrageDetector(), Strategy)


class TestStatelessness:
    def test_fit_accepts_empty_history_and_does_nothing(self) -> None:
        s = ArbitrageDetector()
        s.fit([])
        # Overround book (sum(1/b) = 1/2 + 1/3 + 1/5 ≈ 1.033).
        # Picking exactly representable ratios that sum strictly above
        # 1 avoids the "is this an arb under float drift?" ambiguity
        # (1/2 + 1/3 + 1/6 sums just *below* 1 in IEEE 754 and would
        # accidentally trigger the detector).
        snap = _snapshot_from_back_prices(back_prices=(2.0, 3.0, 5.0))
        assert s.on_odds(snap, _portfolio()) == []

    def test_fit_accepts_arbitrary_history_and_does_nothing(self) -> None:
        s = ArbitrageDetector()
        # Pass an obviously-relevant-looking event that a smart
        # strategy might use; stateless means this must not change
        # on_odds behaviour.
        history = [
            MatchSettled(
                result=MatchResult(
                    match_id="X-0000",
                    timestamp=datetime(2024, 1, 1, 17, tzinfo=UTC),
                    home_goals=2,
                    away_goals=1,
                )
            )
        ]
        s.fit(history)
        arb_prices = _arb_prices((0.5, 0.3, 0.2), 0.02)
        arb_snap = _snapshot_from_back_prices(back_prices=arb_prices)
        first_orders = s.on_odds(arb_snap, _portfolio())
        s.fit(history * 5)
        second_orders = s.on_odds(arb_snap, _portfolio())
        assert first_orders == second_orders

    def test_on_settled_is_observably_noop(self) -> None:
        s = ArbitrageDetector()
        # Snapshot of "state" is really just the configuration; a
        # no-op on_settled must leave every property unchanged.
        before = (s.total_stake_fraction, s.min_margin, s.bankroll_basis)
        s.on_settled(
            MatchResult(
                match_id="X-0000",
                timestamp=datetime(2024, 1, 1, 17, tzinfo=UTC),
                home_goals=1,
                away_goals=1,
            ),
            _portfolio(),
        )
        after = (s.total_stake_fraction, s.min_margin, s.bankroll_basis)
        assert before == after


# ---------- decision logic ------------------------------------------------


class TestNoArbReturnsEmpty:
    def test_overround_book_returns_empty(self) -> None:
        # 1/2 + 1/3 + 1/5 ≈ 1.033 (overround).
        s = ArbitrageDetector(min_margin=0.0)
        snap = _snapshot_from_back_prices(back_prices=(2.0, 3.0, 5.0))
        assert s.on_odds(snap, _portfolio()) == []

    def test_fair_book_returns_empty(self) -> None:
        # Exactly representable fair book: 1/2 + 1/4 + 1/4 = 1.0 with
        # zero float drift, so the boundary case (implied_sum == 1.0)
        # is tested without IEEE-754 ambiguity.
        s = ArbitrageDetector(min_margin=0.0)
        snap = _snapshot_from_back_prices(back_prices=(2.0, 4.0, 4.0))
        assert s.on_odds(snap, _portfolio()) == []

    def test_marginal_arb_blocked_by_min_margin(self) -> None:
        # 0.5% arb filtered out by min_margin = 1%.
        arb_prices = _arb_prices((0.5, 0.3, 0.2), 0.005)
        s = ArbitrageDetector(min_margin=0.01)
        snap = _snapshot_from_back_prices(back_prices=arb_prices)
        assert s.on_odds(snap, _portfolio()) == []


class TestArbEmitsThreeBackOrders:
    ARB_MARGIN = 0.02
    PROBS = (0.5, 0.3, 0.2)

    def test_emits_one_order_per_selection_all_back(self) -> None:
        arb_prices = _arb_prices(self.PROBS, self.ARB_MARGIN)
        s = ArbitrageDetector(total_stake_fraction=0.5)
        snap = _snapshot_from_back_prices(back_prices=arb_prices)
        orders = s.on_odds(snap, _portfolio(available=1000.0))
        assert len(orders) == 3
        assert {o.selection for o in orders} == {
            Selection.HOME,
            Selection.DRAW,
            Selection.AWAY,
        }
        assert all(o.side is Side.BACK for o in orders)
        assert all(o.match_id == snap.match_id for o in orders)

    def test_order_prices_match_snapshot_back_prices(self) -> None:
        arb_prices = _arb_prices(self.PROBS, self.ARB_MARGIN)
        s = ArbitrageDetector()
        snap = _snapshot_from_back_prices(back_prices=arb_prices)
        orders = s.on_odds(snap, _portfolio(available=1000.0))
        by_sel = {o.selection: o for o in orders}
        assert by_sel[Selection.HOME].price == pytest.approx(arb_prices[0])
        assert by_sel[Selection.DRAW].price == pytest.approx(arb_prices[1])
        assert by_sel[Selection.AWAY].price == pytest.approx(arb_prices[2])

    def test_stakes_sum_to_required(self) -> None:
        arb_prices = _arb_prices(self.PROBS, self.ARB_MARGIN)
        s = ArbitrageDetector(total_stake_fraction=0.5)
        snap = _snapshot_from_back_prices(back_prices=arb_prices)
        orders = s.on_odds(snap, _portfolio(available=1000.0))
        total_stake = sum(o.stake for o in orders)
        assert total_stake == pytest.approx(500.0)

    def test_stake_proportions_match_closed_form(self) -> None:
        arb_prices = _arb_prices(self.PROBS, self.ARB_MARGIN)
        s = ArbitrageDetector(total_stake_fraction=0.5)
        snap = _snapshot_from_back_prices(back_prices=arb_prices)
        orders = s.on_odds(snap, _portfolio(available=1000.0))
        implied_sum = sum(1.0 / p for p in arb_prices)
        required = 500.0
        by_sel = {o.selection: o for o in orders}
        assert by_sel[Selection.HOME].stake == pytest.approx(
            required * (1.0 / arb_prices[0]) / implied_sum
        )
        assert by_sel[Selection.DRAW].stake == pytest.approx(
            required * (1.0 / arb_prices[1]) / implied_sum
        )
        assert by_sel[Selection.AWAY].stake == pytest.approx(
            required * (1.0 / arb_prices[2]) / implied_sum
        )

    def test_equal_profit_across_outcomes(self) -> None:
        arb_prices = _arb_prices(self.PROBS, self.ARB_MARGIN)
        s = ArbitrageDetector(total_stake_fraction=0.5)
        snap = _snapshot_from_back_prices(back_prices=arb_prices)
        orders = s.on_odds(snap, _portfolio(available=1000.0))
        required = 500.0
        # For each selection's win: profit = s_i * b_i - required.
        # Under equal-profit staking these must all coincide.
        by_sel = {o.selection: o for o in orders}
        p_home = by_sel[Selection.HOME].stake * arb_prices[0] - required
        p_draw = by_sel[Selection.DRAW].stake * arb_prices[1] - required
        p_away = by_sel[Selection.AWAY].stake * arb_prices[2] - required
        expected = required * self.ARB_MARGIN / (1.0 - self.ARB_MARGIN)
        assert p_home == pytest.approx(expected)
        assert p_draw == pytest.approx(expected)
        assert p_away == pytest.approx(expected)


class TestMinMarginFilter:
    def test_min_margin_discriminates_tight_vs_loose_arbs(self) -> None:
        # Two snapshots: a 0.5% arb and a 2% arb. min_margin=1% should
        # accept the latter, reject the former.
        tight_prices = _arb_prices((0.5, 0.3, 0.2), 0.005)
        loose_prices = _arb_prices((0.5, 0.3, 0.2), 0.02)
        s = ArbitrageDetector(min_margin=0.01)
        tight_snap = _snapshot_from_back_prices(
            match_id="M-tight", back_prices=tight_prices
        )
        loose_snap = _snapshot_from_back_prices(
            match_id="M-loose", back_prices=loose_prices
        )
        assert s.on_odds(tight_snap, _portfolio()) == []
        assert len(s.on_odds(loose_snap, _portfolio())) == 3

    def test_min_margin_zero_accepts_any_strict_arb(self) -> None:
        # A 0.1% arb still triggers under the default min_margin=0.
        tiny_prices = _arb_prices((0.5, 0.3, 0.2), 0.001)
        s = ArbitrageDetector()
        snap = _snapshot_from_back_prices(back_prices=tiny_prices)
        assert len(s.on_odds(snap, _portfolio())) == 3


# ---------- bankroll guards -----------------------------------------------


class TestBankrollGuards:
    ARB_SNAP_PRICES = _arb_prices((0.5, 0.3, 0.2), 0.02)

    def test_non_positive_bankroll_returns_empty(self) -> None:
        s = ArbitrageDetector(bankroll_basis="realised_wealth")
        snap = _snapshot_from_back_prices(back_prices=self.ARB_SNAP_PRICES)
        # realised_pnl wipes out starting_bankroll -> sizing basis is 0.
        port = _portfolio(available=1000.0, starting=1000.0, realised=-1000.0)
        assert s.on_odds(snap, port) == []

    def test_available_below_required_returns_empty_and_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # realised_wealth basis says 2000 of bankroll is available for
        # sizing, but actual cash on hand is only 100 -- not enough
        # for the three bets. Policy: all-or-nothing, skip.
        s = ArbitrageDetector(
            total_stake_fraction=0.5, bankroll_basis="realised_wealth"
        )
        snap = _snapshot_from_back_prices(back_prices=self.ARB_SNAP_PRICES)
        port = _portfolio(available=100.0, starting=1000.0, realised=1000.0)
        with caplog.at_level(logging.WARNING):
            assert s.on_odds(snap, port) == []
        assert any(
            "ArbitrageDetector" in rec.message for rec in caplog.records
        )


class TestBankrollBasis:
    ARB_SNAP_PRICES = _arb_prices((0.5, 0.3, 0.2), 0.02)

    def _orders_under(
        self,
        *,
        basis: str,
        available: float,
        starting: float,
        realised: float,
    ) -> list[BetOrder]:
        s = ArbitrageDetector(
            total_stake_fraction=0.5,
            bankroll_basis=basis,  # type: ignore[arg-type]
        )
        snap = _snapshot_from_back_prices(back_prices=self.ARB_SNAP_PRICES)
        return s.on_odds(
            snap,
            _portfolio(
                available=available, starting=starting, realised=realised
            ),
        )

    def test_available_cash_ignores_realised_pnl(self) -> None:
        # available_cash == 1000, realised_wealth == 2000.
        orders = self._orders_under(
            basis="available_cash",
            available=1000.0,
            starting=1000.0,
            realised=1000.0,
        )
        assert sum(o.stake for o in orders) == pytest.approx(500.0)

    def test_realised_wealth_sizes_on_starting_plus_realised(self) -> None:
        # available_cash == 2000, realised_wealth == 2000; here they
        # happen to coincide -- the test below with divergent values
        # shows the difference.
        orders = self._orders_under(
            basis="realised_wealth",
            available=2000.0,
            starting=1000.0,
            realised=1000.0,
        )
        assert sum(o.stake for o in orders) == pytest.approx(1000.0)

    def test_bases_diverge_when_cash_and_wealth_differ(self) -> None:
        # available_cash=1500, realised_wealth=1000+500=1500 -- both
        # agree numerically. Perturb: available_cash=1500 but
        # realised=200 -> realised_wealth=1200. Sizing differs.
        avail_orders = self._orders_under(
            basis="available_cash",
            available=1500.0,
            starting=1000.0,
            realised=200.0,
        )
        wealth_orders = self._orders_under(
            basis="realised_wealth",
            available=1500.0,
            starting=1000.0,
            realised=200.0,
        )
        assert sum(o.stake for o in avail_orders) == pytest.approx(750.0)
        assert sum(o.stake for o in wealth_orders) == pytest.approx(600.0)


# ---------- determinism ----------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_orders(self) -> None:
        arb_prices = _arb_prices((0.5, 0.3, 0.2), 0.02)
        snap = _snapshot_from_back_prices(back_prices=arb_prices)
        port = _portfolio(available=1000.0)
        s_a = ArbitrageDetector(total_stake_fraction=0.25, min_margin=0.005)
        s_b = ArbitrageDetector(total_stake_fraction=0.25, min_margin=0.005)
        assert s_a.on_odds(snap, port) == s_b.on_odds(snap, port)

    def test_repeated_calls_are_stable(self) -> None:
        arb_prices = _arb_prices((0.5, 0.3, 0.2), 0.02)
        snap = _snapshot_from_back_prices(back_prices=arb_prices)
        port = _portfolio(available=1000.0)
        s = ArbitrageDetector()
        first = s.on_odds(snap, port)
        second = s.on_odds(snap, port)
        assert first == second


# ---------- positional + end-to-end pipeline -------------------------------


class TestPositionalArbs:
    def test_bets_appear_only_on_flagged_match_indices(self) -> None:
        arb_positions = frozenset({2, 7, 15})
        cfg = ArbitrageGeneratorConfig(
            n_matches=20,
            true_probs=TrueProbabilities(home=0.5, draw=0.3, away=0.2),
            seed=7,
            start=UTC_START,
            schedule=FixedArbSchedule(arb_positions),
            arb_margin=0.02,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        strategy = ArbitrageDetector(
            total_stake_fraction=0.25,
            min_margin=0.0,
            bankroll_basis="available_cash",
        )
        backtester = Backtester(
            event_source=gen,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=0.0),
            starting_bankroll=10_000.0,
            seed=1,
        )
        raw = backtester.run()

        # No rejections: required always fits in available_cash
        # because total_stake_fraction < 1 and profits compound slowly.
        assert raw.rejections == ()

        # Exactly 3 bets per arb match, none on the rest.
        bet_match_ids = [b.match_id for b in raw.ledger]
        expected_ids = {
            f"{cfg.league}-{cfg.season}-{i:04d}" for i in arb_positions
        }
        assert set(bet_match_ids) == expected_ids
        for expected in expected_ids:
            assert bet_match_ids.count(expected) == 3

        # Three BACK orders per arb match, covering HOME/DRAW/AWAY.
        for expected in expected_ids:
            per_match = [b for b in raw.ledger if b.match_id == expected]
            assert {b.selection for b in per_match} == {
                Selection.HOME,
                Selection.DRAW,
                Selection.AWAY,
            }
            assert {b.side for b in per_match} == {Side.BACK}


class TestEndToEndPnl:
    def test_aggregate_net_pnl_matches_iterative_expectation(self) -> None:
        arb_positions = frozenset({1, 4, 7, 12, 18})
        arb_margin = 0.02
        half_spread = 0.01
        commission_rate = 0.05
        starting_bankroll = 10_000.0
        fraction = 0.25

        cfg = ArbitrageGeneratorConfig(
            n_matches=20,
            true_probs=TrueProbabilities(home=0.5, draw=0.3, away=0.2),
            seed=42,
            start=UTC_START,
            schedule=FixedArbSchedule(arb_positions),
            arb_margin=arb_margin,
            half_spread=half_spread,
        )
        gen = ArbitrageGenerator(cfg)
        strategy = ArbitrageDetector(
            total_stake_fraction=fraction,
            min_margin=0.0,
            bankroll_basis="available_cash",
        )
        backtester = Backtester(
            event_source=gen,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=commission_rate),
            starting_bankroll=starting_bankroll,
            seed=1,
        )
        raw = backtester.run()
        # Anchor t0 at the first event in the stream.
        first_ts = next(iter(gen.events())).timestamp
        result = BacktestResult.from_raw(
            raw, starting_bankroll=starting_bankroll, t0=first_ts
        )

        expected = _expected_iterative_net_pnl(
            starting_bankroll=starting_bankroll,
            fraction=fraction,
            arb_margin=arb_margin,
            commission_rate=commission_rate,
            n_arbs=len(arb_positions),
        )
        # Arb margin and commission are small multiplicative factors
        # on a 10k bankroll; net pnl is O(100). 1e-6 absolute is a
        # couple of ULPs on O(100).
        assert result.summary_metrics.net_pnl == pytest.approx(
            expected, abs=1e-6
        )
        # No rejections -- the test asserts a clean pipeline.
        assert raw.rejections == ()
        # Sanity: exactly 3 bets per arb, no other bets.
        assert len(raw.ledger) == 3 * len(arb_positions)

    def test_zero_commission_yields_undiluted_arb_profit(self) -> None:
        # Isolates the "commission is a clean (1 - r) scaling" claim:
        # at r = 0 the aggregate net pnl equals the gross arb profit
        # summed across arbs, which is the iterative expectation at
        # r = 0.
        arb_positions = frozenset({0, 3, 9})
        arb_margin = 0.02
        starting_bankroll = 10_000.0
        fraction = 0.5

        cfg = ArbitrageGeneratorConfig(
            n_matches=10,
            true_probs=TrueProbabilities(home=0.5, draw=0.3, away=0.2),
            seed=3,
            start=UTC_START,
            schedule=FixedArbSchedule(arb_positions),
            arb_margin=arb_margin,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        strategy = ArbitrageDetector(
            total_stake_fraction=fraction,
            bankroll_basis="available_cash",
        )
        backtester = Backtester(
            event_source=gen,
            strategy=strategy,
            commission_model=NetWinningsCommission(rate=0.0),
            starting_bankroll=starting_bankroll,
            seed=0,
        )
        raw = backtester.run()
        first_ts = next(iter(gen.events())).timestamp
        result = BacktestResult.from_raw(
            raw, starting_bankroll=starting_bankroll, t0=first_ts
        )

        expected = _expected_iterative_net_pnl(
            starting_bankroll=starting_bankroll,
            fraction=fraction,
            arb_margin=arb_margin,
            commission_rate=0.0,
            n_arbs=len(arb_positions),
        )
        assert result.summary_metrics.net_pnl == pytest.approx(
            expected, abs=1e-6
        )
        # At r = 0, total_commission must be exactly 0.
        assert result.summary_metrics.total_commission == 0.0

    def test_same_seed_runs_produce_identical_results(self) -> None:
        arb_positions = frozenset({1, 5, 9})
        base_kwargs: dict[str, object] = {
            "n_matches": 12,
            "true_probs": TrueProbabilities(home=0.5, draw=0.3, away=0.2),
            "seed": 17,
            "start": UTC_START,
            "schedule": FixedArbSchedule(arb_positions),
            "arb_margin": 0.02,
            "half_spread": 0.01,
        }
        cfg_a = ArbitrageGeneratorConfig(**base_kwargs)  # type: ignore[arg-type]
        cfg_b = ArbitrageGeneratorConfig(**base_kwargs)  # type: ignore[arg-type]

        def _run(cfg: ArbitrageGeneratorConfig) -> tuple[object, ...]:
            gen = ArbitrageGenerator(cfg)
            strategy = ArbitrageDetector(total_stake_fraction=0.5)
            backtester = Backtester(
                event_source=gen,
                strategy=strategy,
                commission_model=NetWinningsCommission(rate=0.05),
                starting_bankroll=10_000.0,
                seed=1,
            )
            raw = backtester.run()
            return (raw.ledger, raw.rejections)

        assert _run(cfg_a) == _run(cfg_b)
