"""Tests for :class:`betting_backtester.strategies.favourite_backer.FavouriteBacker`.

Organisation:

* ``TestConstruction`` pins constructor validation and the read-only
  ``stake`` attribute.
* ``TestFavouriteSelection`` covers clear favourites per selection and
  the tie-break cases (all three equal, plus each pairwise tie).
* ``TestOnOddsContract`` pins the emitted-order shape: exactly one,
  ``Side.BACK``, stake/match_id/price echoed correctly.
* ``TestNoOpCallbacks`` confirms ``fit`` and ``on_settled`` are no-ops
  that mutate nothing.
* ``TestStrategyProtocol`` confirms runtime-checkable protocol conformance.
* ``TestSyntheticIntegration`` runs end-to-end against
  :class:`~betting_backtester.synthetic.SyntheticGenerator`.
* ``TestFootballDataIntegration`` runs end-to-end against the
  ``clean_modern/E0.csv`` fixture and hand-verifies two specific matches.
* ``TestDeterminism`` confirms two runs yield field-for-field identical
  :class:`~betting_backtester.backtest_result.BacktestResult` instances.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from betting_backtester.backtest_result import BacktestResult
from betting_backtester.backtester import (
    Backtester,
    PortfolioView,
    Side,
    Strategy,
)
from betting_backtester.commission import NetWinningsCommission
from betting_backtester.football_data import FootballDataLoader
from betting_backtester.models import (
    MatchResult,
    OddsSnapshot,
    Selection,
    SelectionOdds,
)
from betting_backtester.strategies.favourite_backer import FavouriteBacker
from betting_backtester.synthetic import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
    TrueProbabilities,
)

UTC = timezone.utc
CLEAN_MODERN_E0 = (
    Path(__file__).parent / "fixtures" / "football_data" / "clean_modern" / "E0.csv"
)


# ---------- shared helpers -------------------------------------------------


def _odds(price: float) -> SelectionOdds:
    return SelectionOdds(back_price=price, lay_price=price)


def _snapshot(
    *,
    home: float,
    draw: float,
    away: float,
    match_id: str = "M1",
    timestamp: datetime | None = None,
) -> OddsSnapshot:
    if timestamp is None:
        timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    return OddsSnapshot(
        match_id=match_id,
        timestamp=timestamp,
        home=_odds(home),
        draw=_odds(draw),
        away=_odds(away),
    )


def _portfolio(*, available: float = 1_000.0) -> PortfolioView:
    return PortfolioView(
        available_bankroll=available,
        starting_bankroll=1_000.0,
        open_bets_count=0,
        realised_pnl=0.0,
    )


# ---------- construction ---------------------------------------------------


class TestConstruction:
    def test_valid_stake_exposed_via_property(self) -> None:
        assert FavouriteBacker(stake=5.0).stake == 5.0

    def test_stake_is_read_only(self) -> None:
        strategy = FavouriteBacker(stake=5.0)
        with pytest.raises(AttributeError):
            strategy.stake = 10.0  # type: ignore[misc]

    def test_slots_rejects_arbitrary_attributes(self) -> None:
        strategy = FavouriteBacker(stake=5.0)
        with pytest.raises(AttributeError):
            strategy.extra = 1  # type: ignore[attr-defined]

    @pytest.mark.parametrize("bad", [0.0, -1.0, -0.0001])
    def test_non_positive_stake_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="positive"):
            FavouriteBacker(stake=bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_stake_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            FavouriteBacker(stake=bad)


# ---------- favourite selection + tie-breaks -------------------------------


class TestFavouriteSelection:
    @pytest.fixture
    def strategy(self) -> FavouriteBacker:
        return FavouriteBacker(stake=1.0)

    def test_clear_home_favourite(self, strategy: FavouriteBacker) -> None:
        [order] = strategy.on_odds(
            _snapshot(home=1.5, draw=4.0, away=6.0), _portfolio()
        )
        assert order.selection is Selection.HOME
        assert order.price == 1.5

    def test_clear_draw_favourite(self, strategy: FavouriteBacker) -> None:
        [order] = strategy.on_odds(
            _snapshot(home=4.0, draw=1.5, away=6.0), _portfolio()
        )
        assert order.selection is Selection.DRAW
        assert order.price == 1.5

    def test_clear_away_favourite(self, strategy: FavouriteBacker) -> None:
        [order] = strategy.on_odds(
            _snapshot(home=4.0, draw=6.0, away=1.5), _portfolio()
        )
        assert order.selection is Selection.AWAY
        assert order.price == 1.5

    def test_all_three_equal_prefers_home(self, strategy: FavouriteBacker) -> None:
        [order] = strategy.on_odds(
            _snapshot(home=2.0, draw=2.0, away=2.0), _portfolio()
        )
        assert order.selection is Selection.HOME

    def test_home_equals_draw_prefers_home(self, strategy: FavouriteBacker) -> None:
        [order] = strategy.on_odds(
            _snapshot(home=2.0, draw=2.0, away=5.0), _portfolio()
        )
        assert order.selection is Selection.HOME

    def test_home_equals_away_prefers_home(self, strategy: FavouriteBacker) -> None:
        # HOME before AWAY: this is the case the existing tie-break tests
        # don't otherwise exercise, so it pins the full HOME > DRAW > AWAY
        # walk rather than just HOME > DRAW.
        [order] = strategy.on_odds(
            _snapshot(home=2.0, draw=5.0, away=2.0), _portfolio()
        )
        assert order.selection is Selection.HOME

    def test_draw_equals_away_prefers_draw(self, strategy: FavouriteBacker) -> None:
        [order] = strategy.on_odds(
            _snapshot(home=5.0, draw=2.0, away=2.0), _portfolio()
        )
        assert order.selection is Selection.DRAW


# ---------- on_odds contract -----------------------------------------------


class TestOnOddsContract:
    def test_emits_exactly_one_order(self) -> None:
        orders = FavouriteBacker(stake=2.5).on_odds(
            _snapshot(home=1.5, draw=4.0, away=6.0), _portfolio()
        )
        assert len(orders) == 1

    def test_always_side_back(self) -> None:
        [order] = FavouriteBacker(stake=2.5).on_odds(
            _snapshot(home=1.5, draw=4.0, away=6.0), _portfolio()
        )
        assert order.side is Side.BACK

    def test_stake_echoed_verbatim(self) -> None:
        [order] = FavouriteBacker(stake=3.75).on_odds(
            _snapshot(home=1.5, draw=4.0, away=6.0), _portfolio()
        )
        assert order.stake == 3.75

    def test_match_id_echoed(self) -> None:
        [order] = FavouriteBacker(stake=1.0).on_odds(
            _snapshot(match_id="custom-id", home=1.5, draw=4.0, away=6.0),
            _portfolio(),
        )
        assert order.match_id == "custom-id"

    def test_price_equals_favourite_back_price(self) -> None:
        [order] = FavouriteBacker(stake=1.0).on_odds(
            _snapshot(home=1.5, draw=4.0, away=6.0), _portfolio()
        )
        assert order.price == 1.5

    def test_strategy_ignores_portfolio_state(self) -> None:
        # Zero available bankroll still yields an order: the strategy is
        # bankroll-blind by design, and the backtester alone decides
        # acceptance vs. INSUFFICIENT_BANKROLL rejection.
        broke = PortfolioView(
            available_bankroll=0.0,
            starting_bankroll=1_000.0,
            open_bets_count=0,
            realised_pnl=-1_000.0,
        )
        orders = FavouriteBacker(stake=1.0).on_odds(
            _snapshot(home=1.5, draw=4.0, away=6.0), broke
        )
        assert len(orders) == 1


# ---------- no-op callbacks ------------------------------------------------


class TestNoOpCallbacks:
    def test_fit_does_not_mutate_state(self) -> None:
        strategy = FavouriteBacker(stake=1.0)
        strategy.fit([])
        assert strategy.stake == 1.0

    def test_on_settled_does_not_mutate_state(self) -> None:
        strategy = FavouriteBacker(stake=1.0)
        result = MatchResult(
            match_id="M1",
            timestamp=datetime(2024, 1, 1, 17, tzinfo=UTC),
            home_goals=1,
            away_goals=0,
        )
        strategy.on_settled(result, _portfolio())
        assert strategy.stake == 1.0

    def test_repeated_on_odds_emits_identical_order(self) -> None:
        # Stateless contract: given the same snapshot, subsequent calls
        # emit byte-identical orders.
        strategy = FavouriteBacker(stake=1.0)
        snap = _snapshot(home=1.5, draw=4.0, away=6.0)
        first = strategy.on_odds(snap, _portfolio())
        second = strategy.on_odds(snap, _portfolio())
        assert first == second


# ---------- protocol conformance -------------------------------------------


class TestStrategyProtocol:
    def test_satisfies_runtime_checkable_strategy(self) -> None:
        assert isinstance(FavouriteBacker(stake=1.0), Strategy)


# ---------- synthetic integration ------------------------------------------


class TestSyntheticIntegration:
    def test_home_heavy_run_is_all_home_backs_at_fair_odds(self) -> None:
        generator = SyntheticGenerator(
            SyntheticGeneratorConfig(
                n_matches=10,
                true_probs=TrueProbabilities(home=0.6, draw=0.25, away=0.15),
                seed=42,
                start=datetime(2024, 3, 1, 15, tzinfo=UTC),
            )
        )
        backtester = Backtester(
            event_source=generator,
            strategy=FavouriteBacker(stake=1.0),
            commission_model=NetWinningsCommission(rate=0.05),
            starting_bankroll=100.0,
            seed=0,
        )
        result = BacktestResult.from_raw(
            backtester.run(),
            starting_bankroll=100.0,
            t0=datetime(2024, 3, 1, tzinfo=UTC),
        )
        assert result.summary_metrics.n_bets == 10
        assert result.summary_metrics.n_rejections == 0
        assert result.summary_metrics.roi is not None
        assert all(bet.side is Side.BACK for bet in result.ledger)
        # Fair odds under a home-heavy true distribution put HOME at the
        # lowest back_price (1/0.6 ≈ 1.667 vs 4.0 vs ≈ 6.667), so every
        # bet is on HOME at that price.
        assert all(bet.selection is Selection.HOME for bet in result.ledger)
        assert all(bet.price == pytest.approx(1.0 / 0.6) for bet in result.ledger)


# ---------- football-data integration --------------------------------------


class TestFootballDataIntegration:
    @pytest.fixture
    def result(self) -> BacktestResult:
        loader = FootballDataLoader(csv_paths=[CLEAN_MODERN_E0])
        backtester = Backtester(
            event_source=loader,
            strategy=FavouriteBacker(stake=1.0),
            commission_model=NetWinningsCommission(rate=0.05),
            starting_bankroll=1_000.0,
            seed=0,
        )
        return BacktestResult.from_raw(
            backtester.run(),
            starting_bankroll=1_000.0,
            t0=datetime(2023, 8, 1, tzinfo=UTC),
        )

    def test_one_bet_per_match_no_rejections(self, result: BacktestResult) -> None:
        assert result.summary_metrics.n_bets == 6
        assert result.summary_metrics.n_rejections == 0

    def test_every_bet_is_back(self, result: BacktestResult) -> None:
        assert all(bet.side is Side.BACK for bet in result.ledger)

    def test_arsenal_nottm_forest_backs_home_at_pinnacle_price(
        self, result: BacktestResult
    ) -> None:
        # Row: 12/08/2023 Arsenal vs Nott'm Forest, PSH=1.35, PSD=5.00,
        # PSA=9.00. Hand-verified favourite: HOME @ 1.35.
        [bet] = [
            b
            for b in result.ledger
            if b.match_id == "E0-2023-08-12-Arsenal-Nott_m_Forest"
        ]
        assert bet.selection is Selection.HOME
        assert bet.price == pytest.approx(1.35)

    def test_bournemouth_west_ham_backs_away(self, result: BacktestResult) -> None:
        # Row: 12/08/2023 Bournemouth vs West Ham, PSH=2.90, PSD=3.50,
        # PSA=2.50. Hand-verified favourite: AWAY @ 2.50. Deliberately
        # the non-obvious case -- catches any accidental hard-coded
        # ``snapshot.home`` in the selection logic.
        [bet] = [
            b
            for b in result.ledger
            if b.match_id == "E0-2023-08-12-Bournemouth-West_Ham"
        ]
        assert bet.selection is Selection.AWAY
        assert bet.price == pytest.approx(2.50)

    def test_summary_metrics_populated(self, result: BacktestResult) -> None:
        assert result.summary_metrics.roi is not None
        assert result.summary_metrics.hit_rate is not None
        assert result.summary_metrics.turnover == pytest.approx(6.0)


# ---------- determinism ----------------------------------------------------


class TestDeterminism:
    def test_two_runs_byte_identical_on_football_data(self) -> None:
        def run_once() -> BacktestResult:
            loader = FootballDataLoader(csv_paths=[CLEAN_MODERN_E0])
            backtester = Backtester(
                event_source=loader,
                strategy=FavouriteBacker(stake=1.0),
                commission_model=NetWinningsCommission(rate=0.05),
                starting_bankroll=1_000.0,
                seed=0,
            )
            return BacktestResult.from_raw(
                backtester.run(),
                starting_bankroll=1_000.0,
                t0=datetime(2023, 8, 1, tzinfo=UTC),
            )

        assert run_once() == run_once()

    def test_two_runs_byte_identical_on_synthetic(self) -> None:
        def run_once() -> BacktestResult:
            generator = SyntheticGenerator(
                SyntheticGeneratorConfig(
                    n_matches=10,
                    true_probs=TrueProbabilities(home=0.6, draw=0.25, away=0.15),
                    seed=42,
                    start=datetime(2024, 3, 1, 15, tzinfo=UTC),
                )
            )
            backtester = Backtester(
                event_source=generator,
                strategy=FavouriteBacker(stake=1.0),
                commission_model=NetWinningsCommission(rate=0.05),
                starting_bankroll=100.0,
                seed=0,
            )
            return BacktestResult.from_raw(
                backtester.run(),
                starting_bankroll=100.0,
                t0=datetime(2024, 3, 1, tzinfo=UTC),
            )

        assert run_once() == run_once()
