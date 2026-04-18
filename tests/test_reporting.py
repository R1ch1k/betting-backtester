"""Tests for module 4b's statistical layer: :class:`YieldCI`,
:func:`compute_yield_ci`, and the private type-7 quantile helper.

Organisation:

* ``TestYieldCI`` -- Pydantic model surface.
* ``TestType7Quantile`` -- the private helper vs. numpy.quantile.
* ``TestComputeYieldCI`` -- empty-ledger raise, determinism across
  seeds, confidence-width monotonicity, and the per-match vs. per-bet
  differentiator that locks in the resampling unit.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from betting_backtester.backtest_result import BacktestResult
from betting_backtester.backtester import (
    RawBacktestOutput,
    SettledBet,
    Side,
)
from betting_backtester.models import Selection
from betting_backtester.reporting import (
    YieldCI,
    _type7_quantile,
    compute_yield_ci,
)

UTC = timezone.utc

T0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
SETTLE_A = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
SETTLE_B = datetime(2024, 1, 8, 14, 0, tzinfo=UTC)
SETTLE_C = datetime(2024, 1, 15, 14, 0, tzinfo=UTC)


# ---------- shared builders -----------------------------------------------


def _make_bet(
    *,
    bet_id: str,
    match_id: str,
    stake: float,
    gross_pnl: float,
    commission: float,
    bankroll_after: float,
    settled_at: datetime = SETTLE_A,
    side: Side = Side.BACK,
    selection: Selection = Selection.HOME,
    price: float = 2.0,
    outcome: Selection = Selection.HOME,
) -> SettledBet:
    placed_at = settled_at - timedelta(hours=2)
    committed = stake if side is Side.BACK else (price - 1.0) * stake
    return SettledBet(
        bet_id=bet_id,
        match_id=match_id,
        selection=selection,
        side=side,
        price=price,
        stake=stake,
        placed_at=placed_at,
        committed_funds=committed,
        settled_at=settled_at,
        outcome=outcome,
        gross_pnl=gross_pnl,
        commission=commission,
        net_pnl=gross_pnl - commission,
        bankroll_after=bankroll_after,
    )


def _result_from_bets(
    bets: Sequence[SettledBet], starting_bankroll: float = 1000.0
) -> BacktestResult:
    raw = RawBacktestOutput(ledger=tuple(bets), rejections=())
    return BacktestResult.from_raw(raw, starting_bankroll=starting_bankroll, t0=T0)


def _mixed_result() -> BacktestResult:
    """A 3-match ledger with mixed outcomes. Used for determinism,
    seed-sensitivity and confidence-width tests."""
    # Running net: -10 / +20 / -5. bankroll_after = 990 / 1010 / 1005.
    a = _make_bet(
        bet_id="A#0000",
        match_id="A",
        stake=10.0,
        gross_pnl=-10.0,
        commission=0.0,
        bankroll_after=990.0,
        settled_at=SETTLE_A,
    )
    b = _make_bet(
        bet_id="B#0000",
        match_id="B",
        stake=20.0,
        gross_pnl=30.0,
        commission=0.0,
        bankroll_after=1020.0,
        settled_at=SETTLE_B,
    )
    c = _make_bet(
        bet_id="C#0000",
        match_id="C",
        stake=15.0,
        gross_pnl=-15.0,
        commission=0.0,
        bankroll_after=1005.0,
        settled_at=SETTLE_C,
    )
    return _result_from_bets([a, b, c])


# ---------- YieldCI Pydantic surface --------------------------------------


class TestYieldCI:
    def _valid(self, **overrides: object) -> YieldCI:
        kwargs: dict[str, object] = {
            "mean": 0.05,
            "lower": 0.0,
            "upper": 0.1,
            "confidence": 0.95,
            "n_resamples": 10_000,
            "n_valid_resamples": 10_000,
            "n_match_groups": 50,
        }
        kwargs.update(overrides)
        return YieldCI(**kwargs)  # type: ignore[arg-type]

    def test_valid_construction(self) -> None:
        ci = self._valid()
        assert ci.mean == 0.05
        assert ci.lower <= ci.upper

    def test_lower_above_upper_rejected(self) -> None:
        with pytest.raises(ValidationError, match="<= upper"):
            self._valid(lower=0.2, upper=0.1)

    def test_mean_outside_bounds_is_allowed(self) -> None:
        # Bootstrap mean on a skewed distribution can sit outside
        # percentile bounds; the docstring explicitly permits this.
        ci = self._valid(mean=0.5, lower=0.0, upper=0.1)
        assert ci.mean == 0.5

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1])
    def test_confidence_must_be_in_open_unit_interval(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            self._valid(confidence=bad)

    def test_rejects_n_resamples_below_one(self) -> None:
        with pytest.raises(ValidationError):
            self._valid(n_resamples=0)

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValidationError):
            self._valid(n_valid_resamples=-1)
        with pytest.raises(ValidationError):
            self._valid(n_match_groups=-1)

    def test_frozen(self) -> None:
        ci = self._valid()
        with pytest.raises(ValidationError):
            ci.mean = 0.99


# ---------- _type7_quantile ------------------------------------------------


class TestType7Quantile:
    @pytest.mark.parametrize(
        "values,q",
        [
            ([1.0, 2.0, 3.0, 4.0, 5.0], 0.0),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 0.25),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 0.5),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 0.75),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 1.0),
            ([1.0, 3.0, 7.0, 15.0], 0.3),
            ([-5.0, 0.0, 5.0, 10.0, 20.0], 0.9),
        ],
    )
    def test_matches_numpy_linear(self, values: list[float], q: float) -> None:
        expected = float(np.quantile(values, q, method="linear"))
        assert _type7_quantile(sorted(values), q) == pytest.approx(expected)

    def test_single_value(self) -> None:
        assert _type7_quantile([42.0], 0.5) == 42.0
        assert _type7_quantile([42.0], 0.0) == 42.0
        assert _type7_quantile([42.0], 1.0) == 42.0


# ---------- compute_yield_ci ----------------------------------------------


class TestComputeYieldCI:
    def test_empty_ledger_raises(self) -> None:
        raw = RawBacktestOutput(ledger=(), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)
        with pytest.raises(ValueError, match="empty ledger"):
            compute_yield_ci(result, seed=0)

    def test_rejects_bad_confidence(self) -> None:
        result = _mixed_result()
        with pytest.raises(ValueError, match="confidence"):
            compute_yield_ci(result, seed=0, confidence=0.0)
        with pytest.raises(ValueError, match="confidence"):
            compute_yield_ci(result, seed=0, confidence=1.0)

    def test_rejects_n_resamples_below_one(self) -> None:
        result = _mixed_result()
        with pytest.raises(ValueError, match="n_resamples"):
            compute_yield_ci(result, seed=0, n_resamples=0)

    def test_constant_yield_ledger_produces_point_ci(self) -> None:
        # Every match has identical yield (5/10 = 0.5), so every bootstrap
        # resample has yield 0.5, and the CI collapses to a point.
        bets = [
            _make_bet(
                bet_id=f"{m}#0000",
                match_id=m,
                stake=10.0,
                gross_pnl=5.0,
                commission=0.0,
                bankroll_after=1000.0 + 5.0 * (i + 1),
                settled_at=T0 + timedelta(hours=2 * (i + 1)),
            )
            for i, m in enumerate(["A", "B", "C", "D"])
        ]
        result = _result_from_bets(bets)
        ci = compute_yield_ci(result, seed=0, n_resamples=200)
        assert ci.mean == pytest.approx(0.5)
        assert ci.lower == pytest.approx(0.5)
        assert ci.upper == pytest.approx(0.5)
        assert ci.n_match_groups == 4
        assert ci.n_valid_resamples == 200

    def test_determinism_same_inputs_byte_equal(self) -> None:
        result = _mixed_result()
        ci1 = compute_yield_ci(result, seed=42, n_resamples=500, confidence=0.95)
        ci2 = compute_yield_ci(result, seed=42, n_resamples=500, confidence=0.95)
        assert ci1 == ci2

    def test_different_seeds_produce_different_cis(self) -> None:
        result = _mixed_result()
        ci_a = compute_yield_ci(result, seed=1, n_resamples=500)
        ci_b = compute_yield_ci(result, seed=2, n_resamples=500)
        # At least one of the three summary numbers must differ -- if all
        # three match to byte precision on a non-degenerate ledger, the
        # seed isn't actually reaching the RNG.
        assert (ci_a.mean, ci_a.lower, ci_a.upper) != (
            ci_b.mean,
            ci_b.lower,
            ci_b.upper,
        )

    def test_higher_confidence_gives_wider_bounds(self) -> None:
        result = _mixed_result()
        narrow = compute_yield_ci(result, seed=7, n_resamples=2000, confidence=0.90)
        wide = compute_yield_ci(result, seed=7, n_resamples=2000, confidence=0.99)
        assert wide.lower <= narrow.lower
        assert wide.upper >= narrow.upper

    def test_resampling_is_per_match_not_per_bet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arb-shaped ledger: match A has two bets (back/lay), matches B and
        # C have one each. There are 3 match-groups but 4 bets total.
        #
        # We patch random.Random.choices to (a) capture the population it
        # was handed and (b) always return indices [0, 0, ..., 0]. With
        # per-match grouping, the population has 3 entries and picking
        # group 0 repeatedly pulls BOTH arb bets into the resample; the
        # resulting yield is (sum_A net) / (sum_A stake).
        #
        # If the implementation were per-bet, the population would have 4
        # entries and picking index 0 would pick only the first arb bet.
        # The test's two assertions -- population length and computed mean
        # -- are each sufficient alone; together they pin the invariant
        # from both directions.
        a1 = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=100.0,
            gross_pnl=100.0,
            commission=3.0,
            bankroll_after=1097.0,
            settled_at=SETTLE_A,
        )
        a2 = _make_bet(
            bet_id="A#0001",
            match_id="A",
            stake=40.0,
            gross_pnl=-40.0,
            commission=0.0,
            bankroll_after=1057.0,
            settled_at=SETTLE_A,
            side=Side.LAY,
            selection=Selection.AWAY,
        )
        b = _make_bet(
            bet_id="B#0000",
            match_id="B",
            stake=20.0,
            gross_pnl=-20.0,
            commission=0.0,
            bankroll_after=1037.0,
            settled_at=SETTLE_B,
        )
        c = _make_bet(
            bet_id="C#0000",
            match_id="C",
            stake=20.0,
            gross_pnl=-20.0,
            commission=0.0,
            bankroll_after=1017.0,
            settled_at=SETTLE_C,
        )
        result = _result_from_bets([a1, a2, b, c])

        captured_populations: list[list[int]] = []

        def fake_choices(
            self: random.Random,
            population: Sequence[int],
            weights: object = None,
            *,
            cum_weights: object = None,
            k: int = 1,
        ) -> list[int]:
            captured_populations.append(list(population))
            return [0] * k

        monkeypatch.setattr(random.Random, "choices", fake_choices)

        ci = compute_yield_ci(result, seed=0, n_resamples=1, confidence=0.95)

        # Sanity: choices was called once per resample.
        assert len(captured_populations) == 1
        # Direct proof: the population had one entry per MATCH (3), not
        # one entry per BET (would have been 4).
        assert len(captured_populations[0]) == 3
        assert ci.n_match_groups == 3
        # Computed yield: with picks = [0, 0, 0] all three drawn groups
        # are match A; yield = (3 * net_A) / (3 * turnover_A) = net_A /
        # turnover_A = (97 + -40) / (100 + 40) = 57 / 140.
        assert ci.mean == pytest.approx(57.0 / 140.0)

    def test_n_match_groups_reports_distinct_matches(self) -> None:
        a1 = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=100.0,
            gross_pnl=100.0,
            commission=3.0,
            bankroll_after=1097.0,
            settled_at=SETTLE_A,
        )
        a2 = _make_bet(
            bet_id="A#0001",
            match_id="A",
            stake=40.0,
            gross_pnl=-40.0,
            commission=0.0,
            bankroll_after=1057.0,
            settled_at=SETTLE_A,
            side=Side.LAY,
            selection=Selection.AWAY,
        )
        b = _make_bet(
            bet_id="B#0000",
            match_id="B",
            stake=20.0,
            gross_pnl=-20.0,
            commission=0.0,
            bankroll_after=1037.0,
            settled_at=SETTLE_B,
        )
        result = _result_from_bets([a1, a2, b])
        ci = compute_yield_ci(result, seed=0, n_resamples=100)
        # 3 bets across 2 matches -> n_match_groups == 2
        assert ci.n_match_groups == 2

    def test_n_valid_resamples_equals_n_resamples_under_invariants(self) -> None:
        # Under 4a's stake>0 + non-empty-ledger invariants, no resample
        # can have zero turnover, so the skip branch is never taken.
        result = _mixed_result()
        ci = compute_yield_ci(result, seed=0, n_resamples=1000)
        assert ci.n_valid_resamples == 1000
