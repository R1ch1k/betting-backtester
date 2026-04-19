"""Tests for the arbitrage-ready event source.

Covers configuration validation, the two :class:`ArbSchedule`
implementations (construction validation, RNG-consumption contract,
endpoints), stream invariants (event counts, timestamp ordering,
per-match pairing), price-construction invariants on both arb and
non-arb matches, determinism (same-seed equality, re-callability),
conformance to :class:`EventSource` (structural + behavioural), and
the cross-generator outcome-equivalence property that falls out of
the pinned RNG-draw order under a :class:`FixedArbSchedule`.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from betting_backtester.arbitrage_generator import (
    ArbitrageGenerator,
    ArbitrageGeneratorConfig,
    ArbSchedule,
    BernoulliArbSchedule,
    FixedArbSchedule,
)
from betting_backtester.event_source import EventSource
from betting_backtester.models import (
    MatchSettled,
    OddsAvailable,
    Selection,
)
from betting_backtester.synthetic import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
    TrueProbabilities,
)

UTC_START = datetime(2024, 8, 1, 15, 0, tzinfo=timezone.utc)

# The book-invariant tolerance mirrors the generator's internal
# _ARB_BOOK_TOLERANCE. Exposed as a test-file constant rather than
# imported from the module under test so a change there has to be
# matched by a (visible) change here.
_BOOK_TOL = 1e-9


@pytest.fixture
def fair_probs() -> TrueProbabilities:
    # Deliberately asymmetric so any off-by-selection bug in the
    # price-construction code would shift the per-selection sums in
    # a detectable way.
    return TrueProbabilities(home=0.5, draw=0.3, away=0.2)


@pytest.fixture
def empty_schedule() -> FixedArbSchedule:
    return FixedArbSchedule(frozenset())


@pytest.fixture
def small_config(
    fair_probs: TrueProbabilities,
    empty_schedule: FixedArbSchedule,
) -> ArbitrageGeneratorConfig:
    return ArbitrageGeneratorConfig(
        n_matches=20,
        true_probs=fair_probs,
        seed=42,
        start=UTC_START,
        schedule=empty_schedule,
    )


@pytest.fixture
def small_generator(
    small_config: ArbitrageGeneratorConfig,
) -> ArbitrageGenerator:
    return ArbitrageGenerator(small_config)


def _build_config(
    fair_probs: TrueProbabilities,
    schedule: ArbSchedule,
    **overrides: object,
) -> ArbitrageGeneratorConfig:
    """Convenience builder for configs where only a few fields change."""
    kwargs: dict[str, object] = {
        "n_matches": 10,
        "true_probs": fair_probs,
        "seed": 0,
        "start": UTC_START,
        "schedule": schedule,
    }
    kwargs.update(overrides)
    return ArbitrageGeneratorConfig(**kwargs)  # type: ignore[arg-type]


class TestArbScheduleProtocol:
    def test_fixed_is_instance(self) -> None:
        assert isinstance(FixedArbSchedule(frozenset({1, 2})), ArbSchedule)

    def test_bernoulli_is_instance(self) -> None:
        assert isinstance(BernoulliArbSchedule(0.1), ArbSchedule)

    def test_non_schedule_is_not_instance(self) -> None:
        assert not isinstance(object(), ArbSchedule)
        assert not isinstance(42, ArbSchedule)


class TestFixedArbSchedule:
    def test_accepts_empty(self) -> None:
        schedule = FixedArbSchedule(frozenset())
        assert schedule.arb_positions == frozenset()

    def test_accepts_frozenset(self) -> None:
        schedule = FixedArbSchedule(frozenset({0, 3, 7}))
        assert schedule.arb_positions == frozenset({0, 3, 7})

    def test_accepts_iterable_and_coerces_to_frozenset(self) -> None:
        schedule = FixedArbSchedule([0, 3, 3, 7])
        assert schedule.arb_positions == frozenset({0, 3, 7})

    def test_has_arb_returns_true_for_listed_positions(self) -> None:
        schedule = FixedArbSchedule(frozenset({2, 5}))
        rng = random.Random(0)
        assert schedule.has_arb(2, rng) is True
        assert schedule.has_arb(5, rng) is True

    def test_has_arb_returns_false_for_other_positions(self) -> None:
        schedule = FixedArbSchedule(frozenset({2, 5}))
        rng = random.Random(0)
        assert schedule.has_arb(0, rng) is False
        assert schedule.has_arb(1, rng) is False
        assert schedule.has_arb(999, rng) is False

    def test_has_arb_does_not_consume_rng(self) -> None:
        schedule = FixedArbSchedule(frozenset({2, 5}))
        rng_reference = random.Random(123)
        rng_under_test = random.Random(123)
        schedule.has_arb(0, rng_under_test)
        schedule.has_arb(2, rng_under_test)
        schedule.has_arb(5, rng_under_test)
        schedule.has_arb(999, rng_under_test)
        # Next random() on each must coincide.
        assert rng_under_test.random() == rng_reference.random()

    @pytest.mark.parametrize("bad", [-1, -100])
    def test_rejects_negative_positions(self, bad: int) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            FixedArbSchedule(frozenset({0, bad}))

    def test_rejects_float_positions(self) -> None:
        with pytest.raises(TypeError, match="int"):
            FixedArbSchedule(frozenset({0, 3.0}))  # type: ignore[arg-type]

    def test_rejects_bool_positions(self) -> None:
        # bool is a subclass of int in Python. ``True`` silently acting
        # as ``1`` in a set of match indices would be a surprising bug;
        # FixedArbSchedule rejects it explicitly.
        with pytest.raises(TypeError, match="int"):
            FixedArbSchedule({True, False})  # type: ignore[arg-type]


class TestBernoulliArbSchedule:
    def test_accepts_zero_rate(self) -> None:
        assert BernoulliArbSchedule(0.0).rate == 0.0

    def test_accepts_unit_rate(self) -> None:
        assert BernoulliArbSchedule(1.0).rate == 1.0

    @pytest.mark.parametrize("rate", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_round_trip_rate(self, rate: float) -> None:
        assert BernoulliArbSchedule(rate).rate == rate

    @pytest.mark.parametrize("bad", [-0.01, -1.0, 1.01, 2.0])
    def test_rejects_out_of_range_rate(self, bad: float) -> None:
        with pytest.raises(ValueError, match="rate"):
            BernoulliArbSchedule(bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_rate(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            BernoulliArbSchedule(bad)

    def test_has_arb_consumes_one_rng_draw_per_call(self) -> None:
        schedule = BernoulliArbSchedule(0.5)
        rng_reference = random.Random(123)
        rng_under_test = random.Random(123)
        schedule.has_arb(0, rng_under_test)
        schedule.has_arb(1, rng_under_test)
        # After two has_arb calls, rng_under_test should be two draws
        # ahead of a fresh rng seeded identically.
        rng_reference.random()
        rng_reference.random()
        assert rng_under_test.random() == rng_reference.random()

    def test_rate_zero_is_never_arb(self) -> None:
        schedule = BernoulliArbSchedule(0.0)
        rng = random.Random(0)
        for i in range(1000):
            assert schedule.has_arb(i, rng) is False

    def test_rate_one_is_always_arb(self) -> None:
        schedule = BernoulliArbSchedule(1.0)
        rng = random.Random(0)
        for i in range(1000):
            assert schedule.has_arb(i, rng) is True


class TestArbitrageGeneratorConfig:
    @pytest.mark.parametrize("n", [0, -1, -100])
    def test_rejects_non_positive_n_matches(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
        n: int,
    ) -> None:
        with pytest.raises(ValueError, match="n_matches"):
            _build_config(fair_probs, empty_schedule, n_matches=n)

    def test_rejects_naive_start(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
    ) -> None:
        with pytest.raises(ValueError, match="UTC"):
            _build_config(
                fair_probs,
                empty_schedule,
                start=datetime(2024, 8, 1, 15, 0),
            )

    def test_rejects_non_utc_start(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
    ) -> None:
        with pytest.raises(ValueError, match="UTC"):
            _build_config(
                fair_probs,
                empty_schedule,
                start=datetime(
                    2024, 8, 1, 15, 0, tzinfo=timezone(timedelta(hours=1))
                ),
            )

    @pytest.mark.parametrize("bad", [0.0, -0.01, 0.1, 0.2])
    def test_rejects_arb_margin_out_of_range(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
        bad: float,
    ) -> None:
        with pytest.raises(ValueError, match="arb_margin"):
            _build_config(fair_probs, empty_schedule, arb_margin=bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_arb_margin(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
        bad: float,
    ) -> None:
        with pytest.raises(ValueError, match="arb_margin"):
            _build_config(fair_probs, empty_schedule, arb_margin=bad)

    @pytest.mark.parametrize("bad", [-0.01, 0.1, 0.5])
    def test_rejects_half_spread_out_of_range(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
        bad: float,
    ) -> None:
        with pytest.raises(ValueError, match="half_spread"):
            _build_config(fair_probs, empty_schedule, half_spread=bad)

    def test_accepts_zero_half_spread(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
    ) -> None:
        cfg = _build_config(fair_probs, empty_schedule, half_spread=0.0)
        assert cfg.half_spread == 0.0

    @pytest.mark.parametrize("bad", [timedelta(0), timedelta(hours=-1)])
    def test_rejects_non_positive_fixture_spacing(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
        bad: timedelta,
    ) -> None:
        with pytest.raises(ValueError, match="fixture_spacing"):
            _build_config(fair_probs, empty_schedule, fixture_spacing=bad)

    @pytest.mark.parametrize("bad", [timedelta(0), timedelta(seconds=-1)])
    def test_rejects_non_positive_match_duration(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
        bad: timedelta,
    ) -> None:
        with pytest.raises(ValueError, match="match_duration"):
            _build_config(fair_probs, empty_schedule, match_duration=bad)

    @pytest.mark.parametrize("bad", [timedelta(0), timedelta(minutes=-5)])
    def test_rejects_non_positive_odds_lead(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
        bad: timedelta,
    ) -> None:
        with pytest.raises(ValueError, match="odds_lead"):
            _build_config(fair_probs, empty_schedule, odds_lead=bad)

    def test_rejects_empty_league(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
    ) -> None:
        with pytest.raises(ValueError, match="league"):
            _build_config(fair_probs, empty_schedule, league="")

    def test_rejects_empty_season(
        self,
        fair_probs: TrueProbabilities,
        empty_schedule: FixedArbSchedule,
    ) -> None:
        with pytest.raises(ValueError, match="season"):
            _build_config(fair_probs, empty_schedule, season="")

    def test_rejects_non_schedule_object(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        with pytest.raises(TypeError, match="schedule"):
            _build_config(fair_probs, 42)  # type: ignore[arg-type]

    def test_rejects_probs_that_would_produce_back_price_at_or_below_one(
        self,
        empty_schedule: FixedArbSchedule,
    ) -> None:
        # With arb_margin=0.05 and one selection with p=0.96, the arb
        # leg would produce back_price = 1 / (0.96 * 0.95) = 1.0965 --
        # fine. But 1 / (0.96 * (1 + 0.05)) on the non-arb leg gives
        # back_price = 0.992..., violating SelectionOdds. The config
        # validator catches this before the event stream starts.
        too_peaked = TrueProbabilities(home=0.96, draw=0.02, away=0.02)
        with pytest.raises(ValueError, match="back_price"):
            _build_config(
                too_peaked,
                empty_schedule,
                arb_margin=0.05,
                half_spread=0.05,
            )

    def test_is_frozen(self, small_config: ArbitrageGeneratorConfig) -> None:
        with pytest.raises(FrozenInstanceError):
            small_config.n_matches = 99  # type: ignore[misc]


class TestFixedArbSchedulePositionBounds:
    def test_rejects_positions_beyond_n_matches(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        schedule = FixedArbSchedule(frozenset({0, 5, 99}))
        cfg = _build_config(fair_probs, schedule, n_matches=10)
        with pytest.raises(ValueError, match=r"out-of-range positions"):
            ArbitrageGenerator(cfg)

    def test_accepts_positions_strictly_below_n_matches(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        schedule = FixedArbSchedule(frozenset({0, 5, 9}))
        cfg = _build_config(fair_probs, schedule, n_matches=10)
        # Must not raise.
        ArbitrageGenerator(cfg)

    def test_rejects_equal_to_n_matches(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        schedule = FixedArbSchedule(frozenset({10}))
        cfg = _build_config(fair_probs, schedule, n_matches=10)
        with pytest.raises(ValueError, match=r"out-of-range positions"):
            ArbitrageGenerator(cfg)


class TestStreamShape:
    def test_emits_two_events_per_match(
        self, small_generator: ArbitrageGenerator
    ) -> None:
        events = list(small_generator.events())
        assert len(events) == 2 * 20
        odds_count = sum(1 for e in events if isinstance(e, OddsAvailable))
        settled_count = sum(1 for e in events if isinstance(e, MatchSettled))
        assert odds_count == 20
        assert settled_count == 20

    def test_timestamps_non_decreasing(
        self, small_generator: ArbitrageGenerator
    ) -> None:
        events = list(small_generator.events())
        for earlier, later in zip(
            (e.timestamp for e in events),
            (e.timestamp for e in events[1:]),
            strict=False,
        ):
            assert earlier <= later

    def test_every_match_has_one_odds_and_one_settlement(
        self, small_generator: ArbitrageGenerator
    ) -> None:
        odds_ids: list[str] = []
        settled_ids: list[str] = []
        for ev in small_generator.events():
            if isinstance(ev, OddsAvailable):
                odds_ids.append(ev.snapshot.match_id)
            else:
                settled_ids.append(ev.result.match_id)
        assert set(odds_ids) == set(settled_ids)
        assert len(odds_ids) == len(set(odds_ids)) == 20

    def test_odds_precede_settlement_per_match(
        self, small_generator: ArbitrageGenerator
    ) -> None:
        odds_ts: dict[str, datetime] = {}
        settled_ts: dict[str, datetime] = {}
        for ev in small_generator.events():
            if isinstance(ev, OddsAvailable):
                odds_ts[ev.snapshot.match_id] = ev.timestamp
            else:
                settled_ts[ev.result.match_id] = ev.timestamp
        for match_id, ts in odds_ts.items():
            assert ts < settled_ts[match_id]

    def test_odds_timestamp_equals_kickoff_minus_lead(
        self, small_config: ArbitrageGeneratorConfig
    ) -> None:
        gen = ArbitrageGenerator(small_config)
        kickoffs = {m.match_id: m.kickoff for m in gen.matches.values()}
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                expected = kickoffs[ev.snapshot.match_id] - small_config.odds_lead
                assert ev.snapshot.timestamp == expected

    def test_settlement_timestamp_equals_kickoff_plus_duration(
        self, small_config: ArbitrageGeneratorConfig
    ) -> None:
        gen = ArbitrageGenerator(small_config)
        kickoffs = {m.match_id: m.kickoff for m in gen.matches.values()}
        for ev in gen.events():
            if isinstance(ev, MatchSettled):
                expected = kickoffs[ev.result.match_id] + small_config.match_duration
                assert ev.result.timestamp == expected

    def test_match_ids_unique_and_formatted(
        self,
        small_generator: ArbitrageGenerator,
        small_config: ArbitrageGeneratorConfig,
    ) -> None:
        ids = [m.match_id for m in small_generator.matches.values()]
        assert len(ids) == len(set(ids)) == small_config.n_matches
        for i, match_id in enumerate(ids):
            assert match_id == f"{small_config.league}-{small_config.season}-{i:04d}"

    def test_kickoffs_strictly_increase_by_fixture_spacing(
        self,
        small_generator: ArbitrageGenerator,
        small_config: ArbitrageGeneratorConfig,
    ) -> None:
        kickoffs = [m.kickoff for m in small_generator.matches.values()]
        for i, ts in enumerate(kickoffs):
            assert ts == small_config.start + i * small_config.fixture_spacing

    def test_placeholder_scores_are_consistent_with_outcome(
        self, small_generator: ArbitrageGenerator
    ) -> None:
        allowed = {
            Selection.HOME: (1, 0),
            Selection.AWAY: (0, 1),
            Selection.DRAW: (1, 1),
        }
        for ev in small_generator.events():
            if isinstance(ev, MatchSettled):
                assert (ev.result.home_goals, ev.result.away_goals) == allowed[
                    ev.result.outcome
                ]

    def test_matches_is_read_only_mapping(
        self, small_generator: ArbitrageGenerator
    ) -> None:
        with pytest.raises(TypeError):
            small_generator.matches["x"] = None  # type: ignore[index]


class TestNonArbPrices:
    def test_non_arb_back_book_sums_to_one_plus_half_spread(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset()),
            n_matches=5,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                book = (
                    1 / ev.snapshot.home.back_price
                    + 1 / ev.snapshot.draw.back_price
                    + 1 / ev.snapshot.away.back_price
                )
                assert book == pytest.approx(1.0 + 0.01, abs=_BOOK_TOL)

    def test_non_arb_back_is_fair_over_one_plus_half_spread(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset()),
            n_matches=3,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                snap = ev.snapshot
                assert snap.home.back_price == pytest.approx(
                    1.0 / (fair_probs.home * 1.01)
                )
                assert snap.draw.back_price == pytest.approx(
                    1.0 / (fair_probs.draw * 1.01)
                )
                assert snap.away.back_price == pytest.approx(
                    1.0 / (fair_probs.away * 1.01)
                )

    def test_uniform_lay_over_back_ratio(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        # lay / back = (1 + half_spread) ** 2 everywhere.
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset()),
            n_matches=5,
            half_spread=0.015,
        )
        gen = ArbitrageGenerator(cfg)
        expected_ratio = (1.0 + 0.015) ** 2
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                for so in (ev.snapshot.home, ev.snapshot.draw, ev.snapshot.away):
                    assert so.lay_price / so.back_price == pytest.approx(
                        expected_ratio
                    )

    def test_back_leq_lay_per_selection(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset()),
            n_matches=3,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                for so in (ev.snapshot.home, ev.snapshot.draw, ev.snapshot.away):
                    assert so.back_price <= so.lay_price

    def test_zero_half_spread_collapses_back_and_lay(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset()),
            n_matches=2,
            half_spread=0.0,
        )
        gen = ArbitrageGenerator(cfg)
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                for so in (ev.snapshot.home, ev.snapshot.draw, ev.snapshot.away):
                    assert so.back_price == pytest.approx(so.lay_price)


class TestArbPrices:
    def test_arb_back_book_sums_to_one_minus_arb_margin(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset({0, 1, 2, 3, 4})),
            n_matches=5,
            arb_margin=0.02,
        )
        gen = ArbitrageGenerator(cfg)
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                book = (
                    1 / ev.snapshot.home.back_price
                    + 1 / ev.snapshot.draw.back_price
                    + 1 / ev.snapshot.away.back_price
                )
                assert book == pytest.approx(1.0 - 0.02, abs=_BOOK_TOL)

    def test_arb_back_is_fair_over_one_minus_arb_margin(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset({0, 1, 2})),
            n_matches=3,
            arb_margin=0.02,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                snap = ev.snapshot
                assert snap.home.back_price == pytest.approx(
                    1.0 / (fair_probs.home * 0.98)
                )
                assert snap.draw.back_price == pytest.approx(
                    1.0 / (fair_probs.draw * 0.98)
                )
                assert snap.away.back_price == pytest.approx(
                    1.0 / (fair_probs.away * 0.98)
                )

    def test_arb_lay_over_back_matches_non_arb_ratio(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        # The uniform lay/back invariant holds on both arb and non-arb
        # matches: same (1 + half_spread) ** 2 on either branch.
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset({0, 2, 4})),
            n_matches=5,
            arb_margin=0.02,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        expected_ratio = (1.0 + 0.01) ** 2
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                for so in (ev.snapshot.home, ev.snapshot.draw, ev.snapshot.away):
                    assert so.lay_price / so.back_price == pytest.approx(
                        expected_ratio
                    )

    def test_back_leq_lay_on_arb_matches(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset({0, 1})),
            n_matches=3,
            arb_margin=0.02,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                for so in (ev.snapshot.home, ev.snapshot.draw, ev.snapshot.away):
                    assert so.back_price <= so.lay_price


class TestMixedStream:
    def test_fixed_schedule_marks_only_listed_matches_as_arb(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        # First three matches are arbs, last two are not.
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset({0, 1, 2})),
            n_matches=5,
            arb_margin=0.02,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        # OddsAvailable events are emitted in match-index order (both
        # kickoffs are strictly increasing and only OddsAvailable
        # snapshots get tie-broken against MatchSettled, not against
        # each other). That matches the fixture list order.
        arb_flags: list[bool] = []
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                book = (
                    1 / ev.snapshot.home.back_price
                    + 1 / ev.snapshot.draw.back_price
                    + 1 / ev.snapshot.away.back_price
                )
                arb_flags.append(book < 1.0)
        assert arb_flags == [True, True, True, False, False]

    def test_bernoulli_rate_zero_produces_no_arbs(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        cfg = _build_config(
            fair_probs,
            BernoulliArbSchedule(0.0),
            n_matches=50,
            arb_margin=0.02,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                book = (
                    1 / ev.snapshot.home.back_price
                    + 1 / ev.snapshot.draw.back_price
                    + 1 / ev.snapshot.away.back_price
                )
                assert book > 1.0

    def test_bernoulli_rate_one_produces_all_arbs(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        cfg = _build_config(
            fair_probs,
            BernoulliArbSchedule(1.0),
            n_matches=50,
            arb_margin=0.02,
            half_spread=0.01,
        )
        gen = ArbitrageGenerator(cfg)
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                book = (
                    1 / ev.snapshot.home.back_price
                    + 1 / ev.snapshot.draw.back_price
                    + 1 / ev.snapshot.away.back_price
                )
                assert book == pytest.approx(1.0 - 0.02, abs=_BOOK_TOL)

    def test_bernoulli_rate_converges_to_expected_frequency(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        # n=5_000 at rate=0.1: expected std ~= sqrt(0.09 / 5_000) ~= 0.004,
        # so 0.02 is ~5σ -- tight but seed-insensitive.
        cfg = _build_config(
            fair_probs,
            BernoulliArbSchedule(0.1),
            n_matches=5_000,
            arb_margin=0.02,
            half_spread=0.01,
            seed=7,
        )
        gen = ArbitrageGenerator(cfg)
        arb_count = 0
        total = 0
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                book = (
                    1 / ev.snapshot.home.back_price
                    + 1 / ev.snapshot.draw.back_price
                    + 1 / ev.snapshot.away.back_price
                )
                if book < 1.0:
                    arb_count += 1
                total += 1
        assert abs(arb_count / total - 0.1) < 0.02


class TestDeterminism:
    def test_same_config_same_stream(
        self,
        small_config: ArbitrageGeneratorConfig,
    ) -> None:
        g1 = ArbitrageGenerator(small_config)
        g2 = ArbitrageGenerator(small_config)
        assert list(g1.events()) == list(g2.events())

    def test_recallable(
        self,
        small_generator: ArbitrageGenerator,
    ) -> None:
        first = list(small_generator.events())
        second = list(small_generator.events())
        assert first == second

    def test_different_seed_different_outcomes(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        schedule = FixedArbSchedule(frozenset())
        cfg_a = _build_config(fair_probs, schedule, n_matches=200, seed=1)
        cfg_b = _build_config(fair_probs, schedule, n_matches=200, seed=2)
        outcomes_a = [
            ev.result.outcome
            for ev in ArbitrageGenerator(cfg_a).events()
            if isinstance(ev, MatchSettled)
        ]
        outcomes_b = [
            ev.result.outcome
            for ev in ArbitrageGenerator(cfg_b).events()
            if isinstance(ev, MatchSettled)
        ]
        assert outcomes_a != outcomes_b

    def test_fixed_and_bernoulli_rate_zero_diverge_on_outcomes(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        # Both schedules produce zero arbs on every match, but their
        # RNG-consumption differs (Fixed: 0 draws/match; Bernoulli: 1
        # draw/match). The outcome stream must therefore differ -- a
        # positive confirmation that BernoulliArbSchedule really is
        # consuming the generator's RNG and not a sibling RNG.
        fixed_cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset()),
            n_matches=200,
            seed=42,
        )
        bern_cfg = _build_config(
            fair_probs,
            BernoulliArbSchedule(0.0),
            n_matches=200,
            seed=42,
        )
        outcomes_fixed = [
            ev.result.outcome
            for ev in ArbitrageGenerator(fixed_cfg).events()
            if isinstance(ev, MatchSettled)
        ]
        outcomes_bern = [
            ev.result.outcome
            for ev in ArbitrageGenerator(bern_cfg).events()
            if isinstance(ev, MatchSettled)
        ]
        assert outcomes_fixed != outcomes_bern


class TestCrossGeneratorOutcomeEquivalence:
    """Under :class:`FixedArbSchedule`, :class:`ArbitrageGenerator` and
    :class:`~betting_backtester.synthetic.SyntheticGenerator` consume
    RNG draws identically (both: one draw per match, outcome sample)
    and therefore produce the same outcome sequence for the same seed
    and ``true_probs``. This is a smoke test for the "schedule
    RNG-draw coupling is pinned" claim in the generator docstring."""

    def test_outcomes_match_synthetic_under_fixed_empty_schedule(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        n = 500
        seed = 123
        arb_cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset()),
            n_matches=n,
            seed=seed,
        )
        syn_cfg = SyntheticGeneratorConfig(
            n_matches=n,
            true_probs=fair_probs,
            seed=seed,
            start=UTC_START,
        )
        arb_outcomes = [
            ev.result.outcome
            for ev in ArbitrageGenerator(arb_cfg).events()
            if isinstance(ev, MatchSettled)
        ]
        syn_outcomes = [
            ev.result.outcome
            for ev in SyntheticGenerator(syn_cfg).events()
            if isinstance(ev, MatchSettled)
        ]
        assert arb_outcomes == syn_outcomes

    def test_outcomes_match_synthetic_under_arbitrary_fixed_schedule(
        self,
        fair_probs: TrueProbabilities,
    ) -> None:
        # FixedArbSchedule.has_arb consumes no RNG draws regardless of
        # which positions it flags; outcome equivalence should hold
        # with arbitrary (bounded) arb positions too.
        n = 200
        seed = 99
        arb_cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset({0, 5, 17, 100, 199})),
            n_matches=n,
            seed=seed,
        )
        syn_cfg = SyntheticGeneratorConfig(
            n_matches=n,
            true_probs=fair_probs,
            seed=seed,
            start=UTC_START,
        )
        arb_outcomes = [
            ev.result.outcome
            for ev in ArbitrageGenerator(arb_cfg).events()
            if isinstance(ev, MatchSettled)
        ]
        syn_outcomes = [
            ev.result.outcome
            for ev in SyntheticGenerator(syn_cfg).events()
            if isinstance(ev, MatchSettled)
        ]
        assert arb_outcomes == syn_outcomes


class TestEventSourceProtocol:
    def test_is_event_source_instance(
        self, small_generator: ArbitrageGenerator
    ) -> None:
        assert isinstance(small_generator, EventSource)

    def test_events_returns_iterator(
        self, small_generator: ArbitrageGenerator
    ) -> None:
        events = small_generator.events()
        assert isinstance(events, Iterator)


class TestOutcomeFrequencyConvergence:
    def test_outcome_frequencies_converge(
        self, fair_probs: TrueProbabilities
    ) -> None:
        # n=10_000 under (0.5, 0.3, 0.2): bucket stdev <= sqrt(0.25 / 10_000)
        # = 0.005, so 0.02 is ~4σ -- insensitive to the specific seed.
        cfg = _build_config(
            fair_probs,
            FixedArbSchedule(frozenset()),
            n_matches=10_000,
            seed=42,
        )
        gen = ArbitrageGenerator(cfg)
        counts: Counter[Selection] = Counter()
        for ev in gen.events():
            if isinstance(ev, MatchSettled):
                counts[ev.result.outcome] += 1
        assert abs(counts[Selection.HOME] / 10_000 - fair_probs.home) < 0.02
        assert abs(counts[Selection.DRAW] / 10_000 - fair_probs.draw) < 0.02
        assert abs(counts[Selection.AWAY] / 10_000 - fair_probs.away) < 0.02
