"""Tests for the synthetic event source.

Covers configuration validation, stream invariants (event counts, ordering,
fair-odds pricing, per-match pairing, timestamp offsets), frequency
convergence, determinism (same-seed equality, different-seed divergence,
re-callability, robustness to perturbed external RNG state), and
conformance to the :class:`EventSource` protocol (structural + behavioural).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

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


@pytest.fixture
def fair_probs() -> TrueProbabilities:
    return TrueProbabilities(home=0.5, draw=0.25, away=0.25)


@pytest.fixture
def small_config(fair_probs: TrueProbabilities) -> SyntheticGeneratorConfig:
    return SyntheticGeneratorConfig(
        n_matches=20,
        true_probs=fair_probs,
        seed=42,
        start=UTC_START,
    )


@pytest.fixture
def small_generator(
    small_config: SyntheticGeneratorConfig,
) -> SyntheticGenerator:
    return SyntheticGenerator(small_config)


class TestTrueProbabilities:
    def test_valid_construction(self, fair_probs: TrueProbabilities) -> None:
        assert fair_probs.home == 0.5
        assert fair_probs.draw == 0.25
        assert fair_probs.away == 0.25

    @pytest.mark.parametrize(
        "home, draw, away",
        [
            (0.4, 0.4, 0.4),  # sums to 1.2
            (0.2, 0.2, 0.2),  # sums to 0.6
            (0.5, 0.5, 0.1),  # sums to 1.1
            (0.5, 0.25, 0.2),  # sums to 0.95
        ],
    )
    def test_rejects_probs_not_summing_to_one(
        self, home: float, draw: float, away: float
    ) -> None:
        with pytest.raises(ValidationError):
            TrueProbabilities(home=home, draw=draw, away=away)

    @pytest.mark.parametrize("bad", [0.0, -0.01, -1.0])
    def test_rejects_non_positive_probability(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            TrueProbabilities(home=bad, draw=0.5, away=0.5)

    @pytest.mark.parametrize("bad", [1.0, 1.01, 2.0])
    def test_rejects_probability_at_or_above_one(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            TrueProbabilities(home=bad, draw=0.01, away=0.01)

    def test_for_selection_returns_correct_value(
        self, fair_probs: TrueProbabilities
    ) -> None:
        assert fair_probs.for_selection(Selection.HOME) == 0.5
        assert fair_probs.for_selection(Selection.DRAW) == 0.25
        assert fair_probs.for_selection(Selection.AWAY) == 0.25

    def test_is_frozen(self, fair_probs: TrueProbabilities) -> None:
        with pytest.raises(ValidationError):
            fair_probs.home = 0.7


class TestSyntheticGeneratorConfig:
    @pytest.mark.parametrize("n", [0, -1, -100])
    def test_rejects_non_positive_n_matches(
        self, fair_probs: TrueProbabilities, n: int
    ) -> None:
        with pytest.raises(ValueError, match="n_matches"):
            SyntheticGeneratorConfig(
                n_matches=n, true_probs=fair_probs, seed=0, start=UTC_START
            )

    def test_rejects_naive_start(self, fair_probs: TrueProbabilities) -> None:
        with pytest.raises(ValueError, match="UTC"):
            SyntheticGeneratorConfig(
                n_matches=1,
                true_probs=fair_probs,
                seed=0,
                start=datetime(2024, 8, 1, 15, 0),
            )

    def test_rejects_non_utc_start(self, fair_probs: TrueProbabilities) -> None:
        with pytest.raises(ValueError, match="UTC"):
            SyntheticGeneratorConfig(
                n_matches=1,
                true_probs=fair_probs,
                seed=0,
                start=datetime(2024, 8, 1, 15, 0, tzinfo=timezone(timedelta(hours=1))),
            )

    @pytest.mark.parametrize("bad", [timedelta(0), timedelta(hours=-1)])
    def test_rejects_non_positive_fixture_spacing(
        self, fair_probs: TrueProbabilities, bad: timedelta
    ) -> None:
        with pytest.raises(ValueError, match="fixture_spacing"):
            SyntheticGeneratorConfig(
                n_matches=1,
                true_probs=fair_probs,
                seed=0,
                start=UTC_START,
                fixture_spacing=bad,
            )

    @pytest.mark.parametrize("bad", [timedelta(0), timedelta(seconds=-1)])
    def test_rejects_non_positive_match_duration(
        self, fair_probs: TrueProbabilities, bad: timedelta
    ) -> None:
        with pytest.raises(ValueError, match="match_duration"):
            SyntheticGeneratorConfig(
                n_matches=1,
                true_probs=fair_probs,
                seed=0,
                start=UTC_START,
                match_duration=bad,
            )

    @pytest.mark.parametrize("bad", [timedelta(0), timedelta(minutes=-5)])
    def test_rejects_non_positive_odds_lead(
        self, fair_probs: TrueProbabilities, bad: timedelta
    ) -> None:
        with pytest.raises(ValueError, match="odds_lead"):
            SyntheticGeneratorConfig(
                n_matches=1,
                true_probs=fair_probs,
                seed=0,
                start=UTC_START,
                odds_lead=bad,
            )

    def test_rejects_empty_league(self, fair_probs: TrueProbabilities) -> None:
        with pytest.raises(ValueError, match="league"):
            SyntheticGeneratorConfig(
                n_matches=1,
                true_probs=fair_probs,
                seed=0,
                start=UTC_START,
                league="",
            )

    def test_rejects_empty_season(self, fair_probs: TrueProbabilities) -> None:
        with pytest.raises(ValueError, match="season"):
            SyntheticGeneratorConfig(
                n_matches=1,
                true_probs=fair_probs,
                seed=0,
                start=UTC_START,
                season="",
            )

    def test_is_frozen(self, small_config: SyntheticGeneratorConfig) -> None:
        with pytest.raises(
            Exception
        ):  # FrozenInstanceError, subclass of AttributeError  # noqa: PT011
            small_config.n_matches = 99  # type: ignore[misc]


class TestStream:
    def test_emits_two_events_per_match(
        self, small_generator: SyntheticGenerator
    ) -> None:
        events = list(small_generator.events())
        assert len(events) == 2 * 20
        odds_count = sum(1 for e in events if isinstance(e, OddsAvailable))
        settled_count = sum(1 for e in events if isinstance(e, MatchSettled))
        assert odds_count == 20
        assert settled_count == 20

    def test_timestamps_non_decreasing(
        self, small_generator: SyntheticGenerator
    ) -> None:
        events = list(small_generator.events())
        timestamps = [e.timestamp for e in events]
        for earlier, later in zip(timestamps, timestamps[1:], strict=False):
            assert earlier <= later

    def test_every_match_has_one_odds_and_one_settlement(
        self, small_generator: SyntheticGenerator
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
        assert len(settled_ids) == len(set(settled_ids)) == 20

    def test_odds_precede_settlement_per_match(
        self, small_generator: SyntheticGenerator
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
        self, small_config: SyntheticGeneratorConfig
    ) -> None:
        gen = SyntheticGenerator(small_config)
        kickoffs = {m.match_id: m.kickoff for m in gen.matches}
        for ev in gen.events():
            if isinstance(ev, OddsAvailable):
                expected = kickoffs[ev.snapshot.match_id] - small_config.odds_lead
                assert ev.snapshot.timestamp == expected

    def test_settlement_timestamp_equals_kickoff_plus_duration(
        self, small_config: SyntheticGeneratorConfig
    ) -> None:
        gen = SyntheticGenerator(small_config)
        kickoffs = {m.match_id: m.kickoff for m in gen.matches}
        for ev in gen.events():
            if isinstance(ev, MatchSettled):
                expected = kickoffs[ev.result.match_id] + small_config.match_duration
                assert ev.result.timestamp == expected

    def test_fair_odds_match_true_probs(
        self,
        small_generator: SyntheticGenerator,
        fair_probs: TrueProbabilities,
    ) -> None:
        for ev in small_generator.events():
            if isinstance(ev, OddsAvailable):
                snap = ev.snapshot
                assert snap.home.back_price == pytest.approx(1 / fair_probs.home)
                assert snap.home.lay_price == pytest.approx(1 / fair_probs.home)
                assert snap.draw.back_price == pytest.approx(1 / fair_probs.draw)
                assert snap.draw.lay_price == pytest.approx(1 / fair_probs.draw)
                assert snap.away.back_price == pytest.approx(1 / fair_probs.away)
                assert snap.away.lay_price == pytest.approx(1 / fair_probs.away)

    def test_placeholder_scores_are_consistent_with_outcome(
        self, small_generator: SyntheticGenerator
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

    def test_match_ids_unique_and_formatted(
        self,
        small_generator: SyntheticGenerator,
        small_config: SyntheticGeneratorConfig,
    ) -> None:
        ids = [m.match_id for m in small_generator.matches]
        assert len(ids) == len(set(ids)) == small_config.n_matches
        for i, match_id in enumerate(ids):
            assert match_id == f"{small_config.league}-{small_config.season}-{i:04d}"

    def test_kickoffs_strictly_increase_by_fixture_spacing(
        self,
        small_generator: SyntheticGenerator,
        small_config: SyntheticGeneratorConfig,
    ) -> None:
        kickoffs = [m.kickoff for m in small_generator.matches]
        for i, ts in enumerate(kickoffs):
            assert ts == small_config.start + i * small_config.fixture_spacing

    def test_teams_are_synthetic_and_unique_per_match(
        self, small_generator: SyntheticGenerator
    ) -> None:
        seen: set[str] = set()
        for match in small_generator.matches:
            assert match.home.startswith("SYN-T")
            assert match.away.startswith("SYN-T")
            assert match.home != match.away
            seen.add(match.home)
            seen.add(match.away)
        assert len(seen) == 2 * len(small_generator.matches)

    def test_outcome_frequencies_converge(self, fair_probs: TrueProbabilities) -> None:
        # n=10_000, probs (0.5, 0.25, 0.25): per-bucket stdev ≤ sqrt(0.25/10_000)
        # = 0.005, so 0.02 is a ~4σ bound — comfortably tight while insensitive
        # to the specific seed.
        cfg = SyntheticGeneratorConfig(
            n_matches=10_000,
            true_probs=fair_probs,
            seed=42,
            start=UTC_START,
        )
        gen = SyntheticGenerator(cfg)
        outcomes: list[Selection] = []
        for ev in gen.events():
            if isinstance(ev, MatchSettled):
                outcomes.append(ev.result.outcome)
        counts = Counter(outcomes)
        assert abs(counts[Selection.HOME] / 10_000 - fair_probs.home) < 0.02
        assert abs(counts[Selection.DRAW] / 10_000 - fair_probs.draw) < 0.02
        assert abs(counts[Selection.AWAY] / 10_000 - fair_probs.away) < 0.02


class TestDeterminism:
    def test_same_seed_produces_identical_stream(
        self, small_config: SyntheticGeneratorConfig
    ) -> None:
        a = list(SyntheticGenerator(small_config).events())
        b = list(SyntheticGenerator(small_config).events())
        assert a == b

    def test_different_seed_produces_different_outcomes(
        self, fair_probs: TrueProbabilities
    ) -> None:
        def outcomes(seed: int) -> list[Selection]:
            cfg = SyntheticGeneratorConfig(
                n_matches=100,
                true_probs=fair_probs,
                seed=seed,
                start=UTC_START,
            )
            result: list[Selection] = []
            for ev in SyntheticGenerator(cfg).events():
                if isinstance(ev, MatchSettled):
                    result.append(ev.result.outcome)
            return result

        assert outcomes(1) != outcomes(2)

    def test_events_is_recallable(self, small_generator: SyntheticGenerator) -> None:
        first = list(small_generator.events())
        second = list(small_generator.events())
        assert first == second

    def test_determinism_under_perturbed_external_random_state(
        self, small_config: SyntheticGeneratorConfig
    ) -> None:
        # The generator owns its own random.Random(seed) and must not depend
        # on any global RNG. Reseeding and exercising the module-level
        # random.* API between runs must not change the emitted stream.
        gen = SyntheticGenerator(small_config)
        first = list(gen.events())
        random.seed(999)
        for _ in range(50):
            random.random()
        second = list(gen.events())
        assert first == second

    def test_concurrently_iterating_events_yield_equal_sequences(
        self, small_generator: SyntheticGenerator
    ) -> None:
        # Two iterators drawn from the same generator must be independent:
        # advancing one must not consume from the other. Guards against a
        # shared-state regression (e.g. caching the iterator inside the
        # generator instance).
        it1 = small_generator.events()
        it2 = small_generator.events()
        assert next(it1) == next(it2)
        for a, b in zip(it1, it2, strict=True):
            assert a == b


class TestEventSourceProtocolConformance:
    def test_is_instance_of_event_source(
        self, small_generator: SyntheticGenerator
    ) -> None:
        # Structural check via runtime_checkable Protocol.
        assert isinstance(small_generator, EventSource)

    def test_events_returns_iterator_not_just_iterable(
        self, small_generator: SyntheticGenerator
    ) -> None:
        # Iterator[Event] must be a true iterator, not merely iterable.
        # Lists/tuples satisfy Iterable but not Iterator, so this guards
        # against silent drift from ``Iterator[Event]`` to ``list[Event]``.
        assert isinstance(small_generator.events(), Iterator)

    def test_events_yields_only_event_instances(
        self, small_generator: SyntheticGenerator
    ) -> None:
        # Behavioural check: every yielded item is one of the two concrete
        # Event variants declared in models.Event.
        count = 0
        for ev in small_generator.events():
            assert isinstance(ev, OddsAvailable | MatchSettled)
            count += 1
        assert count > 0
