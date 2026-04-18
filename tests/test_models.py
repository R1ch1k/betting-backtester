"""Tests for canonical models and event types."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from betting_backtester.models import (
    Match,
    MatchResult,
    MatchSettled,
    OddsAvailable,
    OddsSnapshot,
    Selection,
    SelectionOdds,
)


@pytest.fixture
def utc_time() -> datetime:
    return datetime(2024, 1, 15, 14, 59, tzinfo=timezone.utc)


@pytest.fixture
def settlement_time(utc_time: datetime) -> datetime:
    return utc_time + timedelta(hours=1, minutes=50)


@pytest.fixture
def selection_odds() -> SelectionOdds:
    return SelectionOdds(back_price=2.0, lay_price=2.02)


@pytest.fixture
def match(utc_time: datetime) -> Match:
    return Match(
        match_id="m1",
        kickoff=utc_time + timedelta(minutes=1),
        league="E0",
        season="2023-24",
        home="Arsenal",
        away="Chelsea",
    )


@pytest.fixture
def snapshot(utc_time: datetime, selection_odds: SelectionOdds) -> OddsSnapshot:
    return OddsSnapshot(
        match_id="m1",
        timestamp=utc_time,
        home=selection_odds,
        draw=SelectionOdds(back_price=3.5, lay_price=3.55),
        away=SelectionOdds(back_price=4.0, lay_price=4.1),
    )


@pytest.fixture
def match_result(settlement_time: datetime) -> MatchResult:
    return MatchResult(
        match_id="m1",
        timestamp=settlement_time,
        home_goals=2,
        away_goals=1,
    )


class TestSelectionOdds:
    def test_valid_construction(self) -> None:
        odds = SelectionOdds(back_price=2.0, lay_price=2.02)
        assert odds.back_price == 2.0
        assert odds.lay_price == 2.02

    def test_back_equal_lay_is_allowed(self) -> None:
        odds = SelectionOdds(back_price=2.5, lay_price=2.5)
        assert odds.back_price == odds.lay_price

    def test_back_above_lay_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SelectionOdds(back_price=2.10, lay_price=2.05)

    @pytest.mark.parametrize("price", [1.0, 0.999, 0.5, 0.0, -1.0])
    def test_back_price_at_or_below_one_is_rejected(self, price: float) -> None:
        with pytest.raises(ValidationError):
            SelectionOdds(back_price=price, lay_price=2.0)

    @pytest.mark.parametrize("price", [1.0, 0.999, 0.5, 0.0, -1.0])
    def test_lay_price_at_or_below_one_is_rejected(self, price: float) -> None:
        with pytest.raises(ValidationError):
            SelectionOdds(back_price=2.0, lay_price=price)

    @pytest.mark.parametrize("price", [1.005, 1.01])
    def test_prices_just_above_one_are_accepted(self, price: float) -> None:
        odds = SelectionOdds(back_price=price, lay_price=price)
        assert odds.back_price == price
        assert odds.lay_price == price

    def test_round_trips_via_model_dump(self, selection_odds: SelectionOdds) -> None:
        restored = SelectionOdds.model_validate(selection_odds.model_dump())
        assert restored == selection_odds

    def test_equality_on_identical_fields(self) -> None:
        a = SelectionOdds(back_price=2.0, lay_price=2.02)
        b = SelectionOdds(back_price=2.0, lay_price=2.02)
        assert a == b

    @pytest.mark.parametrize(
        "field, value",
        [
            ("back_price", 1.9),
            ("lay_price", 2.03),
        ],
    )
    def test_inequality_when_any_field_differs(
        self, selection_odds: SelectionOdds, field: str, value: float
    ) -> None:
        other = selection_odds.model_copy(update={field: value})
        assert other != selection_odds


class TestMatch:
    def test_valid_construction(self, match: Match) -> None:
        assert match.match_id == "m1"
        assert match.home == "Arsenal"
        assert match.away == "Chelsea"

    def test_naive_kickoff_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Match(
                match_id="m1",
                kickoff=datetime(2024, 1, 15, 15, 0),
                league="E0",
                season="2023-24",
                home="Arsenal",
                away="Chelsea",
            )

    def test_non_utc_kickoff_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Match(
                match_id="m1",
                kickoff=datetime(
                    2024, 1, 15, 15, 0, tzinfo=timezone(timedelta(hours=1))
                ),
                league="E0",
                season="2023-24",
                home="Arsenal",
                away="Chelsea",
            )

    def test_home_equals_away_is_rejected(self, utc_time: datetime) -> None:
        with pytest.raises(ValidationError):
            Match(
                match_id="m1",
                kickoff=utc_time,
                league="E0",
                season="2023-24",
                home="Arsenal",
                away="Arsenal",
            )

    @pytest.mark.parametrize("field", ["match_id", "league", "season", "home", "away"])
    def test_empty_required_string_is_rejected(self, match: Match, field: str) -> None:
        data = match.model_dump()
        data[field] = ""
        with pytest.raises(ValidationError):
            Match.model_validate(data)

    def test_round_trips_via_model_dump(self, match: Match) -> None:
        restored = Match.model_validate(match.model_dump())
        assert restored == match

    def test_equality_on_identical_fields(self, match: Match) -> None:
        twin = Match.model_validate(match.model_dump())
        assert twin == match

    @pytest.mark.parametrize(
        "field, value",
        [
            ("match_id", "m2"),
            ("league", "E1"),
            ("season", "2022-23"),
            ("home", "Tottenham"),
            ("away", "Liverpool"),
        ],
    )
    def test_inequality_when_any_field_differs(
        self, match: Match, field: str, value: str
    ) -> None:
        other = match.model_copy(update={field: value})
        assert other != match


class TestOddsSnapshot:
    def test_valid_construction(
        self, snapshot: OddsSnapshot, utc_time: datetime
    ) -> None:
        assert snapshot.match_id == "m1"
        assert snapshot.timestamp == utc_time

    def test_naive_timestamp_is_rejected(self, selection_odds: SelectionOdds) -> None:
        with pytest.raises(ValidationError):
            OddsSnapshot(
                match_id="m1",
                timestamp=datetime(2024, 1, 15, 14, 59),
                home=selection_odds,
                draw=selection_odds,
                away=selection_odds,
            )

    def test_non_utc_timestamp_is_rejected(self, selection_odds: SelectionOdds) -> None:
        with pytest.raises(ValidationError):
            OddsSnapshot(
                match_id="m1",
                timestamp=datetime(
                    2024, 1, 15, 14, 59, tzinfo=timezone(timedelta(hours=1))
                ),
                home=selection_odds,
                draw=selection_odds,
                away=selection_odds,
            )

    def test_empty_match_id_is_rejected(
        self, utc_time: datetime, selection_odds: SelectionOdds
    ) -> None:
        with pytest.raises(ValidationError):
            OddsSnapshot(
                match_id="",
                timestamp=utc_time,
                home=selection_odds,
                draw=selection_odds,
                away=selection_odds,
            )

    @pytest.mark.parametrize(
        "selection, attr",
        [
            (Selection.HOME, "home"),
            (Selection.DRAW, "draw"),
            (Selection.AWAY, "away"),
        ],
    )
    def test_odds_for_returns_expected_selection(
        self, snapshot: OddsSnapshot, selection: Selection, attr: str
    ) -> None:
        assert snapshot.odds_for(selection) == getattr(snapshot, attr)

    def test_round_trips_via_model_dump(self, snapshot: OddsSnapshot) -> None:
        restored = OddsSnapshot.model_validate(snapshot.model_dump())
        assert restored == snapshot

    def test_equality_on_identical_fields(self, snapshot: OddsSnapshot) -> None:
        twin = OddsSnapshot.model_validate(snapshot.model_dump())
        assert twin == snapshot

    @pytest.mark.parametrize(
        "field, value",
        [
            ("match_id", "m2"),
            ("timestamp", datetime(2024, 1, 15, 15, 0, tzinfo=timezone.utc)),
        ],
    )
    def test_inequality_when_scalar_field_differs(
        self, snapshot: OddsSnapshot, field: str, value: object
    ) -> None:
        other = snapshot.model_copy(update={field: value})
        assert other != snapshot

    def test_inequality_when_selection_odds_differ(
        self, snapshot: OddsSnapshot
    ) -> None:
        other = snapshot.model_copy(
            update={"home": SelectionOdds(back_price=1.5, lay_price=1.52)}
        )
        assert other != snapshot


class TestMatchResult:
    def test_valid_construction(self, match_result: MatchResult) -> None:
        assert match_result.match_id == "m1"
        assert match_result.home_goals == 2
        assert match_result.away_goals == 1

    def test_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MatchResult(
                match_id="m1",
                timestamp=datetime(2024, 1, 15, 16, 49),
                home_goals=1,
                away_goals=1,
            )

    def test_non_utc_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MatchResult(
                match_id="m1",
                timestamp=datetime(
                    2024, 1, 15, 16, 49, tzinfo=timezone(timedelta(hours=1))
                ),
                home_goals=1,
                away_goals=1,
            )

    @pytest.mark.parametrize("goals", [-1, -10])
    def test_negative_home_goals_are_rejected(
        self, settlement_time: datetime, goals: int
    ) -> None:
        with pytest.raises(ValidationError):
            MatchResult(
                match_id="m1",
                timestamp=settlement_time,
                home_goals=goals,
                away_goals=0,
            )

    @pytest.mark.parametrize("goals", [-1, -10])
    def test_negative_away_goals_are_rejected(
        self, settlement_time: datetime, goals: int
    ) -> None:
        with pytest.raises(ValidationError):
            MatchResult(
                match_id="m1",
                timestamp=settlement_time,
                home_goals=0,
                away_goals=goals,
            )

    def test_empty_match_id_is_rejected(self, settlement_time: datetime) -> None:
        with pytest.raises(ValidationError):
            MatchResult(
                match_id="",
                timestamp=settlement_time,
                home_goals=1,
                away_goals=0,
            )

    @pytest.mark.parametrize(
        "home_goals, away_goals, expected",
        [
            (2, 1, Selection.HOME),
            (0, 1, Selection.AWAY),
            (1, 1, Selection.DRAW),
            (5, 0, Selection.HOME),
            (0, 3, Selection.AWAY),
            (0, 0, Selection.DRAW),
        ],
    )
    def test_outcome_derives_from_score(
        self,
        settlement_time: datetime,
        home_goals: int,
        away_goals: int,
        expected: Selection,
    ) -> None:
        result = MatchResult(
            match_id="m1",
            timestamp=settlement_time,
            home_goals=home_goals,
            away_goals=away_goals,
        )
        assert result.outcome is expected

    def test_round_trips_via_model_dump(self, match_result: MatchResult) -> None:
        # outcome is a computed field: it appears in the dump but must be
        # recomputed on validation, not carried through as input.
        restored = MatchResult.model_validate(match_result.model_dump())
        assert restored == match_result
        assert restored.outcome is match_result.outcome

    def test_equality_on_identical_fields(self, match_result: MatchResult) -> None:
        twin = MatchResult.model_validate(match_result.model_dump())
        assert twin == match_result

    @pytest.mark.parametrize(
        "field, value",
        [
            ("match_id", "m2"),
            ("home_goals", 3),
            ("away_goals", 2),
        ],
    )
    def test_inequality_when_any_field_differs(
        self, match_result: MatchResult, field: str, value: object
    ) -> None:
        other = match_result.model_copy(update={field: value})
        assert other != match_result


class TestOddsAvailable:
    def test_valid_construction(self, snapshot: OddsSnapshot) -> None:
        event = OddsAvailable(snapshot=snapshot)
        assert event.snapshot == snapshot

    def test_timestamp_delegates_to_snapshot(self, snapshot: OddsSnapshot) -> None:
        event = OddsAvailable(snapshot=snapshot)
        assert event.timestamp == snapshot.timestamp

    def test_round_trips_via_model_dump(self, snapshot: OddsSnapshot) -> None:
        event = OddsAvailable(snapshot=snapshot)
        restored = OddsAvailable.model_validate(event.model_dump())
        assert restored == event

    def test_equality_on_identical_snapshots(self, snapshot: OddsSnapshot) -> None:
        assert OddsAvailable(snapshot=snapshot) == OddsAvailable(snapshot=snapshot)

    def test_inequality_when_snapshot_differs(
        self, snapshot: OddsSnapshot, utc_time: datetime
    ) -> None:
        other_snapshot = snapshot.model_copy(
            update={"timestamp": utc_time + timedelta(seconds=1)}
        )
        assert OddsAvailable(snapshot=snapshot) != OddsAvailable(
            snapshot=other_snapshot
        )


class TestMatchSettled:
    def test_valid_construction(self, match_result: MatchResult) -> None:
        event = MatchSettled(result=match_result)
        assert event.result == match_result

    def test_timestamp_delegates_to_result(self, match_result: MatchResult) -> None:
        event = MatchSettled(result=match_result)
        assert event.timestamp == match_result.timestamp

    def test_round_trips_via_model_dump(self, match_result: MatchResult) -> None:
        event = MatchSettled(result=match_result)
        restored = MatchSettled.model_validate(event.model_dump())
        assert restored == event

    def test_equality_on_identical_results(self, match_result: MatchResult) -> None:
        assert MatchSettled(result=match_result) == MatchSettled(result=match_result)

    def test_inequality_when_result_differs(self, match_result: MatchResult) -> None:
        other_result = match_result.model_copy(update={"home_goals": 5})
        assert MatchSettled(result=match_result) != MatchSettled(result=other_result)
