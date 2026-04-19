"""Tests for :mod:`betting_backtester.football_data`.

Covers the summary invariant, construction-time validation, per-row
parse outcomes on hand-crafted fixtures, multi-file ordering and
determinism, :class:`EventSource` protocol conformance, and event-stream
invariants (timestamps non-decreasing, tie-breaks, OddsAvailable /
MatchSettled pairing per match).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from betting_backtester.event_source import EventSource
from betting_backtester.football_data import (
    FootballDataLoader,
    FootballDataLoadSummary,
)
from betting_backtester.models import (
    MatchSettled,
    OddsAvailable,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "football_data"

# Real football-data.co.uk filenames are just "E0.csv", "D1.csv" etc.,
# so the league-code regex requires stems like "E0" with no extra suffix.
# Fixture scenarios are therefore split into per-scenario subdirectories
# (each holding an "E0.csv", "D1.csv" etc.) rather than sharing one
# directory with descriptive filenames.
CLEAN_MODERN = FIXTURES_DIR / "clean_modern" / "E0.csv"
CLEAN_MODERN_D1 = FIXTURES_DIR / "clean_modern" / "D1.csv"
PRE_TIME_COLUMN = FIXTURES_DIR / "pre_time_column" / "E0.csv"
MALFORMED_ROWS = FIXTURES_DIR / "malformed_rows" / "E0.csv"
BAD_DATES = FIXTURES_DIR / "bad_dates" / "E0.csv"
TIE_BREAK = FIXTURES_DIR / "tie_break" / "E0.csv"
CROSSES_SEASONS = FIXTURES_DIR / "crosses_seasons" / "E0.csv"


def _valid_summary_kwargs(**overrides: int) -> dict[str, int]:
    base: dict[str, int] = {
        "files_processed": 1,
        "rows_seen": 0,
        "matches_loaded": 0,
        "skipped_missing_date": 0,
        "skipped_missing_pinnacle_odds": 0,
        "skipped_missing_result": 0,
        "skipped_invalid_odds": 0,
    }
    base.update(overrides)
    return base


class TestFootballDataLoadSummary:
    def test_valid_construction(self) -> None:
        s = FootballDataLoadSummary(
            **_valid_summary_kwargs(
                rows_seen=10,
                matches_loaded=8,
                skipped_missing_date=1,
                skipped_invalid_odds=1,
            )
        )
        assert s.rows_seen == 10
        assert s.matches_loaded == 8
        assert s.files_processed == 1

    def test_invariant_violation_raises(self) -> None:
        # matches_loaded (5) + skipped (3) == 8, but rows_seen == 10.
        with pytest.raises(ValidationError, match="internally inconsistent"):
            FootballDataLoadSummary(
                **_valid_summary_kwargs(
                    rows_seen=10,
                    matches_loaded=5,
                    skipped_missing_date=1,
                    skipped_missing_pinnacle_odds=1,
                    skipped_missing_result=1,
                )
            )

    @pytest.mark.parametrize(
        "field",
        [
            "files_processed",
            "rows_seen",
            "matches_loaded",
            "skipped_missing_date",
            "skipped_missing_pinnacle_odds",
            "skipped_missing_result",
            "skipped_invalid_odds",
        ],
    )
    def test_negative_field_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            FootballDataLoadSummary(**_valid_summary_kwargs(**{field: -1}))

    def test_is_frozen(self) -> None:
        s = FootballDataLoadSummary(**_valid_summary_kwargs())
        with pytest.raises(ValidationError):
            s.rows_seen = 99  # type: ignore[misc]


class TestConstructionValidation:
    def test_empty_csv_paths_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            FootballDataLoader([])

    @pytest.mark.parametrize(
        "bad_stem",
        [
            "e0",  # lowercase
            "LEAGUE",  # 6 letters, no digit
            "E00",  # 2 digits
            "E-0",  # symbol
            "123",  # only digits
            "",  # empty stem
        ],
    )
    def test_bad_filename_stem_raises(self, tmp_path: Path, bad_stem: str) -> None:
        path = tmp_path / f"{bad_stem}.csv"
        with pytest.raises(ValueError, match="league-code pattern"):
            FootballDataLoader([path])

    def test_missing_required_columns_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "E0.csv"
        path.write_text(
            "Date,HomeTeam,AwayTeam\n12/08/2023,Arsenal,Chelsea\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing required columns"):
            FootballDataLoader([path])

    def test_missing_required_columns_lists_missing_names(self, tmp_path: Path) -> None:
        path = tmp_path / "E0.csv"
        path.write_text(
            "Date,HomeTeam,AwayTeam\n12/08/2023,Arsenal,Chelsea\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as exc:
            FootballDataLoader([path])
        assert "FTHG" in str(exc.value)
        assert "FTAG" in str(exc.value)

    def test_no_header_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "E0.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no header row"):
            FootballDataLoader([path])

    def test_file_crosses_seasons_raises(self) -> None:
        with pytest.raises(ValueError, match="spans multiple football seasons") as exc:
            FootballDataLoader([CROSSES_SEASONS])
        # Both season labels appear in the message.
        assert "2022-23" in str(exc.value)
        assert "2023-24" in str(exc.value)

    def test_duplicate_match_id_across_files_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate match_id"):
            FootballDataLoader([CLEAN_MODERN, CLEAN_MODERN])

    def test_trailing_whitespace_in_headers_is_tolerated(self, tmp_path: Path) -> None:
        # "Date " (trailing space) and "HomeTeam" etc. should all be
        # accepted post-strip.
        path = tmp_path / "E0.csv"
        path.write_text(
            "Div,Date ,Time,HomeTeam ,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA\n"
            "E0,12/08/2023,15:00,Arsenal,Chelsea,1,0,H,2.0,3.5,4.0\n",
            encoding="utf-8",
        )
        loader = FootballDataLoader([path])
        assert loader.load_summary.matches_loaded == 1


class TestCleanModernFile:
    @pytest.fixture
    def loader(self) -> FootballDataLoader:
        return FootballDataLoader([CLEAN_MODERN])

    def test_six_matches_loaded(self, loader: FootballDataLoader) -> None:
        s = loader.load_summary
        assert s.files_processed == 1
        assert s.rows_seen == 6
        assert s.matches_loaded == 6
        assert s.skipped_missing_date == 0
        assert s.skipped_missing_pinnacle_odds == 0
        assert s.skipped_missing_result == 0
        assert s.skipped_invalid_odds == 0

    def test_twelve_events_emitted(self, loader: FootballDataLoader) -> None:
        events = list(loader.events())
        assert len(events) == 12
        odds_count = sum(1 for e in events if isinstance(e, OddsAvailable))
        settled_count = sum(1 for e in events if isinstance(e, MatchSettled))
        assert odds_count == 6
        assert settled_count == 6

    def test_timestamps_non_decreasing(self, loader: FootballDataLoader) -> None:
        timestamps = [e.timestamp for e in loader.events()]
        for earlier, later in zip(timestamps, timestamps[1:], strict=False):
            assert earlier <= later

    def test_every_match_has_one_odds_and_one_settlement(
        self, loader: FootballDataLoader
    ) -> None:
        odds_ids: list[str] = []
        settled_ids: list[str] = []
        for ev in loader.events():
            if isinstance(ev, OddsAvailable):
                odds_ids.append(ev.snapshot.match_id)
            else:
                settled_ids.append(ev.result.match_id)
        assert set(odds_ids) == set(settled_ids)
        assert len(odds_ids) == len(set(odds_ids)) == 6

    def test_match_id_format_with_apostrophe(self, loader: FootballDataLoader) -> None:
        # "Nott'm Forest" normalises to "Nott_m_Forest".
        expected = "E0-2023-08-12-Arsenal-Nott_m_Forest"
        match_ids = {
            e.snapshot.match_id for e in loader.events() if isinstance(e, OddsAvailable)
        }
        assert expected in match_ids

    def test_match_id_format_with_space(self, loader: FootballDataLoader) -> None:
        expected = "E0-2023-08-12-Bournemouth-West_Ham"
        match_ids = {
            e.snapshot.match_id for e in loader.events() if isinstance(e, OddsAvailable)
        }
        assert expected in match_ids

    def test_kickoff_combines_date_and_time(self, loader: FootballDataLoader) -> None:
        target = "E0-2023-08-12-Brighton-Luton"
        expected = datetime(2023, 8, 12, 17, 25, tzinfo=timezone.utc)
        for ev in loader.events():
            if isinstance(ev, OddsAvailable) and ev.snapshot.match_id == target:
                assert ev.snapshot.timestamp == expected
                return
        pytest.fail(f"{target} not found in odds events")

    def test_odds_timestamp_is_kickoff_minus_five_minutes(
        self, loader: FootballDataLoader
    ) -> None:
        target = "E0-2023-08-13-Chelsea-Liverpool"
        expected = datetime(2023, 8, 13, 13, 55, tzinfo=timezone.utc)
        for ev in loader.events():
            if isinstance(ev, OddsAvailable) and ev.snapshot.match_id == target:
                assert ev.snapshot.timestamp == expected
                return
        pytest.fail(f"{target} not found in odds events")

    def test_settlement_timestamp_is_kickoff_plus_two_hours(
        self, loader: FootballDataLoader
    ) -> None:
        target = "E0-2023-08-12-Brighton-Luton"
        expected = datetime(2023, 8, 12, 19, 30, tzinfo=timezone.utc)
        for ev in loader.events():
            if isinstance(ev, MatchSettled) and ev.result.match_id == target:
                assert ev.result.timestamp == expected
                return
        pytest.fail(f"{target} not found in settled events")

    def test_back_equals_lay_equals_bookmaker_price(
        self, loader: FootballDataLoader
    ) -> None:
        target = "E0-2023-08-12-Arsenal-Nott_m_Forest"
        for ev in loader.events():
            if isinstance(ev, OddsAvailable) and ev.snapshot.match_id == target:
                snap = ev.snapshot
                assert snap.home.back_price == snap.home.lay_price == 1.35
                assert snap.draw.back_price == snap.draw.lay_price == 5.00
                assert snap.away.back_price == snap.away.lay_price == 9.00
                return
        pytest.fail(f"{target} not found in odds events")

    def test_result_goals_match_fixture(self, loader: FootballDataLoader) -> None:
        target = "E0-2023-08-12-Brighton-Luton"
        for ev in loader.events():
            if isinstance(ev, MatchSettled) and ev.result.match_id == target:
                # Fixture row: FTHG=4, FTAG=1
                assert ev.result.home_goals == 4
                assert ev.result.away_goals == 1
                return
        pytest.fail(f"{target} not found in settled events")

    def test_same_timestamp_odds_tie_break_by_match_id(
        self, loader: FootballDataLoader
    ) -> None:
        # Two 15:00 kickoffs on 2023-08-12 → two OddsAvailable at 14:55.
        t = datetime(2023, 8, 12, 14, 55, tzinfo=timezone.utc)
        at_t = [e for e in loader.events() if e.timestamp == t]
        assert len(at_t) == 2
        ids: list[str] = []
        for e in at_t:
            assert isinstance(e, OddsAvailable)
            ids.append(e.snapshot.match_id)
        assert ids == sorted(ids)
        # Specifically: Arsenal sorts before Bournemouth.
        assert ids[0].startswith("E0-2023-08-12-Arsenal")
        assert ids[1].startswith("E0-2023-08-12-Bournemouth")


class TestPreTimeColumnFile:
    @pytest.fixture
    def loader(self) -> FootballDataLoader:
        return FootballDataLoader([PRE_TIME_COLUMN])

    def test_four_matches_loaded(self, loader: FootballDataLoader) -> None:
        assert loader.load_summary.matches_loaded == 4
        assert loader.load_summary.rows_seen == 4

    def test_default_kickoff_time_when_time_column_absent(
        self, loader: FootballDataLoader
    ) -> None:
        # No Time column → default 15:00 kickoff, OddsAvailable at 14:55.
        expected = datetime(2000, 11, 4, 14, 55, tzinfo=timezone.utc)
        for ev in loader.events():
            if isinstance(ev, OddsAvailable) and "2000-11-04" in ev.snapshot.match_id:
                assert ev.snapshot.timestamp == expected

    def test_legacy_ph_pd_pa_columns_used(self, loader: FootballDataLoader) -> None:
        # Row 1: Arsenal vs Leeds, PH=2.10, PD=3.25, PA=3.40
        target = "E0-2000-11-04-Arsenal-Leeds"
        for ev in loader.events():
            if isinstance(ev, OddsAvailable) and ev.snapshot.match_id == target:
                assert ev.snapshot.home.back_price == 2.10
                assert ev.snapshot.draw.back_price == 3.25
                assert ev.snapshot.away.back_price == 3.40
                return
        pytest.fail(f"{target} not found in odds events")


class TestMalformedRows:
    @pytest.fixture
    def loader(self) -> FootballDataLoader:
        return FootballDataLoader([MALFORMED_ROWS])

    def test_summary_counts_match_expected(self, loader: FootballDataLoader) -> None:
        s = loader.load_summary
        assert s.rows_seen == 6
        assert s.matches_loaded == 2
        assert s.skipped_missing_date == 1
        assert s.skipped_missing_pinnacle_odds == 1
        assert s.skipped_missing_result == 1
        assert s.skipped_invalid_odds == 1

    def test_row_accounting_invariant_holds(self, loader: FootballDataLoader) -> None:
        # The FootballDataLoadSummary validator already enforces this at
        # construction, so this test is belt-and-braces — if the loader ever
        # returned an internally-inconsistent summary the construction would
        # have failed, which would show up as a different failure mode.
        s = loader.load_summary
        skipped = (
            s.skipped_missing_date
            + s.skipped_missing_pinnacle_odds
            + s.skipped_missing_result
            + s.skipped_invalid_odds
        )
        assert s.matches_loaded + skipped == s.rows_seen

    def test_warning_logged_for_each_skipped_row(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="betting_backtester.football_data")
        FootballDataLoader([MALFORMED_ROWS])
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == "betting_backtester.football_data"
        ]
        assert len(warnings) == 4

    def test_only_clean_rows_produce_events(self, loader: FootballDataLoader) -> None:
        events = list(loader.events())
        assert len(events) == 4  # 2 loaded matches × 2 events
        ids: set[str] = set()
        for e in events:
            if isinstance(e, OddsAvailable):
                ids.add(e.snapshot.match_id)
            else:
                ids.add(e.result.match_id)
        assert any("Clean1" in i for i in ids)
        assert any("Clean2" in i for i in ids)
        assert not any("Bad" in i for i in ids), "skipped row leaked into events"


class TestBadDates:
    @pytest.fixture
    def loader(self) -> FootballDataLoader:
        return FootballDataLoader([BAD_DATES])

    def test_three_loaded_one_skipped(self, loader: FootballDataLoader) -> None:
        assert loader.load_summary.matches_loaded == 3
        assert loader.load_summary.skipped_missing_date == 1
        assert loader.load_summary.rows_seen == 4

    def test_two_digit_years_parse_to_1999_not_2099(
        self, loader: FootballDataLoader
    ) -> None:
        match_ids = {
            e.snapshot.match_id for e in loader.events() if isinstance(e, OddsAvailable)
        }
        assert "E0-1999-10-15-Alpha-Beta" in match_ids
        assert "E0-1999-10-16-Gamma-Delta" in match_ids

    def test_four_digit_years_parse_alongside_two_digit(
        self, loader: FootballDataLoader
    ) -> None:
        match_ids = {
            e.snapshot.match_id for e in loader.events() if isinstance(e, OddsAvailable)
        }
        assert "E0-1999-10-20-Epsilon-Zeta" in match_ids


class TestTieBreak:
    def test_settlement_sorts_before_odds_at_equal_timestamp(self) -> None:
        loader = FootballDataLoader([TIE_BREAK])
        events = list(loader.events())
        # 14:00 kickoff → settles 16:00; 16:05 kickoff → odds at 16:00.
        t = datetime(2023, 9, 10, 16, 0, tzinfo=timezone.utc)
        at_t = [e for e in events if e.timestamp == t]
        assert len(at_t) == 2
        assert isinstance(at_t[0], MatchSettled)
        assert isinstance(at_t[1], OddsAvailable)


class TestMultiFile:
    def test_stream_sorted_across_files(self) -> None:
        loader = FootballDataLoader([CLEAN_MODERN, CLEAN_MODERN_D1])
        timestamps = [e.timestamp for e in loader.events()]
        for a, b in zip(timestamps, timestamps[1:], strict=False):
            assert a <= b

    def test_both_leagues_present(self) -> None:
        loader = FootballDataLoader([CLEAN_MODERN, CLEAN_MODERN_D1])
        leagues: set[str] = set()
        for e in loader.events():
            match_id = (
                e.snapshot.match_id
                if isinstance(e, OddsAvailable)
                else e.result.match_id
            )
            leagues.add(match_id.split("-", 1)[0])
        assert leagues == {"E0", "D1"}

    def test_match_counts_sum(self) -> None:
        loader = FootballDataLoader([CLEAN_MODERN, CLEAN_MODERN_D1])
        assert loader.load_summary.matches_loaded == 6 + 4
        assert loader.load_summary.rows_seen == 6 + 4
        assert loader.load_summary.files_processed == 2

    def test_determinism_across_input_ordering(self) -> None:
        # User-requested: loading [E0, D1] and [D1, E0] must yield
        # byte-identical event tuples.
        a = tuple(FootballDataLoader([CLEAN_MODERN, CLEAN_MODERN_D1]).events())
        b = tuple(FootballDataLoader([CLEAN_MODERN_D1, CLEAN_MODERN]).events())
        assert a == b


class TestDeterminism:
    def test_events_is_recallable(self) -> None:
        loader = FootballDataLoader([CLEAN_MODERN])
        first = list(loader.events())
        second = list(loader.events())
        assert first == second

    def test_events_is_re_iterable_after_exhaustion(self) -> None:
        # User-requested: exhaust once, call again, assert identical.
        loader = FootballDataLoader([CLEAN_MODERN])
        exhausted = list(loader.events())  # iterator consumed
        again = list(loader.events())
        assert exhausted == again

    def test_two_loader_instances_produce_equal_streams(self) -> None:
        a = tuple(FootballDataLoader([CLEAN_MODERN]).events())
        b = tuple(FootballDataLoader([CLEAN_MODERN]).events())
        assert a == b

    def test_concurrent_iterators_are_independent(self) -> None:
        # Two iterators from the same loader must be independent of each
        # other — advancing one must not consume from the other.
        loader = FootballDataLoader([CLEAN_MODERN])
        it1 = loader.events()
        it2 = loader.events()
        assert next(it1) == next(it2)
        for a, b in zip(it1, it2, strict=True):
            assert a == b


class TestEventSourceProtocolConformance:
    def test_is_instance_of_event_source(self) -> None:
        loader = FootballDataLoader([CLEAN_MODERN])
        assert isinstance(loader, EventSource)

    def test_events_returns_an_iterator(self) -> None:
        loader = FootballDataLoader([CLEAN_MODERN])
        assert isinstance(loader.events(), Iterator)

    def test_events_yields_only_event_instances(self) -> None:
        loader = FootballDataLoader([CLEAN_MODERN])
        count = 0
        for ev in loader.events():
            assert isinstance(ev, OddsAvailable | MatchSettled)
            count += 1
        assert count > 0
