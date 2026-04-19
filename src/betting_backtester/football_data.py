"""FootballDataLoader: canonical event stream from football-data.co.uk CSVs.

Implements the :class:`~betting_backtester.event_source.EventSource`
protocol backed by local football-data.co.uk historical CSV files. One
``OddsAvailable`` (at ``kickoff - 5min``) and one ``MatchSettled`` (at
``kickoff + 2h``) are emitted per loaded match.

Parsing is eager: all files are read and cached at construction; every
call to :meth:`FootballDataLoader.events` returns a fresh iterator over
the cached tuple and yields a byte-identical sequence.

Approximations and assumptions (worth calling out prominently because
they affect strategy results):

* **Pinnacle as canonical bookmaker.** The loader emits odds from
  Pinnacle's closing line: ``PSH``/``PSD``/``PSA`` first, then
  ``PH``/``PD``/``PA`` as a fallback for older seasons. Pinnacle is the
  research benchmark for closing-line value (sharp market-maker, high
  limits); other bookmakers' columns are ignored.
* **Bookmaker odds as back == lay.** Each :class:`SelectionOdds` is
  constructed with ``back_price == lay_price == bookmaker_price``.
  Bookmaker closing data has no real bid-ask spread, and forcing a
  zero spread here is a deliberate approximation. **Strategies that
  back AND lay on the same market (e.g. arbitrage detection) will see
  fake-arbitrage signals against this data source** and should only be
  run against true exchange data.
* **UTC without conversion.** football-data.co.uk publishes local
  calendar dates/times but does not carry a timezone. The loader
  interprets ``Date``/``Time`` cells as UTC directly. For single-league
  analyses the error is a constant per-match offset; cross-league
  comparisons that depend on absolute instants will see sub-day drift.
* **Season labels assume European football.** The July-to-May season
  convention is hard-coded (a date in month ``>= 7`` opens a new
  season). Leagues with different schedules (MLS, Brazilian Serie A)
  still get deterministic labels, but the label may not match the
  league's official naming.
* **File encoding.** CSVs are read as UTF-8 with BOM tolerance
  (``utf-8-sig``). Historical football-data.co.uk files that were
  published in latin-1 must be re-encoded to UTF-8 before loading; the
  loader raises ``UnicodeDecodeError`` rather than guessing.

Row-level parse failures are logged at WARNING and tallied into
:class:`FootballDataLoadSummary` by reason. The skip reasons are
narrow and documented: missing/malformed date, missing Pinnacle odds,
missing result columns (FTHG/FTAG), odds failing parse or the
``SelectionOdds`` validators. Any other row-level anomaly
(``home == away``, a team name with no ASCII alphanumerics) is a
data-integrity error and raises ``ValueError`` at construction.

File-level problems also raise ``ValueError`` at construction: an empty
``csv_paths`` sequence, a filename stem not matching
``^[A-Z]{1,4}\\d?$``, a file missing required columns, a file spanning
multiple football seasons, or a ``match_id`` collision across files.
No events are partially loaded in the failing case.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterator, Sequence
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from betting_backtester._event_ordering import stream_sort_key
from betting_backtester.models import (
    Event,
    MatchResult,
    MatchSettled,
    OddsAvailable,
    OddsSnapshot,
    SelectionOdds,
)

_logger = logging.getLogger(__name__)

# Constants per docs/DESIGN.md decisions 3 and 6.
_ODDS_LEAD: timedelta = timedelta(minutes=5)
_MATCH_DURATION: timedelta = timedelta(hours=2)
_DEFAULT_KICKOFF_TIME: time = time(15, 0)

# Permissive regex covering every league code football-data.co.uk has used:
# 1-4 uppercase letters optionally followed by a single digit (E0, EC, SP1,
# D1, I1, etc.).
_LEAGUE_CODE_PATTERN = re.compile(r"^[A-Z]{1,4}\d?$")

# Team-name normalisation: runs of non-ASCII-alphanumerics collapse to one
# underscore; leading/trailing underscores are stripped. See module docstring
# for the rationale.
_TEAM_NAME_NORMALISE_PATTERN = re.compile(r"[^A-Za-z0-9]+")

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
)
# Order matters: modern Pinnacle columns are tried first; legacy is the
# fallback for files predating the rename.
_PINNACLE_COLUMN_SETS: tuple[tuple[str, str, str], ...] = (
    ("PSH", "PSD", "PSA"),
    ("PH", "PD", "PA"),
)

# Date formats seen across football-data.co.uk history. Try 4-digit years
# first so a value like "12/05/2024" cannot be misparsed as year 20.
_DATE_FORMATS: tuple[str, ...] = ("%d/%m/%Y", "%d/%m/%y")
_TIME_FORMATS: tuple[str, ...] = ("%H:%M", "%H:%M:%S")

# Explicit missing-odds sentinels in Pinnacle cells. Compared against the
# cell stripped and upper-cased. Anything else is a parse/validation
# problem, not a missing-data problem (see ``_parse_pinnacle_odds``).
_MISSING_ODDS_SENTINELS: frozenset[str] = frozenset({"", "NA"})


class _OddsParseOutcome(Enum):
    """Non-happy-path outcomes from :func:`_parse_pinnacle_odds`.

    ``MISSING`` — both column sets had at least one empty-string or
    ``NA`` cell. ``INVALID`` — a column set had values for all three
    selections, but at least one failed ``float()`` conversion or the
    ``SelectionOdds`` validators.
    """

    MISSING = "missing"
    INVALID = "invalid"


class FootballDataLoadSummary(BaseModel):
    """Per-run summary of CSV parse outcomes.

    Every data row across all files is accounted for under exactly one
    outcome: either it loaded, or it was skipped under one of four
    narrow reasons (decision 7 in ``docs/DESIGN.md``). The invariant
    ``matches_loaded + sum(skipped_*) == rows_seen`` is enforced on
    construction; there is no catch-all bucket.
    """

    model_config = ConfigDict(frozen=True)

    files_processed: int = Field(ge=0)
    rows_seen: int = Field(
        ge=0, description="Total data rows across all files (header row excluded)."
    )
    matches_loaded: int = Field(
        ge=0,
        description="Rows that produced a valid OddsAvailable + MatchSettled pair.",
    )
    skipped_missing_date: int = Field(
        ge=0,
        description="``Date`` cell empty or unparseable under known formats.",
    )
    skipped_missing_pinnacle_odds: int = Field(
        ge=0,
        description=(
            "Both the modern (PSH/PSD/PSA) and legacy (PH/PD/PA) Pinnacle "
            "column sets had at least one empty or 'NA' cell for this row."
        ),
    )
    skipped_missing_result: int = Field(
        ge=0,
        description="FTHG or FTAG missing, non-integer, or negative.",
    )
    skipped_invalid_odds: int = Field(
        ge=0,
        description=(
            "Pinnacle cells were non-missing but failed ``float()`` "
            "conversion or the ``SelectionOdds`` validators (e.g. price <= 1)."
        ),
    )

    @model_validator(mode="after")
    def _rows_seen_accounted_for(self) -> FootballDataLoadSummary:
        skipped = (
            self.skipped_missing_date
            + self.skipped_missing_pinnacle_odds
            + self.skipped_missing_result
            + self.skipped_invalid_odds
        )
        total = self.matches_loaded + skipped
        if total != self.rows_seen:
            raise ValueError(
                f"matches_loaded ({self.matches_loaded}) + skipped ({skipped}) "
                f"== {total}, expected rows_seen ({self.rows_seen}); summary "
                "is internally inconsistent"
            )
        return self


class FootballDataLoader:
    """Read football-data.co.uk CSVs and emit the canonical event stream.

    Construction is eager: every file is read and fully parsed in
    ``__init__``. :meth:`events` returns a fresh iterator over the
    cached event tuple and produces a byte-identical sequence on every
    call. See the module docstring for approximations, assumptions, and
    failure modes.
    """

    def __init__(self, csv_paths: Sequence[str | Path]) -> None:
        if len(csv_paths) == 0:
            raise ValueError("csv_paths must not be empty")
        paths: tuple[Path, ...] = tuple(Path(p) for p in csv_paths)

        # Filename league-code check — done before touching any file so a
        # bad filename fails fast without leaving partially-read state.
        for path in paths:
            if not _LEAGUE_CODE_PATTERN.match(path.stem):
                raise ValueError(
                    f"filename stem {path.stem!r} does not match league-code "
                    f"pattern '^[A-Z]{{1,4}}\\d?$' (path: {path})"
                )

        all_events: list[Event] = []
        seen_match_ids: set[str] = set()
        counts: dict[str, int] = {
            "rows_seen": 0,
            "matches_loaded": 0,
            "skipped_missing_date": 0,
            "skipped_missing_pinnacle_odds": 0,
            "skipped_missing_result": 0,
            "skipped_invalid_odds": 0,
        }

        for path in paths:
            _parse_file(path, all_events, seen_match_ids, counts)

        all_events.sort(key=stream_sort_key)
        self._events: tuple[Event, ...] = tuple(all_events)
        self._summary: FootballDataLoadSummary = FootballDataLoadSummary(
            files_processed=len(paths),
            **counts,
        )

    @property
    def load_summary(self) -> FootballDataLoadSummary:
        """Parse-outcome counts, frozen at construction."""
        return self._summary

    def events(self) -> Iterator[Event]:
        """Return a fresh iterator over the canonical event stream.

        Re-callable: every call yields the same sequence from the tuple
        cached at construction.
        """
        return iter(self._events)


def _parse_file(
    path: Path,
    all_events: list[Event],
    seen_match_ids: set[str],
    counts: dict[str, int],
) -> None:
    """Parse one CSV, appending events to ``all_events`` and mutating counts."""
    league_code = path.stem

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, restval="")
        original_fieldnames = reader.fieldnames
        if original_fieldnames is None:
            raise ValueError(f"file {path} has no header row")
        # Strip whitespace from header names in-place so "Date " matches
        # "Date" for the required-columns check and for per-row lookups.
        stripped_fieldnames: list[str] = [name.strip() for name in original_fieldnames]
        reader.fieldnames = stripped_fieldnames

        missing_required = _REQUIRED_COLUMNS - set(stripped_fieldnames)
        if missing_required:
            raise ValueError(
                f"file {path} missing required columns: {sorted(missing_required)}"
            )

        file_events: list[Event] = []
        file_seasons: set[str] = set()

        for row_index, raw_row in enumerate(reader):
            counts["rows_seen"] += 1
            row: dict[str, str] = {
                str(k): ("" if v is None else str(v)) for k, v in raw_row.items()
            }
            _parse_row(
                path=path,
                league_code=league_code,
                row_index=row_index,
                row=row,
                file_events=file_events,
                file_seasons=file_seasons,
                seen_match_ids=seen_match_ids,
                counts=counts,
            )

    if len(file_seasons) > 1:
        raise ValueError(
            f"file {path} spans multiple football seasons: "
            f"{sorted(file_seasons)} (a single file must belong to one season)"
        )

    all_events.extend(file_events)


def _parse_row(
    *,
    path: Path,
    league_code: str,
    row_index: int,
    row: dict[str, str],
    file_events: list[Event],
    file_seasons: set[str],
    seen_match_ids: set[str],
    counts: dict[str, int],
) -> None:
    """Parse one data row, emitting two events or bumping a skip counter."""
    parsed_date = _parse_date(row.get("Date", ""))
    if parsed_date is None:
        counts["skipped_missing_date"] += 1
        _logger.warning(
            "%s row %d: skipping (missing or malformed Date)", path, row_index
        )
        return

    home_goals_raw = row.get("FTHG", "").strip()
    away_goals_raw = row.get("FTAG", "").strip()
    try:
        home_goals = int(home_goals_raw)
        away_goals = int(away_goals_raw)
    except ValueError:
        counts["skipped_missing_result"] += 1
        _logger.warning(
            "%s row %d: skipping (FTHG/FTAG missing or not integer)",
            path,
            row_index,
        )
        return
    if home_goals < 0 or away_goals < 0:
        counts["skipped_missing_result"] += 1
        _logger.warning(
            "%s row %d: skipping (FTHG/FTAG negative: %d, %d)",
            path,
            row_index,
            home_goals,
            away_goals,
        )
        return

    odds_result = _parse_pinnacle_odds(row)
    if odds_result is _OddsParseOutcome.MISSING:
        counts["skipped_missing_pinnacle_odds"] += 1
        _logger.warning(
            "%s row %d: skipping (Pinnacle odds missing or 'NA' in both "
            "modern and legacy columns)",
            path,
            row_index,
        )
        return
    if odds_result is _OddsParseOutcome.INVALID:
        counts["skipped_invalid_odds"] += 1
        _logger.warning(
            "%s row %d: skipping (Pinnacle odds failed parse or "
            "SelectionOdds validators)",
            path,
            row_index,
        )
        return

    parsed_time = _parse_time(row.get("Time", "")) or _DEFAULT_KICKOFF_TIME
    kickoff = datetime.combine(parsed_date, parsed_time, tzinfo=timezone.utc)

    home = row.get("HomeTeam", "").strip()
    away = row.get("AwayTeam", "").strip()
    home_norm = _normalise_team_name(home)
    away_norm = _normalise_team_name(away)
    if not home_norm or not away_norm:
        raise ValueError(
            f"{path} row {row_index}: team name normalised to empty "
            f"(home={home!r}, away={away!r}); cannot form match_id"
        )
    if home_norm == away_norm:
        raise ValueError(
            f"{path} row {row_index}: home and away teams normalise to the "
            f"same value (home={home!r}, away={away!r})"
        )

    match_id = f"{league_code}-{parsed_date.isoformat()}-{home_norm}-{away_norm}"
    if match_id in seen_match_ids:
        raise ValueError(
            f"duplicate match_id {match_id!r} across inputs; second "
            f"occurrence in {path} row {row_index}"
        )

    season = _derive_season_for_date(parsed_date)
    file_seasons.add(season)

    home_odds, draw_odds, away_odds = odds_result
    snapshot = OddsSnapshot(
        match_id=match_id,
        timestamp=kickoff - _ODDS_LEAD,
        home=home_odds,
        draw=draw_odds,
        away=away_odds,
    )
    result = MatchResult(
        match_id=match_id,
        timestamp=kickoff + _MATCH_DURATION,
        home_goals=home_goals,
        away_goals=away_goals,
    )

    seen_match_ids.add(match_id)
    file_events.append(OddsAvailable(snapshot=snapshot))
    file_events.append(MatchSettled(result=result))
    counts["matches_loaded"] += 1


def _parse_pinnacle_odds(
    row: dict[str, str],
) -> tuple[SelectionOdds, SelectionOdds, SelectionOdds] | _OddsParseOutcome:
    """Extract Pinnacle back/lay odds from one row.

    Priority: modern ``PSH``/``PSD``/``PSA`` > legacy ``PH``/``PD``/``PA``.
    A column set is "present" only when all three cells exist and none
    is an empty-string or ``NA`` sentinel; otherwise the next set is
    tried. If neither set is fully present, returns
    :attr:`_OddsParseOutcome.MISSING`.

    If a set is present but any cell fails ``float()`` or
    ``SelectionOdds`` rejects the resulting prices, returns
    :attr:`_OddsParseOutcome.INVALID` without cascading to the fallback
    set. The fallback exists for column-set *availability* (older files
    that lack modern columns entirely), not for salvaging rows where a
    modern cell is garbage.
    """
    for cols in _PINNACLE_COLUMN_SETS:
        values = [row.get(c, "") for c in cols]
        if any(v.strip().upper() in _MISSING_ODDS_SENTINELS for v in values):
            continue
        try:
            prices = [float(v) for v in values]
            home_odds = SelectionOdds(back_price=prices[0], lay_price=prices[0])
            draw_odds = SelectionOdds(back_price=prices[1], lay_price=prices[1])
            away_odds = SelectionOdds(back_price=prices[2], lay_price=prices[2])
        except (ValueError, ValidationError):
            return _OddsParseOutcome.INVALID
        return (home_odds, draw_odds, away_odds)
    return _OddsParseOutcome.MISSING


def _parse_date(s: str) -> date | None:
    """Parse a football-data.co.uk date cell; return None on missing or malformed."""
    s = s.strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_time(s: str) -> time | None:
    """Parse a kickoff time cell; return None on missing or malformed."""
    s = s.strip()
    if not s:
        return None
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    return None


def _normalise_team_name(name: str) -> str:
    """Collapse non-ASCII-alphanumeric runs to '_', strip leading/trailing '_'."""
    return _TEAM_NAME_NORMALISE_PATTERN.sub("_", name).strip("_")


def _derive_season_for_date(d: date) -> str:
    """Season label under the European July-to-May convention.

    A date in month ``>= 7`` opens a new season; earlier months belong
    to the season that started in the previous calendar year. Format is
    ``YYYY-YY`` using the start year and the last two digits of the
    end year. See the module docstring for the non-European caveat.
    """
    start_year = d.year if d.month >= 7 else d.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


__all__ = ["FootballDataLoadSummary", "FootballDataLoader"]
