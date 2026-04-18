"""Canonical data models and event types for the backtester.

The only types that flow through the event stream. Source-specific quirks are
resolved by loaders before instances are constructed; downstream code trusts
these models.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class Selection(StrEnum):
    """The three outcomes on a 1X2 market."""

    HOME = "home"
    DRAW = "draw"
    AWAY = "away"


class SelectionOdds(BaseModel):
    """Best back and lay decimal odds for one selection at one instant."""

    model_config = ConfigDict(frozen=True)

    back_price: float = Field(
        ge=1.01,
        description="Decimal odds paid out if you back this selection and it wins.",
    )
    lay_price: float = Field(
        ge=1.01,
        description="Decimal odds you must pay out at if you lay this selection and it wins.",
    )

    @model_validator(mode="after")
    def _back_not_above_lay(self) -> SelectionOdds:
        if self.back_price > self.lay_price:
            raise ValueError(
                f"back_price ({self.back_price}) must be <= lay_price ({self.lay_price}); "
                "an exchange book where back > lay is already-matched arbitrage."
            )
        return self


class Match(BaseModel):
    """Fixture identity. Produced once per fixture by the loader."""

    model_config = ConfigDict(frozen=True)

    match_id: str = Field(
        min_length=1,
        description="Opaque unique fixture id. Loaders are responsible for stability across files.",
    )
    kickoff: datetime = Field(
        description="Scheduled kickoff, timezone-aware, UTC.",
    )
    league: str = Field(
        min_length=1,
        description="League code, e.g. 'E0' for football-data.co.uk Premier League.",
    )
    season: str = Field(
        min_length=1,
        description="Season label, e.g. '2023-24'. Format set by loader, but must be consistent within a run.",
    )
    home: str = Field(
        min_length=1,
        description="Home team name as emitted by the loader (already normalised across sources).",
    )
    away: str = Field(
        min_length=1,
        description="Away team name as emitted by the loader (already normalised across sources).",
    )

    @model_validator(mode="after")
    def _validate(self) -> Match:
        if self.kickoff.utcoffset() != timedelta(0):
            raise ValueError("kickoff must be UTC (offset 0)")
        if self.home == self.away:
            raise ValueError(f"home and away must differ (got {self.home!r} for both)")
        return self


class OddsSnapshot(BaseModel):
    """Market state for one fixture at one instant."""

    model_config = ConfigDict(frozen=True)

    match_id: str = Field(
        min_length=1,
        description="Links the snapshot back to its Match.",
    )
    timestamp: datetime = Field(
        description="Observation time of this snapshot, timezone-aware UTC. Sort key in the event stream.",
    )
    home: SelectionOdds = Field(
        description="Best back/lay prices for the home-win selection."
    )
    draw: SelectionOdds = Field(
        description="Best back/lay prices for the draw selection."
    )
    away: SelectionOdds = Field(
        description="Best back/lay prices for the away-win selection."
    )

    @model_validator(mode="after")
    def _validate(self) -> OddsSnapshot:
        if self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be UTC (offset 0)")
        return self

    def odds_for(self, selection: Selection) -> SelectionOdds:
        """Return the SelectionOdds for a given selection."""
        return {
            Selection.HOME: self.home,
            Selection.DRAW: self.draw,
            Selection.AWAY: self.away,
        }[selection]


class MatchResult(BaseModel):
    """Final state of a fixture. Emitted once, at settlement time."""

    model_config = ConfigDict(frozen=True)

    match_id: str = Field(
        min_length=1,
        description="Links the result back to its Match.",
    )
    timestamp: datetime = Field(
        description="Settlement time (match end), timezone-aware UTC. Sort key in the event stream.",
    )
    home_goals: int = Field(ge=0, description="Final home score.")
    away_goals: int = Field(ge=0, description="Final away score.")

    @model_validator(mode="after")
    def _validate(self) -> MatchResult:
        if self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be UTC (offset 0)")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def outcome(self) -> Selection:
        """The selection that would pay out a winning back bet. Derived from the score."""
        if self.home_goals > self.away_goals:
            return Selection.HOME
        if self.home_goals < self.away_goals:
            return Selection.AWAY
        return Selection.DRAW


class OddsAvailable(BaseModel):
    """Event: a new odds snapshot for a fixture has become observable."""

    model_config = ConfigDict(frozen=True)

    snapshot: OddsSnapshot = Field(
        description="The market state that just became visible."
    )

    @property
    def timestamp(self) -> datetime:
        """Stream sort key. Equal to the snapshot's timestamp."""
        return self.snapshot.timestamp


class MatchSettled(BaseModel):
    """Event: a fixture has finalised. Triggers bet resolution in the backtester."""

    model_config = ConfigDict(frozen=True)

    result: MatchResult = Field(
        description="The final result that triggered settlement."
    )

    @property
    def timestamp(self) -> datetime:
        """Stream sort key. Equal to the result's timestamp."""
        return self.result.timestamp


Event = OddsAvailable | MatchSettled
"""Union of every event that flows through the backtester's stream.

Loaders emit these in strict non-decreasing ``timestamp`` order; the backtester
dispatches on concrete type (``OddsAvailable`` -> ``on_odds``, ``MatchSettled``
-> ``on_settled``). Any new event type added later must be appended to this
union so exhaustiveness checks light up unhandled cases.
"""
