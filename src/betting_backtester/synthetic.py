"""Synthetic event source for backtester correctness verification.

This module produces a time-ordered stream of ``OddsAvailable`` and
``MatchSettled`` events sampled from a configured true probability
distribution. Its job is to be the **ground-truth rig** for backtester
correctness, not a realism simulator.

Two deliberate v1 choices worth calling out:

* **Fair odds only.** Every emitted ``OddsSnapshot`` has
  ``back_price == lay_price == 1 / p`` for each selection, where ``p`` is the
  configured true probability for that outcome. There is no overround, no
  spread, and no random mispricing. The edge in the backtester correctness
  test comes from the *strategy* being given an ``assumed_probs`` vector
  that differs from the generator's ``true_probs``. Overround and
  commission-drag scenarios are deferred to a future ``OddsPolicy``
  extension rather than baked into this module.
* **Placeholder scores.** ``MatchResult`` requires integer goal counts, but
  score realism is not a goal here. Each sampled outcome maps to a fixed
  minimal score consistent with ``MatchResult.outcome``:
  ``HOME -> (1, 0)``, ``AWAY -> (0, 1)``, ``DRAW -> (1, 1)``. This keeps
  the per-match RNG budget at exactly one draw (the outcome) and keeps
  determinism accounting trivial. A Poisson-based score model belongs with
  the xG strategy, not here.

Determinism is absolute: identical :class:`SyntheticGeneratorConfig` yields
a byte-identical event sequence, and no global RNG state is touched or
relied on.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, model_validator

from betting_backtester.models import (
    Event,
    Match,
    MatchResult,
    MatchSettled,
    OddsAvailable,
    OddsSnapshot,
    Selection,
    SelectionOdds,
)

# Tolerance on the (home + draw + away) == 1 check. Tight enough to catch
# genuine configuration mistakes while tolerating the float error of summing
# three user-supplied values.
_PROB_SUM_EPSILON = 1e-9

_PLACEHOLDER_SCORES: dict[Selection, tuple[int, int]] = {
    Selection.HOME: (1, 0),
    Selection.AWAY: (0, 1),
    Selection.DRAW: (1, 1),
}

# Iteration order of the cumulative-threshold sampler. Pinned explicitly so
# the RNG-draw-to-outcome mapping does not silently change if the Selection
# enum is ever reordered.
_SELECTION_ORDER: tuple[Selection, ...] = (
    Selection.HOME,
    Selection.DRAW,
    Selection.AWAY,
)


class TrueProbabilities(BaseModel):
    """The true outcome distribution applied to every generated match.

    All three probabilities must be strictly positive and sum to 1 within
    ``_PROB_SUM_EPSILON``. The same vector is applied to every generated
    match in v1; per-match distributions are an intentional non-goal here.
    """

    model_config = ConfigDict(frozen=True)

    home: float = Field(gt=0.0, lt=1.0, description="True probability of home win.")
    draw: float = Field(gt=0.0, lt=1.0, description="True probability of a draw.")
    away: float = Field(gt=0.0, lt=1.0, description="True probability of away win.")

    @model_validator(mode="after")
    def _sum_to_one(self) -> TrueProbabilities:
        total = self.home + self.draw + self.away
        if abs(total - 1.0) > _PROB_SUM_EPSILON:
            raise ValueError(
                f"probabilities must sum to 1 within {_PROB_SUM_EPSILON}, "
                f"got {total!r} (home={self.home}, draw={self.draw}, away={self.away})"
            )
        return self

    def for_selection(self, selection: Selection) -> float:
        """Return the probability for one selection."""
        return {
            Selection.HOME: self.home,
            Selection.DRAW: self.draw,
            Selection.AWAY: self.away,
        }[selection]


@dataclass(frozen=True)
class SyntheticGeneratorConfig:
    """Configuration for a :class:`SyntheticGenerator` run.

    Attributes
    ----------
    n_matches:
        Number of fixtures to generate. Strictly positive.
    true_probs:
        Outcome distribution. Every match samples from the same vector.
    seed:
        Integer seed. A fresh ``random.Random(seed)`` is constructed on
        every call to :meth:`SyntheticGenerator.events`; no global RNG state
        is touched.
    start:
        Kickoff of the first fixture. Must be timezone-aware UTC.
    league, season:
        Labels plumbed into every generated :class:`Match`.
    fixture_spacing:
        Kickoff-to-kickoff delta. Consecutive fixtures are spaced by this
        value, producing a strictly increasing kickoff sequence.
    match_duration:
        Kickoff-to-settlement delta. ``MatchSettled`` is emitted at
        ``kickoff + match_duration``.
    odds_lead:
        Settlement-lead for the pre-match odds snapshot. ``OddsAvailable``
        is emitted at ``kickoff - odds_lead``. Exposed rather than magic so
        tests can pin exact timestamps.
    """

    n_matches: int
    true_probs: TrueProbabilities
    seed: int
    start: datetime
    league: str = "SYN"
    season: str = "2024-25"
    fixture_spacing: timedelta = timedelta(days=1)
    match_duration: timedelta = timedelta(hours=2)
    odds_lead: timedelta = timedelta(minutes=5)

    def __post_init__(self) -> None:
        if self.n_matches <= 0:
            raise ValueError(f"n_matches must be positive, got {self.n_matches}")
        if self.start.utcoffset() != timedelta(0):
            raise ValueError("start must be UTC (offset 0)")
        if self.fixture_spacing <= timedelta(0):
            raise ValueError(
                f"fixture_spacing must be positive, got {self.fixture_spacing}"
            )
        if self.match_duration <= timedelta(0):
            raise ValueError(
                f"match_duration must be positive, got {self.match_duration}"
            )
        if self.odds_lead <= timedelta(0):
            raise ValueError(f"odds_lead must be positive, got {self.odds_lead}")
        if not self.league:
            raise ValueError("league must be non-empty")
        if not self.season:
            raise ValueError("season must be non-empty")


class SyntheticGenerator:
    """Generate a deterministic, time-ordered stream of events.

    Implements the :class:`~betting_backtester.event_source.EventSource`
    protocol. One ``OddsAvailable`` and one ``MatchSettled`` are emitted
    per generated match, in strict non-decreasing timestamp order with
    settlement-before-odds tie-breaking at equal timestamps.

    See the module docstring for the fair-odds and placeholder-score
    rationale.
    """

    def __init__(self, config: SyntheticGeneratorConfig) -> None:
        self._config: SyntheticGeneratorConfig = config
        self._matches: tuple[Match, ...] = tuple(self._build_matches())
        self._snapshot_odds: dict[str, SelectionOdds] = self._fair_odds_fields()

    @property
    def matches(self) -> tuple[Match, ...]:
        """Fixtures this generator will emit events for. Stable across calls."""
        return self._matches

    def events(self) -> Iterator[Event]:
        """Yield the canonical event stream.

        Re-callable: each call constructs a fresh ``random.Random(seed)``
        and yields an equivalent, independent event sequence.
        """
        return self._stream()

    def _build_matches(self) -> Iterator[Match]:
        cfg = self._config
        for i in range(cfg.n_matches):
            kickoff = cfg.start + i * cfg.fixture_spacing
            yield Match(
                match_id=f"{cfg.league}-{cfg.season}-{i:04d}",
                kickoff=kickoff,
                league=cfg.league,
                season=cfg.season,
                home=f"SYN-T{2 * i + 1:04d}",
                away=f"SYN-T{2 * i + 2:04d}",
            )

    def _stream(self) -> Iterator[Event]:
        cfg = self._config
        rng = random.Random(cfg.seed)
        buffered: list[Event] = []
        for match in self._matches:
            outcome = self._sample_outcome(rng)
            home_goals, away_goals = _PLACEHOLDER_SCORES[outcome]
            buffered.append(
                OddsAvailable(
                    snapshot=OddsSnapshot(
                        match_id=match.match_id,
                        timestamp=match.kickoff - cfg.odds_lead,
                        **self._snapshot_odds,
                    )
                )
            )
            buffered.append(
                MatchSettled(
                    result=MatchResult(
                        match_id=match.match_id,
                        timestamp=match.kickoff + cfg.match_duration,
                        home_goals=home_goals,
                        away_goals=away_goals,
                    )
                )
            )
        buffered.sort(key=_stream_sort_key)
        yield from buffered

    def _sample_outcome(self, rng: random.Random) -> Selection:
        # Exactly one RNG call per match — keeps the per-match RNG budget
        # trivially auditable.
        u = rng.random()
        cumulative = 0.0
        probs = self._config.true_probs
        for selection in _SELECTION_ORDER:
            cumulative += probs.for_selection(selection)
            if u < cumulative:
                return selection
        # If u rounds just above the final cumulative by float slack, fall
        # through to the last selection rather than raise. Probs are already
        # validated to sum to 1 within _PROB_SUM_EPSILON.
        return _SELECTION_ORDER[-1]

    def _fair_odds_fields(self) -> dict[str, SelectionOdds]:
        probs = self._config.true_probs
        return {
            "home": _fair_odds(probs.home),
            "draw": _fair_odds(probs.draw),
            "away": _fair_odds(probs.away),
        }


def _fair_odds(p: float) -> SelectionOdds:
    price = 1.0 / p
    return SelectionOdds(back_price=price, lay_price=price)


def _stream_sort_key(event: Event) -> tuple[datetime, int, str]:
    # Settlement (0) sorts before new-odds (1) at equal timestamps, then by
    # match_id. Matches the invariant in docs/DESIGN.md.
    if isinstance(event, MatchSettled):
        return (event.timestamp, 0, event.result.match_id)
    return (event.timestamp, 1, event.snapshot.match_id)
