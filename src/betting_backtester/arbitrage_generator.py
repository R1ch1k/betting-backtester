"""Arbitrage-ready event source for module 10 of the backtester.

This module produces a time-ordered stream of ``OddsAvailable`` and
``MatchSettled`` events with realistic exchange-shaped prices --
``back_price <= lay_price`` per selection -- and a configurable
injection of back-side arbitrage opportunities. Its job is to feed the
``ArbitrageDetector`` strategy a stream where sum(1/back_i) < 1 occurs
on known matches, so the detector and the per-market commission
aggregation from module 3 can be exercised end-to-end.

Why a separate generator from :class:`~betting_backtester.synthetic.SyntheticGenerator`.
``SyntheticGenerator`` emits fair, zero-spread odds
(``back_price == lay_price == 1 / p``); by construction every snapshot
is a break-even book and no arbitrage is possible. Football-data.co.uk
closing lines are similarly fake for arb purposes because the loader
sets ``back_price == lay_price == bookmaker_price`` (see the
:class:`~betting_backtester.football_data.FootballDataLoader` docstring).
To exercise the arbitrage strategy honestly we need a data source that
places back and lay on either side of a chosen reference with a
controllable book-percentage gap; that is this module.

Price construction
------------------

Given ``true_probs`` with ``fair_i = 1 / true_probs_i`` and a
``half_spread`` ``h >= 0``:

* **Non-arb match.** ``back_i = fair_i / (1 + h)``,
  ``lay_i = fair_i * (1 + h)``. The back book sums to ``1 + h`` (a
  small overround, no arbitrage).
* **Arb match.** ``back_i = fair_i / (1 - arb_margin)``. The back book
  then sums to exactly ``1 - arb_margin`` within
  ``_ARB_BOOK_TOLERANCE``.

Across both cases the **uniform lay/back invariant**
``lay_i = back_i * (1 + h) ** 2`` is preserved, so the multiplicative
spread between back and lay is identical on arb and non-arb rows. In
particular, on a non-arb match the formula reduces to the symmetric
``lay_i = fair_i * (1 + h)``; on an arb match the same factor is
applied on top of the shifted back reference, which keeps
``back_i <= lay_i`` (and hence :class:`~betting_backtester.models.SelectionOdds`'s
invariant) for every selection.

``arb_margin`` follows the book-percentage convention used in the
exchange-arbitrage literature: it is the gap ``1 - sum(1/back_i)`` on
the displayed back book, **not** the realised return per unit
commitment. The realised return under equal-profit staking is
``arb_margin / (1 - arb_margin)`` per unit committed; see
:class:`~betting_backtester.strategies.arbitrage_detector.ArbitrageDetector`
for the stake-allocation derivation.

Arb injection via :class:`ArbSchedule`
--------------------------------------

Per-match arb injection is delegated to an :class:`ArbSchedule`
object. Two implementations ship:

* :class:`FixedArbSchedule` -- an exact set of match indices marked as
  arb. RNG-inert (``has_arb`` does not call ``rng``), so under a
  ``FixedArbSchedule`` the RNG draw sequence inside
  :meth:`ArbitrageGenerator._stream` is identical to
  :class:`~betting_backtester.synthetic.SyntheticGenerator`'s for the
  same seed and ``true_probs``. Outcomes are therefore byte-identical
  across the two generators -- a property used by
  ``test_arbitrage_generator.py`` as a smoke test for the
  schedule/RNG contract.
* :class:`BernoulliArbSchedule` -- per-match Bernoulli draw at
  ``rate``. Consumes exactly one ``rng.random()`` draw per match,
  called **before** the outcome sample. Swapping schedule
  implementations therefore changes the RNG draw sequence, and the
  outcome stream along with it; that is intentional (schedule is part
  of the config).

Determinism
-----------

Identical :class:`ArbitrageGeneratorConfig` yields a byte-identical
event sequence. No global RNG state is touched or relied on: the
generator constructs a fresh ``random.Random(seed)`` on every call to
:meth:`ArbitrageGenerator.events`, and the per-match draw order
inside :meth:`ArbitrageGenerator._stream` is pinned (``has_arb``
before outcome sampling).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from betting_backtester._event_ordering import stream_sort_key
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
from betting_backtester.synthetic import TrueProbabilities

# Tolerance on the ``sum(1/back_i) == 1 - arb_margin`` invariant. Bounded
# by ``TrueProbabilities``' own ``sum(p) == 1`` epsilon (1e-9) scaled by
# ``(1 - arb_margin) < 1``, so 1e-9 is safe for assertions in tests and
# internal sanity checks alike.
_ARB_BOOK_TOLERANCE = 1e-9

# Upper bound on ``arb_margin``. A 10% back-book gap is already an
# unrealistic market anomaly; anything above this is almost certainly
# a configuration mistake.
_ARB_MARGIN_MAX = 0.1

# Upper bound on ``half_spread``. A 10% one-sided spread implies a
# ~20% round-trip cost -- far beyond any real exchange book; a
# configuration mistake rather than a realistic scenario.
_HALF_SPREAD_MAX = 0.1

_PLACEHOLDER_SCORES: dict[Selection, tuple[int, int]] = {
    Selection.HOME: (1, 0),
    Selection.AWAY: (0, 1),
    Selection.DRAW: (1, 1),
}

# Iteration order of the cumulative-threshold outcome sampler. Pinned
# explicitly so the RNG-draw-to-outcome mapping matches
# :class:`~betting_backtester.synthetic.SyntheticGenerator`'s mapping
# byte-for-byte (relied on by the cross-generator determinism test).
_SELECTION_ORDER: tuple[Selection, ...] = (
    Selection.HOME,
    Selection.DRAW,
    Selection.AWAY,
)


@runtime_checkable
class ArbSchedule(Protocol):
    """Decides, per generated match, whether an arbitrage should be injected.

    Implementations must be pure functions of ``match_index`` and the
    RNG state they consume. The generator calls
    :meth:`has_arb` exactly once per match, **before** sampling that
    match's outcome; schedules that consume RNG draws therefore shift
    the outcome stream relative to a zero-draw schedule at the same
    seed.
    """

    def has_arb(self, match_index: int, rng: random.Random) -> bool:
        """Return ``True`` iff match ``match_index`` should be arb-priced.

        ``match_index`` is 0-based within the generator's fixture list.
        ``rng`` is the generator's per-run RNG; implementations may
        consume zero or more ``rng.random()`` draws but must do so
        deterministically for determinism of the event stream to hold.
        """
        ...


class FixedArbSchedule:
    """Arbs at exact, caller-supplied match indices. RNG-inert.

    Used by tests (and notebooks) that need to assert "exactly these
    matches are arbs" without depending on the RNG state. Because
    :meth:`has_arb` does not call ``rng``, the outcome-sampling draw
    sequence inside :meth:`ArbitrageGenerator._stream` matches
    :class:`~betting_backtester.synthetic.SyntheticGenerator`'s draw
    sequence byte-for-byte under the same seed and ``true_probs``.

    Out-of-range positions (``>= n_matches``) are rejected at
    :class:`ArbitrageGenerator` construction time, not here: this
    class is ``n_matches``-agnostic by design, so the check lives
    where the bound is known.
    """

    __slots__ = ("_arb_positions",)

    def __init__(self, arb_positions: Iterable[int]) -> None:
        """Validate positions and store as a frozenset.

        Parameters
        ----------
        arb_positions:
            Any iterable of non-negative integer match indices.
            Coerced to :class:`frozenset` internally; duplicates in
            the input are therefore collapsed without comment.

        Raises
        ------
        TypeError
            Any element is not an ``int`` (``bool`` is explicitly
            rejected even though it is a subclass of ``int`` in
            Python -- accidental ``True``/``False`` in a set of match
            indices is a bug, not a feature).
        ValueError
            Any element is negative.
        """
        positions: frozenset[int] = frozenset(arb_positions)
        for pos in positions:
            # bool is a subclass of int; exclude it explicitly so
            # ``FixedArbSchedule({True})`` does not silently mean "arb
            # at position 1".
            if isinstance(pos, bool) or not isinstance(pos, int):
                raise TypeError(
                    f"arb_positions entries must be int, got "
                    f"{type(pos).__name__} ({pos!r})"
                )
            if pos < 0:
                raise ValueError(
                    f"arb_positions entries must be non-negative, got {pos}"
                )
        self._arb_positions: frozenset[int] = positions

    @property
    def arb_positions(self) -> frozenset[int]:
        """The set of match indices this schedule flags as arbs."""
        return self._arb_positions

    def has_arb(self, match_index: int, rng: random.Random) -> bool:
        """Return whether ``match_index`` is flagged. Does not touch ``rng``."""
        del rng  # explicitly unused
        return match_index in self._arb_positions


class BernoulliArbSchedule:
    """Per-match Bernoulli arb injection. Consumes one RNG draw per match.

    Used for realistic streams where the arb density approximates a
    steady rate over a long horizon. ``has_arb`` calls
    ``rng.random()`` exactly once per invocation regardless of
    ``match_index``, so the RNG budget is trivially auditable.
    """

    __slots__ = ("_rate",)

    def __init__(self, rate: float) -> None:
        """Validate the rate.

        Parameters
        ----------
        rate:
            Probability that any given match is an arb. Must be a
            finite float in ``[0, 1]``. Endpoints are permitted:
            ``rate = 0`` means "never arb" (still consumes one draw
            per match, so the outcome stream differs from
            :class:`FixedArbSchedule` at the same seed);
            ``rate = 1`` means "always arb".

        Raises
        ------
        ValueError
            ``rate`` is non-finite or outside ``[0, 1]``.
        """
        if not math.isfinite(rate):
            raise ValueError(f"rate must be finite, got {rate!r}")
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self._rate: float = float(rate)

    @property
    def rate(self) -> float:
        """The per-match arb probability."""
        return self._rate

    def has_arb(self, match_index: int, rng: random.Random) -> bool:
        """Return ``rng.random() < rate``. ``match_index`` is ignored."""
        del match_index  # explicitly unused
        return rng.random() < self._rate


@dataclass(frozen=True)
class ArbitrageGeneratorConfig:
    """Configuration for an :class:`ArbitrageGenerator` run.

    Mirrors :class:`~betting_backtester.synthetic.SyntheticGeneratorConfig`'s
    shape for consistency, with three arbitrage-specific additions:
    ``schedule``, ``arb_margin``, ``half_spread``.

    Attributes
    ----------
    n_matches:
        Number of fixtures to generate. Strictly positive.
    true_probs:
        Outcome distribution. Every match samples from the same vector.
    seed:
        Integer seed. A fresh ``random.Random(seed)`` is constructed on
        every call to :meth:`ArbitrageGenerator.events`.
    start:
        Kickoff of the first fixture. Must be timezone-aware UTC.
    schedule:
        Per-match arb injection rule; see :class:`ArbSchedule`. Must
        implement the ``has_arb(match_index, rng) -> bool`` method.
    arb_margin:
        Book-percentage gap on arb matches: ``sum(1 / back_i)`` on an
        arb match equals ``1 - arb_margin`` within
        :data:`_ARB_BOOK_TOLERANCE`. Must be in ``(0, 0.1)``. The
        upper bound is a sanity cap; realistic arbs rarely exceed
        a few percent.
    half_spread:
        One-sided multiplicative spread between the reference price
        and each side of the book. ``0`` collapses back and lay
        prices to the reference. Must be in ``[0, 0.1)``.
    league, season:
        Labels plumbed into every generated :class:`Match`. Defaults
        differ from :class:`SyntheticGeneratorConfig` (``"SYN-ARB"``
        vs ``"SYN"``) so a mixed DataFrame remains self-describing.
    fixture_spacing, match_duration, odds_lead:
        Same semantics as in :class:`SyntheticGeneratorConfig`.
    """

    n_matches: int
    true_probs: TrueProbabilities
    seed: int
    start: datetime
    schedule: ArbSchedule
    arb_margin: float = 0.02
    half_spread: float = 0.01
    league: str = "SYN-ARB"
    season: str = "2024-25"
    fixture_spacing: timedelta = timedelta(days=1)
    match_duration: timedelta = timedelta(hours=2)
    odds_lead: timedelta = timedelta(minutes=5)

    def __post_init__(self) -> None:
        if self.n_matches <= 0:
            raise ValueError(f"n_matches must be positive, got {self.n_matches}")
        if self.start.utcoffset() != timedelta(0):
            raise ValueError("start must be UTC (offset 0)")
        if not math.isfinite(self.arb_margin):
            raise ValueError(f"arb_margin must be finite, got {self.arb_margin!r}")
        if not 0.0 < self.arb_margin < _ARB_MARGIN_MAX:
            raise ValueError(
                f"arb_margin must be in (0, {_ARB_MARGIN_MAX}), got {self.arb_margin}"
            )
        if not math.isfinite(self.half_spread):
            raise ValueError(f"half_spread must be finite, got {self.half_spread!r}")
        if not 0.0 <= self.half_spread < _HALF_SPREAD_MAX:
            raise ValueError(
                f"half_spread must be in [0, {_HALF_SPREAD_MAX}), "
                f"got {self.half_spread}"
            )
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
        if not isinstance(self.schedule, ArbSchedule):
            raise TypeError(
                f"schedule must implement the ArbSchedule protocol "
                f"(has_arb(match_index, rng) -> bool), got "
                f"{type(self.schedule).__name__}"
            )

        # Sanity: keep every non-arb back_price strictly above 1.0.
        # Only the non-arb leg can violate this bound: the arb leg
        # ``back_i = fair_i / (1 - arb_margin) = 1 / (p * (1 - arb_margin))``
        # is strictly above ``fair_i`` and therefore above 1 for any
        # valid probability (``p < 1`` and ``arb_margin < 1``). The
        # non-arb leg ``back_i = 1 / (p * (1 + half_spread))`` dips to
        # 1 or below when ``p >= 1 / (1 + half_spread)`` and must be
        # rejected. Fail at config construction rather than deep
        # inside the event stream where ``SelectionOdds`` would raise.
        probs = self.true_probs
        p_max = max(probs.home, probs.draw, probs.away)
        p_bound = 1.0 / (1.0 + self.half_spread)
        if p_max >= p_bound:
            raise ValueError(
                f"max(true_probs) = {p_max} must be < 1 / (1 + half_spread) "
                f"= {p_bound} to keep every non-arb back_price strictly "
                f"above 1.0; tighten half_spread or true_probs. (The "
                f"arb-leg back_price = 1 / (p * (1 - arb_margin)) is "
                f"always > 1 for p <= 1 and arb_margin < 1, so it never "
                f"contributes to this bound.)"
            )


class ArbitrageGenerator:
    """Generate a deterministic, time-ordered event stream with injected arbs.

    Implements the :class:`~betting_backtester.event_source.EventSource`
    protocol. One ``OddsAvailable`` and one ``MatchSettled`` are
    emitted per generated match, in strict non-decreasing timestamp
    order with settlement-before-odds tie-breaking at equal
    timestamps. Matches flagged by the schedule carry arb-priced back
    odds; all others carry the overround non-arb prices.
    """

    def __init__(self, config: ArbitrageGeneratorConfig) -> None:
        """Validate schedule/config compatibility and pre-compute odds fields.

        Raises
        ------
        ValueError
            ``config.schedule`` is a :class:`FixedArbSchedule` whose
            positions are not all in ``[0, config.n_matches)``. Silent
            misconfiguration is worse than a loud error in a research
            framework, so out-of-range positions fail at construction
            rather than being ignored.
        """
        self._config: ArbitrageGeneratorConfig = config

        schedule = config.schedule
        if isinstance(schedule, FixedArbSchedule):
            out_of_range = sorted(
                p for p in schedule.arb_positions if p >= config.n_matches
            )
            if out_of_range:
                raise ValueError(
                    f"FixedArbSchedule positions must all be in "
                    f"[0, n_matches={config.n_matches}); out-of-range "
                    f"positions: {out_of_range}"
                )

        self._matches: tuple[Match, ...] = tuple(self._build_matches())
        self._matches_by_id: Mapping[str, Match] = MappingProxyType(
            {m.match_id: m for m in self._matches}
        )
        self._non_arb_odds: dict[str, SelectionOdds] = self._build_odds_fields(
            arb=False
        )
        self._arb_odds: dict[str, SelectionOdds] = self._build_odds_fields(arb=True)

    @property
    def matches(self) -> Mapping[str, Match]:
        """Read-only ``match_id -> Match`` directory for every generated fixture.

        Shape matches :attr:`~betting_backtester.football_data.FootballDataLoader.matches`
        so strategies that expect a directory (e.g. the Dixon-Coles
        rating model) can be wired to either loader without change.
        """
        return self._matches_by_id

    def events(self) -> Iterator[Event]:
        """Yield the canonical event stream.

        Re-callable: every call constructs a fresh
        ``random.Random(seed)`` and yields an equivalent, independent
        event sequence.
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

    def _build_odds_fields(self, *, arb: bool) -> dict[str, SelectionOdds]:
        probs = self._config.true_probs
        return {
            "home": self._make_selection_odds(probs.home, arb=arb),
            "draw": self._make_selection_odds(probs.draw, arb=arb),
            "away": self._make_selection_odds(probs.away, arb=arb),
        }

    def _make_selection_odds(self, p: float, *, arb: bool) -> SelectionOdds:
        cfg = self._config
        fair = 1.0 / p
        if arb:
            back_price = fair / (1.0 - cfg.arb_margin)
        else:
            back_price = fair / (1.0 + cfg.half_spread)
        # Uniform lay/back invariant, applied on top of whichever back
        # reference was chosen: ``lay / back = (1 + half_spread) ** 2``
        # for every match. See the module docstring.
        lay_price = back_price * (1.0 + cfg.half_spread) ** 2
        return SelectionOdds(back_price=back_price, lay_price=lay_price)

    def _stream(self) -> Iterator[Event]:
        cfg = self._config
        rng = random.Random(cfg.seed)
        buffered: list[Event] = []
        for i, match in enumerate(self._matches):
            # Pinned per-match draw order: schedule first, outcome
            # second. FixedArbSchedule.has_arb consumes zero draws so
            # the outcome sample comes from the same RNG position as
            # in SyntheticGenerator; BernoulliArbSchedule consumes
            # one draw before the outcome sample.
            is_arb = cfg.schedule.has_arb(i, rng)
            outcome = self._sample_outcome(rng)
            home_goals, away_goals = _PLACEHOLDER_SCORES[outcome]
            odds_fields = self._arb_odds if is_arb else self._non_arb_odds
            buffered.append(
                OddsAvailable(
                    snapshot=OddsSnapshot(
                        match_id=match.match_id,
                        timestamp=match.kickoff - cfg.odds_lead,
                        **odds_fields,
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
        buffered.sort(key=stream_sort_key)
        yield from buffered

    def _sample_outcome(self, rng: random.Random) -> Selection:
        # Exactly one RNG call per match, matching
        # :meth:`SyntheticGenerator._sample_outcome` byte-for-byte so
        # the cross-generator outcome equivalence claim in the module
        # docstring holds.
        u = rng.random()
        cumulative = 0.0
        probs = self._config.true_probs
        for selection in _SELECTION_ORDER:
            cumulative += probs.for_selection(selection)
            if u < cumulative:
                return selection
        return _SELECTION_ORDER[-1]


__all__ = [
    "ArbSchedule",
    "ArbitrageGenerator",
    "ArbitrageGeneratorConfig",
    "BernoulliArbSchedule",
    "FixedArbSchedule",
]
