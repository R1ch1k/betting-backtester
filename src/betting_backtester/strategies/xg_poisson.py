"""XgPoissonStrategy: Dixon-Coles probability model with fractional Kelly sizing.

Module 9c. Composes:

* :class:`~betting_backtester.dixon_coles.DixonColesModel` -- per-team
  attack/defence ratings plus a global home-advantage parameter, fit by
  weighted L2-regularised MLE on settled matches from the training
  window (OddsAvailable events are ignored during training; a
  market-informed variant would use them, v1 does not).
* :func:`~betting_backtester.kelly.back_kelly` /
  :func:`~betting_backtester.kelly.lay_kelly` -- Kelly-optimal sizing
  for back (stake fraction) and lay (liability fraction).

Team identity plumbing
----------------------

The canonical event stream intentionally carries only ``match_id``.
Team names therefore arrive via a ``match_directory: Mapping[str, Match]``
passed at construction, per the design agreed for module 9. Both
shipped loaders already expose such a directory:
:attr:`~betting_backtester.football_data.FootballDataLoader.matches`
and :attr:`~betting_backtester.synthetic.SyntheticGenerator.matches`.
The strategy treats it as read-only and never mutates.

Bet decision rule
-----------------

For each incoming :class:`~betting_backtester.models.OddsSnapshot`, and
each of the three selections (HOME, DRAW, AWAY), the rule is:

* **Back** if ``our_prob > (1 / back_price) + edge_threshold``.
* **Lay**  if ``our_prob < (1 / lay_price)  - edge_threshold``.
* Otherwise, no order.

The three selections are evaluated independently, so one fixture can
yield up to three orders (at most one per selection: back XOR lay XOR
nothing). Rare in practice -- on a single-bookmaker snapshot with
realistic odds the edge thresholds only cross for one selection -- but
the protocol supports it and the arbitrage strategy in module 10 will
rely on the same per-selection independence.

Stake sizing: fractional Kelly with exposure cap
------------------------------------------------

For a candidate back at decimal odds ``b`` with our probability ``p``:

.. code-block:: text

    fraction = min(
        kelly_fraction * back_kelly(p, b),
        max_exposure_fraction,
    )
    stake = fraction * bankroll

For a candidate lay, the Kelly function returns a **liability**
fraction; the same cap is applied to that liability, and the stake
sent to the backer is ``liability / (b - 1)``. The exposure cap
therefore bounds *amount at risk* for both sides symmetrically, which
is the risk-consistent reading. See
:mod:`~betting_backtester.kelly` for the full stake-vs-liability
distinction -- the single most bug-prone detail in exchange sizing.

``kelly_fraction = 0.25`` (quarter-Kelly) is the v1 default: full
Kelly on a slightly miscalibrated probability produces large drawdowns,
and the literature converges on fractional Kelly for model-based
betting. ``max_exposure_fraction = 0.05`` ensures no single bet risks
more than 5% of bankroll even when Kelly is enthusiastic.

Bankroll basis for sizing
-------------------------

Two options, selected by the keyword-only ``bankroll_basis``:

* ``"realised_wealth"`` (default) -- ``starting_bankroll + realised_pnl``.
  The standard Kelly "current bankroll" interpretation; matches how bet
  sizing is conventionally modelled in the betting literature. Can
  occasionally produce a stake that exceeds available cash if multiple
  bets are open concurrently, in which case the backtester records an
  ``INSUFFICIENT_BANKROLL`` rejection on its rejections log -- not
  silent, not on the P&L ledger.
* ``"available_cash"`` -- :attr:`PortfolioView.available_bankroll`,
  i.e. cash after current commitments. Never triggers
  ``INSUFFICIENT_BANKROLL`` but de-levers the strategy relative to
  textbook Kelly when bets overlap in time. V1 has one snapshot per
  match with near-simultaneous settlement, so the two bases are
  typically identical; the knob exists for clarity, not because v1
  exercises it heavily.

Unseen-team / unseen-match handling
-----------------------------------

If a snapshot's ``match_id`` is missing from the directory, or either
team is missing from the fitted model's
:meth:`~betting_backtester.dixon_coles.DixonColesModel.known_teams`,
the strategy returns an empty order list, logs a WARNING, and
increments the :attr:`unseen_skips` counter. The counter is reset on
every :meth:`fit` call so per-window tallies (under the walk-forward
evaluator) are meaningful.

Empty-history graceful degradation
----------------------------------

Per the locked design: if :meth:`fit` receives a history with no
usable :class:`~betting_backtester.models.MatchSettled` events (either
empty, or none of the settled matches are in the directory), the
strategy does **not** raise: it sets its internal model to ``None``,
logs a WARNING explaining why, and from then on :meth:`on_odds`
returns ``[]`` until the next successful :meth:`fit`. This matters in
practice for early walk-forward windows where training coverage is
thin.

Determinism
-----------

Given an unchanged fitted model and an unchanged snapshot,
:meth:`on_odds` returns byte-identical orders across calls. The only
sources of potential non-determinism are the MLE (deterministic via
``x0 = 0`` and L-BFGS-B's deterministic convergence on the convex
objective) and the Python dictionary iteration inside
``match_directory``; the strategy never iterates the directory
(lookups only), so dict order is irrelevant.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Literal

from betting_backtester.backtester import BetOrder, PortfolioView, Side
from betting_backtester.dixon_coles import DixonColesModel, TrainingMatch
from betting_backtester.kelly import back_kelly, lay_kelly
from betting_backtester.models import (
    Event,
    Match,
    MatchResult,
    MatchSettled,
    OddsSnapshot,
    Selection,
    SelectionOdds,
)

_logger = logging.getLogger(__name__)

_VALID_BANKROLL_BASES: frozenset[str] = frozenset(
    {"realised_wealth", "available_cash"}
)


class XgPoissonStrategy:
    """Dixon-Coles model + fractional-Kelly sizing on 1X2 exchange markets.

    Implements the :class:`~betting_backtester.backtester.Strategy`
    protocol. One instance per walk-forward window (the evaluator's
    ``strategy_factory`` contract); reusing an instance across windows
    is permitted -- :meth:`fit` is idempotent -- but the walk-forward
    evaluator discourages it to prevent subtle state leaks.
    """

    __slots__ = (
        "_match_directory",
        "_edge_threshold",
        "_kelly_fraction",
        "_max_exposure_fraction",
        "_l2_reg",
        "_decay_rate",
        "_bankroll_basis",
        "_model",
        "_unseen_skips",
    )

    def __init__(
        self,
        match_directory: Mapping[str, Match],
        edge_threshold: float = 0.02,
        kelly_fraction: float = 0.25,
        max_exposure_fraction: float = 0.05,
        l2_reg: float = 0.001,
        decay_rate: float = 0.0019,
        *,
        bankroll_basis: Literal[
            "realised_wealth", "available_cash"
        ] = "realised_wealth",
    ) -> None:
        """Validate hyperparameters and initialise the unfitted strategy.

        Parameters
        ----------
        match_directory:
            Read-only ``match_id -> Match`` directory. Must be a
            :class:`~collections.abc.Mapping` and non-empty. Supplies
            team identities that the event stream does not carry.
        edge_threshold:
            Minimum absolute edge over the market-implied probability
            before a bet is considered. ``edge_threshold = 0`` enables
            any positive-edge bet. Must be finite and in ``[0, 1)``.
        kelly_fraction:
            Multiplier on full Kelly. ``1.0`` is full Kelly (aggressive);
            v1 default is ``0.25`` (quarter-Kelly). Must be finite and
            in ``(0, 1]``.
        max_exposure_fraction:
            Hard cap on *amount at risk* as a fraction of bankroll for
            any single bet. Applied to stake for backs and to liability
            for lays. Must be finite and in ``(0, 1]``.
        l2_reg:
            L2 penalty coefficient passed through to the underlying
            :class:`~betting_backtester.dixon_coles.DixonColesModel`.
            Must be finite and ``>= 0``.
        decay_rate:
            Exponential time-decay coefficient (per-day) passed
            through to the underlying model. Must be finite and
            ``>= 0``.
        bankroll_basis:
            Keyword-only. ``"realised_wealth"`` (default) sizes Kelly
            stakes on ``starting_bankroll + realised_pnl``;
            ``"available_cash"`` sizes on
            :attr:`PortfolioView.available_bankroll`. See the module
            docstring for the tradeoff.

        Raises
        ------
        TypeError
            ``match_directory`` is not a :class:`~collections.abc.Mapping`.
        ValueError
            Any numeric parameter is outside its documented range, or
            ``match_directory`` is empty, or ``bankroll_basis`` is not
            a recognised label.
        """
        if not isinstance(match_directory, Mapping):
            raise TypeError(
                f"match_directory must be a Mapping[str, Match], got "
                f"{type(match_directory).__name__}"
            )
        if len(match_directory) == 0:
            raise ValueError("match_directory must be non-empty")

        _validate_finite_in_range(
            "edge_threshold", edge_threshold, low=0.0, high=1.0, low_inclusive=True
        )
        _validate_finite_in_range(
            "kelly_fraction", kelly_fraction, low=0.0, high=1.0, high_inclusive=True
        )
        _validate_finite_in_range(
            "max_exposure_fraction",
            max_exposure_fraction,
            low=0.0,
            high=1.0,
            high_inclusive=True,
        )
        _validate_finite_non_negative("l2_reg", l2_reg)
        _validate_finite_non_negative("decay_rate", decay_rate)

        if bankroll_basis not in _VALID_BANKROLL_BASES:
            raise ValueError(
                f"bankroll_basis must be one of {sorted(_VALID_BANKROLL_BASES)}, "
                f"got {bankroll_basis!r}"
            )

        self._match_directory: Mapping[str, Match] = match_directory
        self._edge_threshold: float = float(edge_threshold)
        self._kelly_fraction: float = float(kelly_fraction)
        self._max_exposure_fraction: float = float(max_exposure_fraction)
        self._l2_reg: float = float(l2_reg)
        self._decay_rate: float = float(decay_rate)
        self._bankroll_basis: Literal[
            "realised_wealth", "available_cash"
        ] = bankroll_basis

        self._model: DixonColesModel | None = None
        self._unseen_skips: int = 0

    # ----- read-only provenance ------------------------------------------------

    @property
    def edge_threshold(self) -> float:
        """Minimum absolute edge required before sizing a bet."""
        return self._edge_threshold

    @property
    def kelly_fraction(self) -> float:
        """Multiplier on full Kelly applied during sizing."""
        return self._kelly_fraction

    @property
    def max_exposure_fraction(self) -> float:
        """Per-bet cap on amount at risk as a fraction of bankroll."""
        return self._max_exposure_fraction

    @property
    def l2_reg(self) -> float:
        """L2 regularisation coefficient forwarded to :class:`DixonColesModel`."""
        return self._l2_reg

    @property
    def decay_rate(self) -> float:
        """Time-decay coefficient forwarded to :class:`DixonColesModel`."""
        return self._decay_rate

    @property
    def bankroll_basis(self) -> Literal["realised_wealth", "available_cash"]:
        """Which bankroll field is read from :class:`PortfolioView` for sizing."""
        return self._bankroll_basis

    @property
    def unseen_skips(self) -> int:
        """Count of ``on_odds`` calls skipped because of missing directory or unseen teams.

        Reset to 0 on every :meth:`fit` call. Does NOT include settled
        matches dropped from :meth:`fit` history due to missing directory
        entries -- those are logged at WARNING level but not counted
        here, because the counter is intended to surface inference-time
        skips that affect bet placement, not training-time data gaps.
        """
        return self._unseen_skips

    @property
    def model(self) -> DixonColesModel | None:
        """The underlying fitted model, or ``None`` before :meth:`fit` / after degraded fit.

        Exposed for introspection (tests, notebooks). External callers
        should not mutate the returned object.
        """
        return self._model

    # ----- Strategy protocol ---------------------------------------------------

    def fit(self, history: Iterable[Event]) -> None:
        """Build ``TrainingMatch``-es from history and fit the Dixon-Coles model.

        Processing:

        1. Materialise the history once and note its maximum timestamp
           as the MLE's ``anchor_time``. The strategy never scans ahead
           of training events, so this timestamp is safe to derive here.
        2. Filter for :class:`MatchSettled` events and join against
           ``match_directory`` to build
           :class:`~betting_backtester.dixon_coles.TrainingMatch`
           records. Settled matches whose ``match_id`` is not in the
           directory are dropped with a WARNING (rare; indicates caller
           wiring error).
        3. If the resulting training set is empty, set
           :attr:`model` to ``None`` and return without raising -- see
           the module docstring's "empty-history graceful degradation"
           section. Subsequent :meth:`on_odds` calls return ``[]``.
        4. Otherwise construct a fresh
           :class:`~betting_backtester.dixon_coles.DixonColesModel` with
           the strategy's hyperparameters and fit.

        Re-fittable: each call replaces the prior model and resets
        :attr:`unseen_skips` to 0.
        """
        events = list(history)
        self._unseen_skips = 0

        if not events:
            self._model = None
            _logger.warning(
                "XgPoissonStrategy.fit: empty history; model not fitted, "
                "on_odds will emit no orders until the next successful fit()."
            )
            return

        anchor_time = max(event.timestamp for event in events)

        training_matches: list[TrainingMatch] = []
        for event in events:
            if not isinstance(event, MatchSettled):
                continue
            result: MatchResult = event.result
            match_info = self._match_directory.get(result.match_id)
            if match_info is None:
                _logger.warning(
                    "XgPoissonStrategy.fit: settled match %r has no entry in "
                    "match_directory; dropping from training set.",
                    result.match_id,
                )
                continue
            training_matches.append(
                TrainingMatch(
                    home_team=match_info.home,
                    away_team=match_info.away,
                    home_goals=result.home_goals,
                    away_goals=result.away_goals,
                    settled_at=result.timestamp,
                )
            )

        if not training_matches:
            self._model = None
            _logger.warning(
                "XgPoissonStrategy.fit: no trainable matches in history "
                "(0 MatchSettled events joined to match_directory); model "
                "not fitted, on_odds will emit no orders."
            )
            return

        model = DixonColesModel(
            l2_reg=self._l2_reg, decay_rate=self._decay_rate
        )
        model.fit(training_matches, anchor_time=anchor_time)
        self._model = model

    def on_odds(
        self, snapshot: OddsSnapshot, portfolio: PortfolioView
    ) -> list[BetOrder]:
        """Emit zero to three orders for the snapshot.

        Returns ``[]`` immediately when the strategy is not fitted
        (:attr:`model` is ``None``), when the snapshot's ``match_id``
        is absent from the directory, when either team in the fixture
        is unknown to the fitted model, or when the chosen bankroll
        basis is non-positive. The first three cases also increment
        :attr:`unseen_skips` (the last does not -- a depleted bankroll
        is the portfolio's state, not a data gap).

        When fitted and all lookups succeed, evaluates each of HOME,
        DRAW, AWAY independently under the back/lay decision rule
        documented in the module docstring.
        """
        model = self._model
        if model is None:
            return []

        match_info = self._match_directory.get(snapshot.match_id)
        if match_info is None:
            self._unseen_skips += 1
            _logger.warning(
                "XgPoissonStrategy.on_odds: snapshot for match %r has no "
                "match_directory entry; skipping.",
                snapshot.match_id,
            )
            return []

        home_team = match_info.home
        away_team = match_info.away
        known = model.known_teams()
        if home_team not in known or away_team not in known:
            self._unseen_skips += 1
            _logger.warning(
                "XgPoissonStrategy.on_odds: match %r has at least one unseen "
                "team (home=%r, away=%r); skipping.",
                snapshot.match_id,
                home_team,
                away_team,
            )
            return []

        bankroll = self._bankroll_for_sizing(portfolio)
        if bankroll <= 0.0:
            return []

        probabilities = model.predict(home_team, away_team)

        orders: list[BetOrder] = []
        for selection, our_prob, selection_odds in (
            (Selection.HOME, probabilities.home, snapshot.home),
            (Selection.DRAW, probabilities.draw, snapshot.draw),
            (Selection.AWAY, probabilities.away, snapshot.away),
        ):
            order = self._decide_order(
                match_id=snapshot.match_id,
                selection=selection,
                our_prob=our_prob,
                selection_odds=selection_odds,
                bankroll=bankroll,
            )
            if order is not None:
                orders.append(order)
        return orders

    def on_settled(self, result: MatchResult, portfolio: PortfolioView) -> None:
        """No-op. This strategy tracks no per-settlement state of its own."""

    # ----- internals -----------------------------------------------------------

    def _bankroll_for_sizing(self, portfolio: PortfolioView) -> float:
        """Return the bankroll used as the Kelly-sizing base.

        See the module docstring for the tradeoff between the two
        bases. Kept as a method rather than inlined so the choice is
        trivially patchable and the decision has one named home.
        """
        if self._bankroll_basis == "realised_wealth":
            return portfolio.starting_bankroll + portfolio.realised_pnl
        return portfolio.available_bankroll

    def _decide_order(
        self,
        *,
        match_id: str,
        selection: Selection,
        our_prob: float,
        selection_odds: SelectionOdds,
        bankroll: float,
    ) -> BetOrder | None:
        """Evaluate back-then-lay for one selection and size if triggered.

        Back takes precedence in the (impossible on a well-formed book
        where ``back_price <= lay_price``) event that both thresholds
        trigger. Returns ``None`` when neither side has edge, or when
        Kelly sizing collapses to a non-positive stake (which is the
        same signal: no bet).
        """
        back_price = selection_odds.back_price
        lay_price = selection_odds.lay_price
        back_implied = 1.0 / back_price
        lay_implied = 1.0 / lay_price

        if our_prob > back_implied + self._edge_threshold:
            kelly_f = back_kelly(our_prob, back_price)
            stake = self._sized_at_risk(kelly_f, bankroll)
            if stake <= 0.0:
                return None
            return BetOrder(
                match_id=match_id,
                selection=selection,
                side=Side.BACK,
                price=back_price,
                stake=stake,
            )

        if our_prob < lay_implied - self._edge_threshold:
            kelly_f = lay_kelly(our_prob, lay_price)
            liability = self._sized_at_risk(kelly_f, bankroll)
            if liability <= 0.0:
                return None
            stake = liability / (lay_price - 1.0)
            if stake <= 0.0:
                # Defensive: lay_price > 1 and liability > 0 imply
                # stake > 0, but the BetOrder validator rejects
                # non-positive stakes and it is cheaper to skip than to
                # produce a rejection.
                return None
            return BetOrder(
                match_id=match_id,
                selection=selection,
                side=Side.LAY,
                price=lay_price,
                stake=stake,
            )

        return None

    def _sized_at_risk(
        self, kelly_fraction_of_bankroll: float, bankroll: float
    ) -> float:
        """Apply the fractional-Kelly multiplier and the exposure cap.

        ``kelly_fraction_of_bankroll`` is the raw full-Kelly fraction
        returned by :func:`back_kelly` or :func:`lay_kelly` (stake for
        backs, liability for lays). The returned value is the
        corresponding currency amount at risk, after multiplying by
        ``kelly_fraction`` and capping at ``max_exposure_fraction *
        bankroll``.
        """
        if kelly_fraction_of_bankroll <= 0.0:
            return 0.0
        scaled = self._kelly_fraction * kelly_fraction_of_bankroll
        capped = min(scaled, self._max_exposure_fraction)
        return capped * bankroll


def _validate_finite_in_range(
    name: str,
    value: float,
    *,
    low: float,
    high: float,
    low_inclusive: bool = False,
    high_inclusive: bool = False,
) -> None:
    """Check ``value`` is finite and within the specified open/closed interval."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    below = value < low if low_inclusive else value <= low
    above = value > high if high_inclusive else value >= high
    if below or above:
        lb = "[" if low_inclusive else "("
        ub = "]" if high_inclusive else ")"
        raise ValueError(
            f"{name} must be in {lb}{low}, {high}{ub}, got {value}"
        )


def _validate_finite_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")


__all__ = ["XgPoissonStrategy"]
