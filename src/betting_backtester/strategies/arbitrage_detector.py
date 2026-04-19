"""ArbitrageDetector: back-side three-way arbitrage on 1X2 markets.

Module 10. The final shipped strategy. Unlike
:class:`~betting_backtester.strategies.xg_poisson.XgPoissonStrategy`,
which takes a view on probabilities, this one takes a view on the
market's internal consistency: when the displayed back book sums to
less than 1, backing all three selections in the right proportions
locks in a profit independent of the outcome. The strategy does not
forecast; it reads the book.

Decision rule
-------------

For each incoming :class:`~betting_backtester.models.OddsSnapshot`
compute

.. code-block:: text

    implied_sum = 1/b_home + 1/b_draw + 1/b_away

where each ``b_*`` is the corresponding ``back_price``. An
arbitrage is present iff ``implied_sum < 1 - min_margin``.
``min_margin`` defaults to 0; raising it filters out marginal arbs
that are likely to evaporate before settlement or be swallowed by
slippage in a real venue.

Stake allocation (equal-profit staking)
---------------------------------------

When an arb triggers, the strategy stakes

.. code-block:: text

    required = total_stake_fraction * bankroll
    s_i      = required * (1 / b_i) / implied_sum        for i in {H, D, A}

and emits three ``BetOrder`` instances, all with ``Side.BACK``, at
the snapshot's back prices. The stakes
sum to ``required`` and the payoff on any winning selection is
``s_i * b_i = required / implied_sum``, a constant; gross P&L per
arb is therefore outcome-invariant and equal to

.. code-block:: text

    gross = required * (1 - implied_sum) / implied_sum
          = required * arb_margin / (1 - arb_margin)

where ``arb_margin = 1 - implied_sum`` is the book-percentage gap.
Post-commission net is ``gross * (1 - rate)`` under the Betfair-style
:class:`~betting_backtester.commission.NetWinningsCommission`: the
three bets on the same market are charged as one, so the arb's net
profit survives as a clean multiplicative scaling. Lay-side arbs are
a v2 extension; v1 ships back-side only.

Bankroll basis
--------------

Selected by the keyword-only ``bankroll_basis``:

* ``"available_cash"`` (**default**, different from
  :class:`~betting_backtester.strategies.xg_poisson.XgPoissonStrategy`'s) --
  :attr:`~betting_backtester.backtester.PortfolioView.available_bankroll`,
  i.e. cash after current commitments. Under a realistic arb
  stream, matches settle before the next arb arrives, so the cash
  basis gives a clean per-arb sizing that is guaranteed to fit in
  actual cash. Chosen as the default because arbs produce
  deterministic returns -- unlike Kelly on a probability model,
  there is no variance argument for sizing on realised wealth over
  available cash.
* ``"realised_wealth"`` --
  ``starting_bankroll + realised_pnl``. Larger when open bets tie
  up cash elsewhere; in a v1-style stream with fully-settled
  non-overlapping arbs the two bases are numerically equal.

Independently of the sizing basis, the all-or-nothing cash check
uses ``available_bankroll`` directly: if the calculated
``required`` exceeds the actual cash on hand, the strategy logs a
WARNING and emits no orders. Partial arbitrage -- placing two of
the three legs because cash ran out -- would replace a deterministic
payoff with an outcome-dependent one, so the policy is strictly
all-or-nothing.

Statelessness and determinism
-----------------------------

:meth:`fit` and :meth:`on_settled` are deliberate no-ops. The
strategy holds no per-match history, no model, no training state;
the only internal configuration is the three construction
parameters, all behind read-only properties. Identical inputs
produce byte-identical orders on every call.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Literal

from betting_backtester.backtester import BetOrder, PortfolioView, Side
from betting_backtester.models import (
    Event,
    MatchResult,
    OddsSnapshot,
    Selection,
)

_logger = logging.getLogger(__name__)

_VALID_BANKROLL_BASES: frozenset[str] = frozenset(
    {"available_cash", "realised_wealth"}
)


class ArbitrageDetector:
    """Three-way back-side arbitrage detector with equal-profit sizing.

    Implements the :class:`~betting_backtester.backtester.Strategy`
    protocol. Stateless; see the module docstring for the decision
    rule, stake allocation, and bankroll-basis semantics.
    """

    __slots__ = (
        "_total_stake_fraction",
        "_min_margin",
        "_bankroll_basis",
    )

    def __init__(
        self,
        total_stake_fraction: float = 0.5,
        min_margin: float = 0.0,
        *,
        bankroll_basis: Literal[
            "available_cash", "realised_wealth"
        ] = "available_cash",
    ) -> None:
        """Validate hyperparameters and initialise the strategy.

        Parameters
        ----------
        total_stake_fraction:
            Fraction of the chosen bankroll committed on each
            detected arb. ``required = total_stake_fraction *
            bankroll``. Must be finite and in ``(0, 1]``. Values
            close to 1 are numerically allowed but leave essentially
            no cash slack for the three-bet market, which can
            interact with float drift in stake sums; the v1 default
            of 0.5 has ample slack.
        min_margin:
            Minimum book-percentage gap required to trigger. An arb
            is taken iff ``1 - implied_sum > min_margin``. Must be
            finite and in ``[0, 1)``; default 0 takes every strict
            arb.
        bankroll_basis:
            Keyword-only. ``"available_cash"`` (default) sizes on
            :attr:`PortfolioView.available_bankroll`;
            ``"realised_wealth"`` sizes on ``starting_bankroll +
            realised_pnl``. See the module docstring for the
            tradeoff.

        Raises
        ------
        ValueError
            Any numeric parameter is non-finite or outside its
            documented range, or ``bankroll_basis`` is not a
            recognised label.
        """
        if not math.isfinite(total_stake_fraction):
            raise ValueError(
                f"total_stake_fraction must be finite, got "
                f"{total_stake_fraction!r}"
            )
        if not 0.0 < total_stake_fraction <= 1.0:
            raise ValueError(
                f"total_stake_fraction must be in (0, 1], got "
                f"{total_stake_fraction}"
            )
        if not math.isfinite(min_margin):
            raise ValueError(f"min_margin must be finite, got {min_margin!r}")
        if not 0.0 <= min_margin < 1.0:
            raise ValueError(f"min_margin must be in [0, 1), got {min_margin}")
        if bankroll_basis not in _VALID_BANKROLL_BASES:
            raise ValueError(
                f"bankroll_basis must be one of "
                f"{sorted(_VALID_BANKROLL_BASES)}, got {bankroll_basis!r}"
            )

        self._total_stake_fraction: float = float(total_stake_fraction)
        self._min_margin: float = float(min_margin)
        self._bankroll_basis: Literal[
            "available_cash", "realised_wealth"
        ] = bankroll_basis

    # ----- read-only provenance -----------------------------------------------

    @property
    def total_stake_fraction(self) -> float:
        """Fraction of the sizing bankroll committed per detected arb."""
        return self._total_stake_fraction

    @property
    def min_margin(self) -> float:
        """Minimum book-percentage gap required to trigger."""
        return self._min_margin

    @property
    def bankroll_basis(self) -> Literal["available_cash", "realised_wealth"]:
        """Which bankroll field is read from :class:`PortfolioView` for sizing."""
        return self._bankroll_basis

    # ----- Strategy protocol --------------------------------------------------

    def fit(self, history: Iterable[Event]) -> None:
        """No-op. :class:`ArbitrageDetector` is stateless."""
        del history  # explicitly unused

    def on_odds(
        self, snapshot: OddsSnapshot, portfolio: PortfolioView
    ) -> list[BetOrder]:
        """Emit zero orders or three back orders on an arbitrage.

        Returns ``[]`` when
        ``1 / b_home + 1 / b_draw + 1 / b_away >= 1 - min_margin``
        (no arb, or not tight enough), when the chosen bankroll
        basis yields a non-positive value, or when the computed
        ``required`` exceeds
        :attr:`PortfolioView.available_bankroll` (all-or-nothing
        cash guard; a WARNING is logged in this case because the
        strategy detected a real arb but could not act on it).

        Otherwise emits three :class:`BetOrder` s, one per selection,
        all ``Side.BACK``, priced at the snapshot's back prices,
        with stakes chosen for equal profit across outcomes.
        """
        back_home = snapshot.home.back_price
        back_draw = snapshot.draw.back_price
        back_away = snapshot.away.back_price
        implied_sum = (1.0 / back_home) + (1.0 / back_draw) + (1.0 / back_away)

        if implied_sum >= 1.0 - self._min_margin:
            return []

        bankroll = self._bankroll_for_sizing(portfolio)
        if bankroll <= 0.0:
            return []

        required = self._total_stake_fraction * bankroll
        if portfolio.available_bankroll < required:
            _logger.warning(
                "ArbitrageDetector.on_odds: arb detected on %s "
                "(implied_sum=%s) but required stake %s exceeds "
                "available_bankroll %s; skipping to avoid partial "
                "arbitrage.",
                snapshot.match_id,
                implied_sum,
                required,
                portfolio.available_bankroll,
            )
            return []

        orders: list[BetOrder] = []
        for selection, back_price in (
            (Selection.HOME, back_home),
            (Selection.DRAW, back_draw),
            (Selection.AWAY, back_away),
        ):
            stake = required * (1.0 / back_price) / implied_sum
            orders.append(
                BetOrder(
                    match_id=snapshot.match_id,
                    selection=selection,
                    side=Side.BACK,
                    price=back_price,
                    stake=stake,
                )
            )
        return orders

    def on_settled(self, result: MatchResult, portfolio: PortfolioView) -> None:
        """No-op. :class:`ArbitrageDetector` tracks no per-settlement state."""
        del result, portfolio  # explicitly unused

    # ----- internals ----------------------------------------------------------

    def _bankroll_for_sizing(self, portfolio: PortfolioView) -> float:
        """Return the bankroll used as the sizing base.

        See the module docstring for the tradeoff between the two
        bases. Kept as a method rather than inlined so the choice
        has one named home.
        """
        if self._bankroll_basis == "realised_wealth":
            return portfolio.starting_bankroll + portfolio.realised_pnl
        return portfolio.available_bankroll


__all__ = ["ArbitrageDetector"]
