"""FavouriteBacker: a trivial Strategy that backs the market favourite.

Module 6. The first real :class:`~betting_backtester.backtester.Strategy`
implementation. Deliberately unthinking: its purpose is to prove the
end-to-end pipeline (``EventSource`` -> ``Backtester`` -> ``BacktestResult``)
works against real data and to provide an unblinking baseline against
which smarter future strategies (xG/Poisson in module 9, arbitrage in
module 10) can be compared. Expected outcome over a full season: small
losses honestly reflecting the bookmaker margin -- a feature, not a bug.

Selection rule (documented here and restated on :meth:`FavouriteBacker.on_odds`):
the favourite is the selection with the lowest ``back_price`` on the
snapshot. Ties are broken in canonical 1X2 order, HOME before DRAW
before AWAY. "Lowest back price" is equivalent to "highest
market-implied probability" on a one-bookmaker snapshot; the prose
characterisation is the latter, but we implement via ``min`` on
``back_price`` because that is what the snapshot exposes directly.

Design choices worth naming:

* **Plain class with ``__slots__``, not a Pydantic model.** The
  :class:`~betting_backtester.backtester.Strategy` contract is a
  ``Protocol``, so there is no subclassing reason to drag in Pydantic.
  A single validated ``float`` behind a read-only property mirrors the
  :class:`~betting_backtester.commission.NetWinningsCommission` pattern
  and keeps the surface minimal. ``__slots__`` makes the
  "read-only attribute after construction" contract literal: no
  ``__dict__`` means no sneaking extra state onto an instance.
* **Stateless.** The constructor stores one immutable configuration
  value; :meth:`fit` and :meth:`on_settled` are deliberate no-ops.
  Determinism is inherited from the backtester -- given the same
  event stream this strategy emits byte-identical orders on every run.
* **One order per snapshot, always.** No price filters, no warnings,
  no bankroll checks. If the backtester rejects the order (e.g.
  ``INSUFFICIENT_BANKROLL``) it appears in ``result.rejections``; the
  strategy itself does not read rejection signals.
* **Back only.** ``Side.BACK`` is hard-coded. A "lay the favourite"
  counterpart, if ever wanted, is a separate class in a separate file.

Assumption on snapshot shape. Every :class:`~betting_backtester.models.SelectionOdds`
enforces ``back_price > 1.0`` at construction, so every incoming
snapshot has three well-defined, strictly-positive back prices and the
``min`` below is always well-defined. There is no "unusable snapshot"
failure mode under either shipped loader.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from betting_backtester.backtester import BetOrder, PortfolioView, Side
from betting_backtester.models import (
    Event,
    MatchResult,
    OddsSnapshot,
    Selection,
)


class FavouriteBacker:
    """Back the market favourite for a fixed flat stake on every snapshot.

    Selection rule: the favourite is the selection with the lowest
    ``back_price`` on the snapshot. Ties are broken in canonical 1X2
    order, HOME before DRAW before AWAY -- documented behaviour, not a
    side-effect of Python's stable ``min`` that readers are expected to
    know.

    Implements the :class:`~betting_backtester.backtester.Strategy`
    protocol. Stateless: the only configuration is the flat stake,
    stored behind a read-only :attr:`stake` property. :meth:`fit` and
    :meth:`on_settled` are deliberate no-ops.
    """

    __slots__ = ("_stake",)

    def __init__(self, stake: float) -> None:
        """Construct with a fixed flat stake in bankroll currency units.

        ``stake`` must be a finite, strictly-positive ``float``.
        Non-finite or non-positive values raise :class:`ValueError`
        immediately rather than failing later inside a callback when
        :class:`~betting_backtester.backtester.BetOrder` construction
        would reject the same input -- fail at the earliest
        construction site the user controls.
        """
        if not math.isfinite(stake):
            raise ValueError(f"stake must be finite, got {stake!r}")
        if stake <= 0.0:
            raise ValueError(f"stake must be positive, got {stake}")
        self._stake: float = float(stake)

    @property
    def stake(self) -> float:
        """The flat stake in currency units applied to every emitted order."""
        return self._stake

    def fit(self, history: Iterable[Event]) -> None:
        """No-op. This strategy has no model to train."""

    def on_odds(
        self, snapshot: OddsSnapshot, portfolio: PortfolioView
    ) -> list[BetOrder]:
        """Emit exactly one :attr:`Side.BACK` order on the favourite.

        Favourite = the selection with the lowest ``back_price`` on the
        snapshot; ties broken HOME before DRAW before AWAY.
        Implemented as ``min`` over the canonical ``(HOME, DRAW, AWAY)``
        pair sequence: Python's ``min`` returns the first minimum in
        iteration order, which bakes the tie-break into the data
        layout rather than a separate branch.

        The emitted order is priced at the favourite's exact
        ``back_price`` so the backtester's snapshot check
        (:meth:`~betting_backtester.backtester.Backtester._process_order`)
        accepts it; the only remaining rejection path is
        ``INSUFFICIENT_BANKROLL``, which this strategy does not react
        to.
        """
        favourite, odds = min(
            (
                (Selection.HOME, snapshot.home),
                (Selection.DRAW, snapshot.draw),
                (Selection.AWAY, snapshot.away),
            ),
            key=lambda pair: pair[1].back_price,
        )
        return [
            BetOrder(
                match_id=snapshot.match_id,
                selection=favourite,
                side=Side.BACK,
                price=odds.back_price,
                stake=self._stake,
            )
        ]

    def on_settled(self, result: MatchResult, portfolio: PortfolioView) -> None:
        """No-op. This strategy tracks no state of its own."""


__all__ = ["FavouriteBacker"]
