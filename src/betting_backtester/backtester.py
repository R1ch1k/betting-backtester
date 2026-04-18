"""Event-driven backtester core: lifecycle types, strategy protocol, main loop.

This is module 4a. It owns the bet lifecycle (order -> pending -> settled), the
rejections log, the bankroll ledger, and the main event dispatch loop. Module
4b wraps the :class:`RawBacktestOutput` produced here into a
``BacktestResult`` (equity curve, summary metrics, bootstrap CI).

The invariants enforced here, and why each matters:

* **Orders are validated at emission only.** A :class:`BetOrder` returned by
  ``strategy.on_odds`` is checked against the snapshot that prompted it. V1 has
  at most one snapshot per match, so no re-validation is needed at settlement.
* **Rejections stay off the ledger.** A rejected order is appended to the
  rejections log with its reason and detail. The ledger contains only accepted,
  settled bets, which keeps ROI, yield, and the bootstrap CI in module 4b
  operating on actually-placed bets.
* **Committed funds asymmetry.** Back bets reserve ``stake`` from cash; lay
  bets reserve ``(price - 1) * stake`` (the liability). See
  :func:`committed_funds`.
* **Bankroll invariant.**
  ``cash + sum(committed across open bets) == starting_bankroll + realised_pnl``
  at every settlement boundary. Checked per-match to fail loudly on
  accounting drift.
* **Commission applied per market.** At ``MatchSettled`` all pending bets on
  that match are resolved together and handed to
  :meth:`CommissionModel.commission_for_market` once; per-bet attributions are
  written to the corresponding :class:`SettledBet` rows. This is the only path
  that produces correct aggregation for the module 10 arbitrage strategy.
* **Settlement order.** Resolve bets first (populate ledger), then notify
  ``strategy.on_settled``. The :class:`PortfolioView` that reaches
  ``on_settled`` already reflects the match's realised P&L.
* **End-of-stream open bets.** With ``strict_settlement=True`` (default) a
  ``RuntimeError`` is raised; with ``False`` a warning is logged and the
  pending bets are dropped. Non-strict mode exists for adversarial tests.

Note on marketable orders priced better than the snapshot. A back order whose
``price`` is strictly below the snapshot's ``back_price`` is accepted and
settled at the *order's* price, not at the improved snapshot price. A lay
order priced above the snapshot's ``lay_price`` is handled symmetrically. This
is deliberate: the core loop treats the strategy's stated price as the
contract. Modelling price-improvement fills — where a better available price
is taken on the strategy's behalf — is a realism concern that belongs to
later layers (the real-data adapter in module 6 and any future in-play
handling), not the v1 core loop. The behaviour is predictable and pessimistic:
if a strategy wants a better fill, it must ask for it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from betting_backtester.commission import CommissionModel, SettledBetLine
from betting_backtester.event_source import EventSource
from betting_backtester.models import (
    Event,
    MatchResult,
    MatchSettled,
    OddsAvailable,
    OddsSnapshot,
    Selection,
)

_logger = logging.getLogger(__name__)

# Tolerance on ``abs(net_pnl - (gross_pnl - commission))``. Commission
# attribution itself is only enforced to 1e-9 by ``CommissionBreakdown``; the
# subtraction ``gross - attribution`` compounds that drift by a small constant,
# so 1e-8 is a conservative allowance without masking genuine bugs.
_NET_PNL_TOLERANCE = 1e-8

# Tolerance on the "cash + committed == starting + realised" identity, checked
# once per match. 1e-6 comfortably covers accumulated float drift across a
# long run while still flagging any real accounting mistake (which would
# produce O(stake) drift, not O(1e-9)).
_BANKROLL_INVARIANT_TOLERANCE = 1e-6


class Side(StrEnum):
    """Which side of the book a bet sits on. Lives here, not in ``models``,
    because sides are a backtester-level concept: the canonical event stream
    doesn't know about orders."""

    BACK = "back"
    LAY = "lay"


class RejectionReason(StrEnum):
    """Why an order failed validation at emission.

    Kept small on purpose: each value maps to exactly one failure path in
    :meth:`Backtester._process_order`. New reasons go here only when a new
    failure path is added; they never serve as a catch-all.
    """

    OFF_SNAPSHOT = "off_snapshot"
    INSUFFICIENT_BANKROLL = "insufficient_bankroll"
    MISSING_MATCH = "missing_match"


def committed_funds(side: Side, price: float, stake: float) -> float:
    """Funds tied up by placing one bet.

    Back: ``stake`` is debited immediately (if the bet loses, the stake is
    already gone).

    Lay: ``(price - 1) * stake`` is reserved (the liability paid out if the
    laid selection wins).
    """
    if side is Side.BACK:
        return stake
    return (price - 1.0) * stake


class BetOrder(BaseModel):
    """An order emitted by a strategy at a decision point.

    Validated against the current :class:`OddsSnapshot` when it reaches the
    backtester. No ``bet_id``: identifiers are assigned by the backtester at
    acceptance (see :meth:`Backtester._next_bet_id`).
    """

    model_config = ConfigDict(frozen=True)

    match_id: str = Field(
        min_length=1,
        description="Fixture the order targets. Must match the snapshot's match_id.",
    )
    selection: Selection = Field(description="1X2 selection to back or lay.")
    side: Side = Field(description="Back or lay.")
    price: float = Field(
        gt=1.0,
        allow_inf_nan=False,
        description="Decimal odds. Validated against the snapshot by the backtester.",
    )
    stake: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="Stake in bankroll currency units. Non-positive stakes are rejected at construction.",
    )


def _require_utc(name: str, ts: datetime) -> None:
    if ts.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC (offset 0)")


class PendingBet(BaseModel):
    """An accepted order that has not yet settled.

    Stored in :attr:`Backtester._open_bets` keyed by ``match_id``. ``placed_at``
    is the snapshot timestamp that prompted the order, not wall-clock time.
    """

    model_config = ConfigDict(frozen=True)

    bet_id: str = Field(min_length=1, description="Backtester-assigned identifier.")
    match_id: str = Field(min_length=1)
    selection: Selection
    side: Side
    price: float = Field(gt=1.0, allow_inf_nan=False)
    stake: float = Field(gt=0.0, allow_inf_nan=False)
    placed_at: datetime = Field(description="Snapshot timestamp, UTC.")
    committed_funds: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="Funds reserved from cash by this bet. See ``committed_funds``.",
    )

    @model_validator(mode="after")
    def _validate(self) -> PendingBet:
        _require_utc("placed_at", self.placed_at)
        return self


class SettledBet(BaseModel):
    """A bet after settlement: one row in the ledger.

    ``bankroll_after`` is a **realised wealth** measure, not a cash-available
    measure: it equals ``starting_bankroll + running_sum(net_pnl)`` up to and
    including this row. This equality is an invariant of the backtester,
    verified after each match in addition to the per-row
    ``net_pnl == gross_pnl - commission`` check.

    Consequently, for a multi-bet market (e.g. a back/lay pair) an
    intermediate row can temporarily read *ahead* of the machine's actual
    cash balance: the later bets in the same market still have committed
    funds or lay liabilities tied up, which only return to cash once every
    bet in the market has settled. The realised-wealth semantics are what
    module 4b's equity curve needs; the transient mismatch against cash is
    a property of settling a market one row at a time, not a bug.
    """

    model_config = ConfigDict(frozen=True)

    bet_id: str = Field(min_length=1)
    match_id: str = Field(min_length=1)
    selection: Selection
    side: Side
    price: float = Field(gt=1.0, allow_inf_nan=False)
    stake: float = Field(gt=0.0, allow_inf_nan=False)
    placed_at: datetime
    committed_funds: float = Field(gt=0.0, allow_inf_nan=False)
    settled_at: datetime
    outcome: Selection = Field(description="The selection that paid out.")
    gross_pnl: float = Field(
        allow_inf_nan=False,
        description="P&L before commission, from decision 8 in docs/DESIGN.md.",
    )
    commission: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description="Per-bet attribution from the per-market commission total.",
    )
    net_pnl: float = Field(
        allow_inf_nan=False,
        description="``gross_pnl - commission`` within _NET_PNL_TOLERANCE.",
    )
    bankroll_after: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate(self) -> SettledBet:
        _require_utc("placed_at", self.placed_at)
        _require_utc("settled_at", self.settled_at)
        if self.settled_at < self.placed_at:
            raise ValueError(
                f"settled_at ({self.settled_at}) must be >= placed_at ({self.placed_at})"
            )
        drift = abs(self.net_pnl - (self.gross_pnl - self.commission))
        if drift > _NET_PNL_TOLERANCE:
            raise ValueError(
                f"net_pnl ({self.net_pnl}) != gross_pnl - commission "
                f"({self.gross_pnl - self.commission}); drift {drift} exceeds "
                f"tolerance {_NET_PNL_TOLERANCE}"
            )
        return self


class RejectedOrder(BaseModel):
    """One rejected order with machine-readable reason and human-readable detail."""

    model_config = ConfigDict(frozen=True)

    order: BetOrder
    rejected_at: datetime = Field(
        description="The snapshot timestamp at which validation failed."
    )
    reason: RejectionReason
    detail: str = Field(
        min_length=1,
        description="Short human-readable explanation. Intended for debugging, not parsing.",
    )

    @model_validator(mode="after")
    def _validate(self) -> RejectedOrder:
        _require_utc("rejected_at", self.rejected_at)
        return self


class PortfolioView(BaseModel):
    """Read-only view handed to strategies at each callback.

    Constructed fresh every call; strategies cannot mutate the backtester's
    state through it. ``available_bankroll`` is cash after current commitments,
    i.e. what the next bet's ``committed_funds`` is compared against.
    """

    model_config = ConfigDict(frozen=True)

    available_bankroll: float = Field(allow_inf_nan=False)
    starting_bankroll: float = Field(gt=0.0, allow_inf_nan=False)
    open_bets_count: int = Field(ge=0)
    realised_pnl: float = Field(allow_inf_nan=False)


class RawBacktestOutput(BaseModel):
    """The output of one :meth:`Backtester.run` call.

    Intermediate shape: module 4b wraps this into a ``BacktestResult`` that
    adds equity curve, summary metrics, and bootstrap CI. Tuples (not lists)
    so equality is stable and the output cannot be retroactively mutated.
    """

    model_config = ConfigDict(frozen=True)

    ledger: tuple[SettledBet, ...] = Field(
        description="Accepted bets, in insertion order (placement order within a match; "
        "match-settlement order across matches)."
    )
    rejections: tuple[RejectedOrder, ...] = Field(
        description="Orders that failed validation, in the order they were emitted."
    )


@runtime_checkable
class Strategy(Protocol):
    """The only object the backtester invokes during a run.

    Contract: the strategy must not hold a reference to the event iterator or
    any forward-looking data. Everything it sees at time ``t`` comes through
    ``on_odds`` or ``on_settled``. See ``docs/DESIGN.md`` for the lookahead
    invariant.

    ``fit`` takes an ``Iterable[Event]`` rather than a ``Sequence``: strategies
    that need random access can ``list()`` the argument explicitly, while
    strategies that stream through once pay no materialisation cost.
    """

    def fit(self, history: Iterable[Event]) -> None:
        """Optional training hook. Trivial strategies implement as ``pass``."""
        ...

    def on_odds(
        self, snapshot: OddsSnapshot, portfolio: PortfolioView
    ) -> list[BetOrder]:
        """Decision point. Return zero or more orders to place."""
        ...

    def on_settled(self, result: MatchResult, portfolio: PortfolioView) -> None:
        """Notification hook after a match has been resolved. May not emit orders."""
        ...


def _gross_pnl(pending: PendingBet, outcome: Selection) -> float:
    """P&L per decision 8 in docs/DESIGN.md."""
    selection_won = pending.selection == outcome
    if pending.side is Side.BACK:
        if selection_won:
            return pending.stake * (pending.price - 1.0)
        return -pending.stake
    # LAY: lay "wins" iff the laid selection lost.
    if selection_won:
        return -pending.stake * (pending.price - 1.0)
    return pending.stake


class Backtester:
    """Single-pass, event-driven backtester.

    Re-runnable: every ``.run()`` call resets internal state, so one instance
    can be used for multiple runs. Thread safety is not a concern in v1.
    """

    def __init__(
        self,
        event_source: EventSource,
        strategy: Strategy,
        commission_model: CommissionModel,
        starting_bankroll: float,
        seed: int,
        *,
        strict_settlement: bool = True,
    ) -> None:
        """Construct a backtester.

        Parameters
        ----------
        event_source:
            Produces the canonical event stream.
        strategy:
            Object implementing the :class:`Strategy` protocol. The backtester
            does not call ``fit``; whoever owns the strategy lifecycle
            (caller, or the walk-forward evaluator in module 8) does.
        commission_model:
            Applied per match at settlement.
        starting_bankroll:
            Initial cash. Must be positive and finite.
        seed:
            Integer seed. Reserved for deterministic bootstrap sampling in
            module 4b; stored on the instance but unused by the core loop.
        strict_settlement:
            When True (default), raise ``RuntimeError`` if any bets remain
            open at end of stream. When False, log a warning and drop them.
            Non-strict mode exists for adversarial tests that deliberately
            truncate the stream.
        """
        if not math.isfinite(starting_bankroll):
            raise ValueError(f"starting_bankroll must be finite, got {starting_bankroll}")
        if starting_bankroll <= 0.0:
            raise ValueError(
                f"starting_bankroll must be positive, got {starting_bankroll}"
            )

        self._event_source: EventSource = event_source
        self._strategy: Strategy = strategy
        self._commission_model: CommissionModel = commission_model
        self._starting_bankroll: float = float(starting_bankroll)
        self._seed: int = seed
        self._strict_settlement: bool = strict_settlement

        self._cash: float = 0.0
        self._realised_pnl: float = 0.0
        self._open_bets: dict[str, list[PendingBet]] = {}
        self._ledger: list[SettledBet] = []
        self._rejections: list[RejectedOrder] = []
        self._bet_counters: dict[str, int] = {}

    @property
    def seed(self) -> int:
        """The seed passed at construction. Unused in 4a; read by module 4b."""
        return self._seed

    def run(self) -> RawBacktestOutput:
        """Consume the event stream once and return the raw ledger + rejections."""
        self._reset_state()

        for event in self._event_source.events():
            if isinstance(event, OddsAvailable):
                self._handle_odds(event)
            elif isinstance(event, MatchSettled):
                self._handle_settled(event)
            else:
                # ``Event`` is a closed union; this is a belt-and-braces guard
                # in case a new event type is added without updating the
                # dispatch. No silent fall-through.
                raise TypeError(
                    f"unexpected event type: {type(event).__name__}"
                )

        self._handle_end_of_stream()

        return RawBacktestOutput(
            ledger=tuple(self._ledger),
            rejections=tuple(self._rejections),
        )

    # ----- main-loop handlers --------------------------------------------------

    def _handle_odds(self, event: OddsAvailable) -> None:
        snapshot = event.snapshot
        orders = self._strategy.on_odds(snapshot, self._portfolio_view())
        for order in orders:
            self._process_order(order, snapshot)

    def _handle_settled(self, event: MatchSettled) -> None:
        result = event.result
        self._settle_match(result)
        self._strategy.on_settled(result, self._portfolio_view())

    def _handle_end_of_stream(self) -> None:
        if not self._open_bets:
            return
        unsettled = {mid: len(bets) for mid, bets in self._open_bets.items()}
        if self._strict_settlement:
            raise RuntimeError(
                f"unsettled bets at end of stream: {unsettled}. "
                "Every OddsAvailable must be followed by a MatchSettled for "
                "the same match_id. Pass strict_settlement=False if truncation "
                "is intentional."
            )
        # Non-strict mode: refund committed funds back to cash so the
        # internal bankroll invariant still holds after the drop. Without
        # this, cash stays artificially low by the committed total even
        # though the bets are no longer open — any downstream read of
        # _cash (including a subsequent .run()) would be corrupted.
        refunded = sum(
            bet.committed_funds
            for bets in self._open_bets.values()
            for bet in bets
        )
        _logger.warning(
            "strict_settlement=False: dropping %d unsettled bets across %d matches "
            "(refunding %s committed funds to cash): %s",
            sum(unsettled.values()),
            len(unsettled),
            refunded,
            unsettled,
        )
        self._cash += refunded
        self._open_bets.clear()

    # ----- order validation + acceptance --------------------------------------

    def _process_order(self, order: BetOrder, snapshot: OddsSnapshot) -> None:
        if order.match_id != snapshot.match_id:
            self._record_rejection(
                order,
                snapshot.timestamp,
                RejectionReason.MISSING_MATCH,
                f"order.match_id {order.match_id!r} does not match snapshot "
                f"{snapshot.match_id!r}",
            )
            return

        selection_odds = snapshot.odds_for(order.selection)
        if order.side is Side.BACK:
            if order.price > selection_odds.back_price:
                self._record_rejection(
                    order,
                    snapshot.timestamp,
                    RejectionReason.OFF_SNAPSHOT,
                    f"back price {order.price} exceeds snapshot back_price "
                    f"{selection_odds.back_price}",
                )
                return
        else:  # LAY
            if order.price < selection_odds.lay_price:
                self._record_rejection(
                    order,
                    snapshot.timestamp,
                    RejectionReason.OFF_SNAPSHOT,
                    f"lay price {order.price} below snapshot lay_price "
                    f"{selection_odds.lay_price}",
                )
                return

        committed = committed_funds(order.side, order.price, order.stake)
        if committed > self._cash:
            self._record_rejection(
                order,
                snapshot.timestamp,
                RejectionReason.INSUFFICIENT_BANKROLL,
                f"committed funds {committed} exceed available cash {self._cash}",
            )
            return

        bet_id = self._next_bet_id(order.match_id)
        pending = PendingBet(
            bet_id=bet_id,
            match_id=order.match_id,
            selection=order.selection,
            side=order.side,
            price=order.price,
            stake=order.stake,
            placed_at=snapshot.timestamp,
            committed_funds=committed,
        )
        self._cash -= committed
        self._open_bets.setdefault(order.match_id, []).append(pending)

    # ----- settlement ----------------------------------------------------------

    def _settle_match(self, result: MatchResult) -> None:
        pending_list = self._open_bets.pop(result.match_id, [])
        if not pending_list:
            return

        gross_by_id: dict[str, float] = {}
        lines: list[SettledBetLine] = []
        for pending in pending_list:
            gross = _gross_pnl(pending, result.outcome)
            gross_by_id[pending.bet_id] = gross
            lines.append(
                SettledBetLine(
                    bet_id=pending.bet_id, stake=pending.stake, gross_pnl=gross
                )
            )

        breakdown = self._commission_model.commission_for_market(lines)

        for pending in pending_list:
            gross = gross_by_id[pending.bet_id]
            attribution = breakdown.per_bet[pending.bet_id]
            net = gross - attribution
            self._cash += pending.committed_funds + gross - attribution
            self._realised_pnl += net
            self._ledger.append(
                SettledBet(
                    bet_id=pending.bet_id,
                    match_id=pending.match_id,
                    selection=pending.selection,
                    side=pending.side,
                    price=pending.price,
                    stake=pending.stake,
                    placed_at=pending.placed_at,
                    committed_funds=pending.committed_funds,
                    settled_at=result.timestamp,
                    outcome=result.outcome,
                    gross_pnl=gross,
                    commission=attribution,
                    net_pnl=net,
                    bankroll_after=self._starting_bankroll + self._realised_pnl,
                )
            )

        self._check_bankroll_invariant()

    def _check_bankroll_invariant(self) -> None:
        committed_total = sum(
            bet.committed_funds
            for bets in self._open_bets.values()
            for bet in bets
        )
        expected = self._starting_bankroll + self._realised_pnl
        actual = self._cash + committed_total
        drift = abs(actual - expected)
        if drift > _BANKROLL_INVARIANT_TOLERANCE:
            raise RuntimeError(
                f"bankroll invariant violated: cash+committed={actual}, "
                f"expected starting+realised={expected}, drift={drift}"
            )

    # ----- helpers -------------------------------------------------------------

    def _portfolio_view(self) -> PortfolioView:
        return PortfolioView(
            available_bankroll=self._cash,
            starting_bankroll=self._starting_bankroll,
            open_bets_count=sum(len(bets) for bets in self._open_bets.values()),
            realised_pnl=self._realised_pnl,
        )

    def _record_rejection(
        self,
        order: BetOrder,
        timestamp: datetime,
        reason: RejectionReason,
        detail: str,
    ) -> None:
        self._rejections.append(
            RejectedOrder(
                order=order,
                rejected_at=timestamp,
                reason=reason,
                detail=detail,
            )
        )

    def _next_bet_id(self, match_id: str) -> str:
        counter = self._bet_counters.get(match_id, 0)
        self._bet_counters[match_id] = counter + 1
        return f"{match_id}#{counter:04d}"

    def _reset_state(self) -> None:
        self._cash = self._starting_bankroll
        self._realised_pnl = 0.0
        self._open_bets = {}
        self._ledger = []
        self._rejections = []
        self._bet_counters = {}


__all__ = [
    "BetOrder",
    "Backtester",
    "PendingBet",
    "PortfolioView",
    "RawBacktestOutput",
    "RejectedOrder",
    "RejectionReason",
    "SettledBet",
    "Side",
    "Strategy",
    "committed_funds",
]
