"""Commission models for settlement-time charging on placed bets.

V1 ships a single implementation, :class:`NetWinningsCommission`, mirroring
Betfair's "percentage of net market winnings per customer per market" rule.
The protocol itself is rule-agnostic: a :class:`CommissionModel` receives one
group of settled bet lines that belong together (per-customer, per-market),
returns the total commission owed on the group, and attributes it back to the
contributing rows. The backtester owns grouping; the model owns the rule.

Precision: v1 uses ``float`` throughout, consistent with stake and price in
the rest of the codebase. Decimal is deferred as a future hardening pass if
rounding error ever becomes measurable against the bootstrap CI width on
yield -- today it is not.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Protocol, runtime_checkable

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


class SettledBetLine(BaseModel):
    """One placed bet, at the moment commission is computed on it.

    The minimal view of a settled bet that any commission rule might need.
    The backtester constructs these from its richer internal representation
    just before handing them in.
    """

    model_config = ConfigDict(frozen=True)

    bet_id: str = Field(
        min_length=1,
        description="Stable identifier used to key the per-bet commission attribution.",
    )
    stake: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Stake at placement time. Present for commission rules that charge "
            "on stake; ignored by NetWinningsCommission."
        ),
    )
    gross_pnl: float = Field(
        allow_inf_nan=False,
        description=(
            "Profit or loss in currency units at settlement, before any "
            "commission is applied. Negative for losing bets."
        ),
    )


def _freeze_per_bet(value: Mapping[str, float]) -> Mapping[str, float]:
    # Wrap in a MappingProxyType over a fresh copy so the stored mapping is
    # genuinely read-only (breakdown.per_bet["a"] = 999 raises TypeError) and
    # cannot be aliased to a dict the caller still holds a mutable reference
    # to. Runs as an AfterValidator because Pydantic's own Mapping schema
    # coerces to a plain dict before this point; a model_validator(mode="before")
    # wrapper would not survive that coercion.
    return MappingProxyType(dict(value))


class CommissionBreakdown(BaseModel):
    """Commission owed on one market, plus attribution back to its bets.

    ``per_bet`` is keyed by ``bet_id``; every input bet's id appears exactly
    once. Per-bet values are non-negative and sum to ``total`` within float
    tolerance. The stored ``per_bet`` is a ``types.MappingProxyType`` so the
    invariants cannot be silently broken by mutating the mapping after
    construction (``breakdown.per_bet["a"] = 999`` raises ``TypeError``).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    total: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description="Total commission charged on this market group.",
    )
    per_bet: Annotated[Mapping[str, float], AfterValidator(_freeze_per_bet)] = Field(
        description="Per-bet attribution, keyed by bet_id. Non-negative; sums to total.",
    )

    @model_validator(mode="after")
    def _validate(self) -> CommissionBreakdown:
        for bet_id, value in self.per_bet.items():
            if not math.isfinite(value):
                raise ValueError(f"per_bet[{bet_id!r}] must be finite, got {value!r}")
            if value < 0.0:
                raise ValueError(f"per_bet[{bet_id!r}] must be >= 0, got {value}")
        if self.total > 0.0 and not self.per_bet:
            raise ValueError(
                f"total is {self.total} but per_bet is empty; a non-zero "
                "charge must be attributed to at least one bet"
            )
        attributed = math.fsum(self.per_bet.values())
        if abs(attributed - self.total) > 1e-9:
            raise ValueError(
                f"per_bet sums to {attributed}, expected total {self.total} "
                f"(difference {attributed - self.total})"
            )
        return self


@runtime_checkable
class CommissionModel(Protocol):
    """Computes commission owed on a group of related settled bets.

    Implementations own both the aggregation rule (how the bets combine
    before applying the charge) and the attribution rule (how the total is
    apportioned back to each bet). The backtester groups bets per
    customer-market before calling.
    """

    def commission_for_market(
        self, bets: Sequence[SettledBetLine]
    ) -> CommissionBreakdown:
        """Return total commission and per-bet attribution for one market."""
        ...


class NetWinningsCommission:
    """Commission as a percentage of net market winnings, Betfair-style.

    The rule (per customer, per market):

    1. ``net = sum(b.gross_pnl for b in bets)``.
    2. If ``net <= 0``, no commission is owed; every per-bet attribution is 0.
    3. Otherwise, ``total = rate * net``, attributed pro-rata to bets with
       ``gross_pnl > 0``. Losers and break-even bets receive 0. Per-bet
       attributions sum to ``total``.

    Worked example (arbitrage case). A back bet with ``gross_pnl = +100`` and
    a lay bet with ``gross_pnl = -40`` on the same market net to ``+60``. At
    the default ``rate = 0.05``, total commission is ``3.0``; attribution is
    ``3.0`` to the back bet and ``0.0`` to the lay bet. Charging each bet in
    isolation would yield ``5.0``, a 67% overstatement -- this is why the
    arbitrage strategy in module 10 depends on net-winnings aggregation.

    The default rate is ``0.05``. That is a Betfair-ish convention, not a
    verified current rate -- Betfair's real rate has varied roughly 2-6.5%
    by product and discount over the years.
    """

    def __init__(self, rate: float = 0.05) -> None:
        if not math.isfinite(rate):
            raise ValueError(f"rate must be finite, got {rate!r}")
        if rate < 0.0:
            raise ValueError(f"rate must be >= 0, got {rate}")
        if rate > 1.0:
            raise ValueError(f"rate must be <= 1, got {rate}")
        self._rate: float = rate

    @property
    def rate(self) -> float:
        """The commission rate applied to net market winnings."""
        return self._rate

    def commission_for_market(
        self, bets: Sequence[SettledBetLine]
    ) -> CommissionBreakdown:
        if not bets:
            return CommissionBreakdown(total=0.0, per_bet={})

        seen: set[str] = set()
        for bet in bets:
            if bet.bet_id in seen:
                raise ValueError(
                    f"duplicate bet_id {bet.bet_id!r} in commission input; "
                    "the caller must supply each bet exactly once per market"
                )
            seen.add(bet.bet_id)

        per_bet: dict[str, float] = {bet.bet_id: 0.0 for bet in bets}
        net = math.fsum(bet.gross_pnl for bet in bets)
        if net <= 0.0:
            return CommissionBreakdown(total=0.0, per_bet=per_bet)

        total = self._rate * net
        winners_gross_sum = math.fsum(
            bet.gross_pnl for bet in bets if bet.gross_pnl > 0.0
        )
        # ``net > 0`` implies at least one bet has ``gross_pnl > 0``, so
        # ``winners_gross_sum > 0``. No division-by-zero guard needed.
        for bet in bets:
            if bet.gross_pnl > 0.0:
                per_bet[bet.bet_id] = total * (bet.gross_pnl / winners_gross_sum)
        return CommissionBreakdown(total=total, per_bet=per_bet)
