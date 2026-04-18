"""Module 4b: reporting data types.

Wraps the :class:`RawBacktestOutput` produced by module 4a into a flat,
immutable :class:`BacktestResult` with a per-match equity curve and
aggregate summary metrics. DataFrame and Parquet helpers live here too;
the bootstrap confidence interval on yield lives in
:mod:`betting_backtester.reporting`.

Design notes worth surfacing:

* **Flat shape.** ``BacktestResult`` does not wrap the raw output; the
  six fields are laid out directly so walk-forward aggregation (module
  8) can build one without reconstructing a ``RawBacktestOutput``.
* **Per-match equity granularity.** One :class:`EquityPoint` per match
  that produced at least one settled bet, timestamped at that match's
  settlement. A baseline point at ``t0`` is always prepended.
* **Hit rate uses strict positivity.** A bet with ``gross_pnl == 0``
  (possible for some lay edge cases) does not count as a win. See
  :class:`SummaryMetrics`.
* **Metrics surface is intentionally tight.** No per-side or
  per-rejection-reason breakdowns -- those are derivable from
  :meth:`BacktestResult.ledger_dataframe` /
  :meth:`BacktestResult.rejections_dataframe` at call sites that
  actually need them.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from betting_backtester.backtester import (
    RawBacktestOutput,
    RejectedOrder,
    SettledBet,
)

# Tolerance on the ``bankroll_after == starting_bankroll + running_net``
# cross-check in :meth:`BacktestResult.from_raw`. 4a already enforces this
# identity internally per-match (``_BANKROLL_INVARIANT_TOLERANCE = 1e-6``);
# the boundary check here uses the same tolerance so a drift is caught
# without flagging legitimate accumulated float error on long runs.
_LEDGER_DERIVATION_TOLERANCE = 1e-6


def _require_utc(name: str, ts: datetime) -> None:
    if ts.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC (offset 0)")


def group_ledger_by_match(
    ledger: Sequence[SettledBet],
) -> list[tuple[str, list[SettledBet]]]:
    """Group a ledger into contiguous ``(match_id, bets)`` runs.

    The 4a backtester settles every bet on a given match in one step,
    so all bets for one match are contiguous in the ledger. This
    helper encodes that invariant: if a ``match_id`` reappears in a
    non-adjacent group, :class:`ValueError` is raised. Both
    :meth:`BacktestResult.from_raw` and
    :func:`~betting_backtester.reporting.compute_yield_ci` depend on
    contiguity, so a single shared helper keeps the check in one place.
    """
    groups: list[tuple[str, list[SettledBet]]] = []
    seen: set[str] = set()
    current_id: str | None = None
    for bet in ledger:
        if bet.match_id != current_id:
            if bet.match_id in seen:
                raise ValueError(
                    f"match_id {bet.match_id!r} reappears non-contiguously "
                    "in the ledger; the 4a backtester invariant is that all "
                    "bets for one match settle together."
                )
            seen.add(bet.match_id)
            groups.append((bet.match_id, [bet]))
            current_id = bet.match_id
        else:
            groups[-1][1].append(bet)
    return groups


class EquityPoint(BaseModel):
    """One snapshot of realised wealth over the run.

    Emitted per match (not per bet): matches with multiple settled bets
    collapse to a single point carrying the last ledger row's
    ``bankroll_after``, so the curve reads as "state after each market
    settled". A baseline point anchored at ``t0`` is always the first
    element of the curve.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(description="UTC instant of this snapshot.")
    bankroll: float = Field(allow_inf_nan=False)
    realised_pnl: float = Field(allow_inf_nan=False)
    cumulative_turnover: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate(self) -> EquityPoint:
        _require_utc("timestamp", self.timestamp)
        return self


class SummaryMetrics(BaseModel):
    """Aggregate performance metrics over one run.

    Definitions:

    * ``hit_rate`` -- fraction of bets with ``gross_pnl > 0``. Bets with
      ``gross_pnl == 0`` (possible for some lay edge cases) count as
      not-a-win. ``None`` iff ``n_bets == 0``.
    * ``roi`` -- ``net_pnl / turnover``. ``None`` iff ``turnover == 0``.
    * ``max_drawdown`` -- maximum peak-to-trough decline on the equity
      curve, non-negative. ``None`` iff the equity curve is empty or
      consists of only the t0 baseline.

    Per-side and per-rejection-reason breakdowns are intentionally
    omitted. They are trivially derivable from
    :meth:`BacktestResult.ledger_dataframe` and
    :meth:`BacktestResult.rejections_dataframe`; keeping them out of
    this type preserves a tight, stable metrics surface.
    """

    model_config = ConfigDict(frozen=True)

    n_bets: int = Field(ge=0)
    n_rejections: int = Field(ge=0)
    turnover: float = Field(ge=0.0, allow_inf_nan=False)
    gross_pnl: float = Field(allow_inf_nan=False)
    total_commission: float = Field(ge=0.0, allow_inf_nan=False)
    net_pnl: float = Field(allow_inf_nan=False)
    hit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    roi: float | None = Field(default=None, allow_inf_nan=False)
    max_drawdown: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)


def _max_drawdown(curve: Sequence[EquityPoint]) -> float | None:
    """Maximum peak-to-trough decline on an equity curve, non-negative.

    Single pass tracking a running peak; for each subsequent point the
    drop ``peak - bankroll`` is non-negative by construction, so the
    maximum is too. Returns ``0.0`` for monotonically non-decreasing
    curves. Returns ``None`` if ``curve`` has fewer than two points
    (empty, or only the t0 baseline with no settled bets).
    """
    if len(curve) < 2:
        return None
    running_peak = curve[0].bankroll
    max_dd = 0.0
    for point in curve[1:]:
        if point.bankroll > running_peak:
            running_peak = point.bankroll
        drop = running_peak - point.bankroll
        if drop > max_dd:
            max_dd = drop
    return max_dd


class BacktestResult(BaseModel):
    """Immutable, flat record of one backtest run.

    Sole construction path is :meth:`from_raw`. The Backtester itself
    returns :class:`RawBacktestOutput`; wrapping into a
    ``BacktestResult`` is deferred so the walk-forward evaluator in
    module 8 can aggregate raw outputs before applying the reporting
    layer.
    """

    model_config = ConfigDict(frozen=True)

    ledger: tuple[SettledBet, ...]
    rejections: tuple[RejectedOrder, ...]
    equity_curve: tuple[EquityPoint, ...]
    summary_metrics: SummaryMetrics
    starting_bankroll: float = Field(gt=0.0, allow_inf_nan=False)
    t0: datetime

    @model_validator(mode="after")
    def _validate(self) -> BacktestResult:
        _require_utc("t0", self.t0)
        return self

    @classmethod
    def from_raw(
        cls,
        raw: RawBacktestOutput,
        starting_bankroll: float,
        t0: datetime,
    ) -> BacktestResult:
        """Wrap a :class:`RawBacktestOutput` into a ``BacktestResult``.

        Single pass over the ledger builds both the equity curve and
        the summary metrics. Equity points are emitted one per match,
        using the last ledger row's ``bankroll_after`` for that match.

        No validation that ``t0`` precedes the first settlement;
        callers are responsible for supplying a sensible anchor
        (typically the first event's timestamp).
        """
        _require_utc("t0", t0)

        groups = group_ledger_by_match(raw.ledger)

        equity_curve: list[EquityPoint] = [
            EquityPoint(
                timestamp=t0,
                bankroll=starting_bankroll,
                realised_pnl=0.0,
                cumulative_turnover=0.0,
            )
        ]

        running_turnover = 0.0
        running_gross = 0.0
        running_commission = 0.0
        running_net = 0.0
        wins = 0
        for _, bets in groups:
            for bet in bets:
                running_turnover += bet.stake
                running_gross += bet.gross_pnl
                running_commission += bet.commission
                running_net += bet.net_pnl
                if bet.gross_pnl > 0.0:
                    wins += 1
            last = bets[-1]
            # Cross-check the 4a bankroll invariant at the 4b derivation
            # boundary: ``bankroll_after`` must equal ``starting_bankroll +
            # running_net`` up to the same tolerance 4a enforces. 4a already
            # checks this internally, so a drift here would indicate a bug in
            # either layer or a desynchronised refactor.
            expected_bankroll = starting_bankroll + running_net
            drift = abs(last.bankroll_after - expected_bankroll)
            if drift > _LEDGER_DERIVATION_TOLERANCE:
                raise ValueError(
                    f"ledger row {last.bet_id!r} has bankroll_after "
                    f"{last.bankroll_after}, expected {expected_bankroll} "
                    f"(starting_bankroll + running net_pnl); drift {drift} "
                    f"exceeds tolerance {_LEDGER_DERIVATION_TOLERANCE}"
                )
            equity_curve.append(
                EquityPoint(
                    timestamp=last.settled_at,
                    bankroll=last.bankroll_after,
                    realised_pnl=running_net,
                    cumulative_turnover=running_turnover,
                )
            )

        n_bets = len(raw.ledger)
        n_rejections = len(raw.rejections)
        hit_rate = wins / n_bets if n_bets > 0 else None
        roi = running_net / running_turnover if running_turnover > 0.0 else None
        max_dd = _max_drawdown(equity_curve)

        summary = SummaryMetrics(
            n_bets=n_bets,
            n_rejections=n_rejections,
            turnover=running_turnover,
            gross_pnl=running_gross,
            total_commission=running_commission,
            net_pnl=running_net,
            hit_rate=hit_rate,
            roi=roi,
            max_drawdown=max_dd,
        )

        return cls(
            ledger=raw.ledger,
            rejections=raw.rejections,
            equity_curve=tuple(equity_curve),
            summary_metrics=summary,
            starting_bankroll=starting_bankroll,
            t0=t0,
        )

    # ----- DataFrame helpers -------------------------------------------------

    def ledger_dataframe(self) -> pd.DataFrame:
        """One row per settled bet.

        Columns: ``bet_id``, ``match_id``, ``selection``, ``side``,
        ``price``, ``stake``, ``placed_at``, ``committed_funds``,
        ``settled_at``, ``outcome``, ``gross_pnl``, ``commission``,
        ``net_pnl``, ``bankroll_after``. Enum fields are emitted as
        their string values so the frame round-trips cleanly to
        Parquet.
        """
        rows = [
            {
                "bet_id": b.bet_id,
                "match_id": b.match_id,
                "selection": b.selection.value,
                "side": b.side.value,
                "price": b.price,
                "stake": b.stake,
                "placed_at": b.placed_at,
                "committed_funds": b.committed_funds,
                "settled_at": b.settled_at,
                "outcome": b.outcome.value,
                "gross_pnl": b.gross_pnl,
                "commission": b.commission,
                "net_pnl": b.net_pnl,
                "bankroll_after": b.bankroll_after,
            }
            for b in self.ledger
        ]
        columns = [
            "bet_id",
            "match_id",
            "selection",
            "side",
            "price",
            "stake",
            "placed_at",
            "committed_funds",
            "settled_at",
            "outcome",
            "gross_pnl",
            "commission",
            "net_pnl",
            "bankroll_after",
        ]
        return pd.DataFrame(rows, columns=columns)

    def rejections_dataframe(self) -> pd.DataFrame:
        """One row per rejected order.

        Columns: ``rejected_at``, ``reason``, ``detail``,
        ``order_match_id``, ``order_selection``, ``order_side``,
        ``order_price``, ``order_stake``. The nested :class:`BetOrder`
        is flattened into ``order_*`` columns so consumers do not have
        to unpack a dict-valued cell.
        """
        rows = [
            {
                "rejected_at": r.rejected_at,
                "reason": r.reason.value,
                "detail": r.detail,
                "order_match_id": r.order.match_id,
                "order_selection": r.order.selection.value,
                "order_side": r.order.side.value,
                "order_price": r.order.price,
                "order_stake": r.order.stake,
            }
            for r in self.rejections
        ]
        columns = [
            "rejected_at",
            "reason",
            "detail",
            "order_match_id",
            "order_selection",
            "order_side",
            "order_price",
            "order_stake",
        ]
        return pd.DataFrame(rows, columns=columns)

    def equity_curve_dataframe(self) -> pd.DataFrame:
        """One row per equity point.

        Columns: ``timestamp``, ``bankroll``, ``realised_pnl``,
        ``cumulative_turnover``. The first row is the ``t0`` baseline.
        """
        rows = [
            {
                "timestamp": p.timestamp,
                "bankroll": p.bankroll,
                "realised_pnl": p.realised_pnl,
                "cumulative_turnover": p.cumulative_turnover,
            }
            for p in self.equity_curve
        ]
        columns = [
            "timestamp",
            "bankroll",
            "realised_pnl",
            "cumulative_turnover",
        ]
        return pd.DataFrame(rows, columns=columns)

    # ----- Parquet helpers ---------------------------------------------------

    def write_ledger_parquet(self, path: str | Path) -> None:
        """Write the ledger DataFrame to ``path`` as Parquet."""
        self.ledger_dataframe().to_parquet(path)

    def write_rejections_parquet(self, path: str | Path) -> None:
        """Write the rejections DataFrame to ``path`` as Parquet."""
        self.rejections_dataframe().to_parquet(path)

    def write_equity_curve_parquet(self, path: str | Path) -> None:
        """Write the equity-curve DataFrame to ``path`` as Parquet."""
        self.equity_curve_dataframe().to_parquet(path)

    def write_all_parquet(self, directory: str | Path) -> None:
        """Write ``ledger.parquet``, ``rejections.parquet`` and
        ``equity_curve.parquet`` under ``directory``.

        The directory must already exist; this is a convenience
        wrapper over the three single-file writers, not a filesystem
        setup routine.
        """
        directory = Path(directory)
        self.write_ledger_parquet(directory / "ledger.parquet")
        self.write_rejections_parquet(directory / "rejections.parquet")
        self.write_equity_curve_parquet(directory / "equity_curve.parquet")


__all__ = [
    "BacktestResult",
    "EquityPoint",
    "SummaryMetrics",
    "group_ledger_by_match",
]
