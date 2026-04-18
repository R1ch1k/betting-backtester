"""Tests for module 4b: :class:`BacktestResult`, :class:`EquityPoint`,
:class:`SummaryMetrics`, and the DataFrame / Parquet helpers.

Organisation:

* ``TestGroupLedgerByMatch`` -- the shared contiguity helper used by
  both ``backtest_result`` and ``reporting``.
* ``TestEquityPoint`` / ``TestSummaryMetrics`` -- Pydantic model surface.
* ``TestFromRaw`` -- the primary correctness gate: hand-computed
  expected metrics and equity curves on small raw outputs, the arb
  case, hit-rate edge cases, the bankroll cross-check, and
  same-inputs byte-equality.
* ``TestMaxDrawdown`` -- shape coverage (monotonic, V, W) exercised via
  ``from_raw``.
* ``TestDataFrames`` -- column order and the ``order_*`` flattening
  convention on rejections.
* ``TestParquet`` -- tmp_path round-trip for each of the three single
  writers plus ``write_all_parquet``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from betting_backtester.backtest_result import (
    BacktestResult,
    EquityPoint,
    SummaryMetrics,
    group_ledger_by_match,
)
from betting_backtester.backtester import (
    BetOrder,
    RawBacktestOutput,
    RejectedOrder,
    RejectionReason,
    SettledBet,
    Side,
)
from betting_backtester.models import Selection

UTC = timezone.utc

T0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
SETTLE_A = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)
SETTLE_B = datetime(2024, 1, 8, 14, 0, tzinfo=UTC)
SETTLE_C = datetime(2024, 1, 15, 14, 0, tzinfo=UTC)


# ---------- shared test builders ------------------------------------------


def _make_bet(
    *,
    bet_id: str,
    match_id: str,
    stake: float,
    gross_pnl: float,
    commission: float,
    bankroll_after: float,
    settled_at: datetime = SETTLE_A,
    side: Side = Side.BACK,
    selection: Selection = Selection.HOME,
    price: float = 2.0,
    outcome: Selection = Selection.HOME,
) -> SettledBet:
    placed_at = settled_at - timedelta(hours=2)
    committed = stake if side is Side.BACK else (price - 1.0) * stake
    return SettledBet(
        bet_id=bet_id,
        match_id=match_id,
        selection=selection,
        side=side,
        price=price,
        stake=stake,
        placed_at=placed_at,
        committed_funds=committed,
        settled_at=settled_at,
        outcome=outcome,
        gross_pnl=gross_pnl,
        commission=commission,
        net_pnl=gross_pnl - commission,
        bankroll_after=bankroll_after,
    )


def _make_rejection(
    match_id: str = "R1",
    reason: RejectionReason = RejectionReason.OFF_SNAPSHOT,
) -> RejectedOrder:
    order = BetOrder(
        match_id=match_id,
        selection=Selection.HOME,
        side=Side.BACK,
        price=2.0,
        stake=5.0,
    )
    return RejectedOrder(
        order=order,
        rejected_at=T0,
        reason=reason,
        detail="test rejection",
    )


# ---------- group_ledger_by_match -----------------------------------------


class TestGroupLedgerByMatch:
    def test_empty_ledger(self) -> None:
        assert group_ledger_by_match([]) == []

    def test_single_bet(self) -> None:
        bet = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=10.0,
            commission=0.0,
            bankroll_after=1010.0,
        )
        groups = group_ledger_by_match([bet])
        assert len(groups) == 1
        assert groups[0][0] == "A"
        assert groups[0][1] == [bet]

    def test_two_contiguous_matches(self) -> None:
        a1 = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=10.0,
            commission=0.0,
            bankroll_after=1010.0,
            settled_at=SETTLE_A,
        )
        a2 = _make_bet(
            bet_id="A#0001",
            match_id="A",
            stake=5.0,
            gross_pnl=-5.0,
            commission=0.0,
            bankroll_after=1005.0,
            settled_at=SETTLE_A,
        )
        b1 = _make_bet(
            bet_id="B#0000",
            match_id="B",
            stake=20.0,
            gross_pnl=-20.0,
            commission=0.0,
            bankroll_after=985.0,
            settled_at=SETTLE_B,
        )
        groups = group_ledger_by_match([a1, a2, b1])
        assert [mid for mid, _ in groups] == ["A", "B"]
        assert groups[0][1] == [a1, a2]
        assert groups[1][1] == [b1]

    def test_non_contiguous_match_raises(self) -> None:
        a1 = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=10.0,
            commission=0.0,
            bankroll_after=1010.0,
            settled_at=SETTLE_A,
        )
        b1 = _make_bet(
            bet_id="B#0000",
            match_id="B",
            stake=10.0,
            gross_pnl=-10.0,
            commission=0.0,
            bankroll_after=1000.0,
            settled_at=SETTLE_B,
        )
        a2 = _make_bet(
            bet_id="A#0001",
            match_id="A",
            stake=10.0,
            gross_pnl=5.0,
            commission=0.0,
            bankroll_after=1005.0,
            settled_at=SETTLE_C,
        )
        with pytest.raises(ValueError, match="reappears non-contiguously"):
            group_ledger_by_match([a1, b1, a2])


# ---------- Pydantic model surface ----------------------------------------


class TestEquityPoint:
    def _valid(self, **overrides: object) -> EquityPoint:
        kwargs: dict[str, object] = {
            "timestamp": T0,
            "bankroll": 1000.0,
            "realised_pnl": 0.0,
            "cumulative_turnover": 0.0,
        }
        kwargs.update(overrides)
        return EquityPoint(**kwargs)  # type: ignore[arg-type]

    def test_valid_construction(self) -> None:
        point = self._valid()
        assert point.bankroll == 1000.0
        assert point.realised_pnl == 0.0

    def test_rejects_non_utc_timestamp(self) -> None:
        naive = datetime(2024, 1, 1, 12, 0)
        with pytest.raises(ValidationError):
            self._valid(timestamp=naive)

    def test_rejects_negative_turnover(self) -> None:
        with pytest.raises(ValidationError):
            self._valid(cumulative_turnover=-0.01)

    def test_frozen(self) -> None:
        point = self._valid()
        with pytest.raises(ValidationError):
            point.bankroll = 999.0


class TestSummaryMetrics:
    def _valid(self, **overrides: object) -> SummaryMetrics:
        kwargs: dict[str, object] = {
            "n_bets": 0,
            "n_rejections": 0,
            "turnover": 0.0,
            "gross_pnl": 0.0,
            "total_commission": 0.0,
            "net_pnl": 0.0,
        }
        kwargs.update(overrides)
        return SummaryMetrics(**kwargs)  # type: ignore[arg-type]

    def test_all_none_defaults(self) -> None:
        metrics = self._valid()
        assert metrics.hit_rate is None
        assert metrics.roi is None
        assert metrics.max_drawdown is None

    def test_rejects_negative_turnover(self) -> None:
        with pytest.raises(ValidationError):
            self._valid(turnover=-1.0)

    def test_rejects_negative_total_commission(self) -> None:
        with pytest.raises(ValidationError):
            self._valid(total_commission=-0.01)

    def test_rejects_hit_rate_outside_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            self._valid(hit_rate=1.5)
        with pytest.raises(ValidationError):
            self._valid(hit_rate=-0.1)

    def test_rejects_negative_max_drawdown(self) -> None:
        with pytest.raises(ValidationError):
            self._valid(max_drawdown=-0.01)

    def test_frozen(self) -> None:
        metrics = self._valid()
        with pytest.raises(ValidationError):
            metrics.n_bets = 99


# ---------- from_raw correctness -----------------------------------------


class TestFromRaw:
    def test_empty_ledger_empty_rejections(self) -> None:
        raw = RawBacktestOutput(ledger=(), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)

        assert result.ledger == ()
        assert result.rejections == ()
        assert len(result.equity_curve) == 1
        assert result.equity_curve[0] == EquityPoint(
            timestamp=T0,
            bankroll=1000.0,
            realised_pnl=0.0,
            cumulative_turnover=0.0,
        )
        assert result.summary_metrics.n_bets == 0
        assert result.summary_metrics.n_rejections == 0
        assert result.summary_metrics.turnover == 0.0
        assert result.summary_metrics.hit_rate is None
        assert result.summary_metrics.roi is None
        assert result.summary_metrics.max_drawdown is None

    def test_rejections_only(self) -> None:
        raw = RawBacktestOutput(
            ledger=(),
            rejections=(_make_rejection("R1"), _make_rejection("R2")),
        )
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)
        assert result.summary_metrics.n_bets == 0
        assert result.summary_metrics.n_rejections == 2
        assert len(result.equity_curve) == 1
        assert result.summary_metrics.hit_rate is None

    def test_single_winning_bet(self) -> None:
        bet = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=10.0,
            commission=0.5,
            bankroll_after=1009.5,
        )
        raw = RawBacktestOutput(ledger=(bet,), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)

        assert result.summary_metrics.n_bets == 1
        assert result.summary_metrics.turnover == 10.0
        assert result.summary_metrics.gross_pnl == 10.0
        assert result.summary_metrics.total_commission == 0.5
        assert result.summary_metrics.net_pnl == 9.5
        assert result.summary_metrics.hit_rate == 1.0
        assert result.summary_metrics.roi == pytest.approx(0.95)
        assert result.summary_metrics.max_drawdown == 0.0

        assert len(result.equity_curve) == 2
        assert result.equity_curve[1].bankroll == 1009.5
        assert result.equity_curve[1].realised_pnl == 9.5
        assert result.equity_curve[1].cumulative_turnover == 10.0
        assert result.equity_curve[1].timestamp == SETTLE_A

    def test_arb_case_one_point_per_match(self) -> None:
        # match A, two bets: +100 (commissioned) and -40 lay.
        # Running net: +97 after bet1, +57 after bet2. bankroll_after tracks.
        bet1 = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=100.0,
            gross_pnl=100.0,
            commission=3.0,
            bankroll_after=1097.0,
        )
        bet2 = _make_bet(
            bet_id="A#0001",
            match_id="A",
            stake=40.0,
            gross_pnl=-40.0,
            commission=0.0,
            bankroll_after=1057.0,
            side=Side.LAY,
            selection=Selection.AWAY,
        )
        raw = RawBacktestOutput(ledger=(bet1, bet2), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)

        assert len(result.equity_curve) == 2
        assert result.equity_curve[1].bankroll == 1057.0
        assert result.equity_curve[1].cumulative_turnover == 140.0
        assert result.summary_metrics.n_bets == 2
        assert result.summary_metrics.turnover == 140.0
        assert result.summary_metrics.hit_rate == 0.5
        assert result.summary_metrics.net_pnl == 57.0
        assert result.summary_metrics.roi == pytest.approx(57.0 / 140.0)

    def test_two_matches_mixed_outcomes(self) -> None:
        a = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=-10.0,
            commission=0.0,
            bankroll_after=990.0,
            settled_at=SETTLE_A,
        )
        b = _make_bet(
            bet_id="B#0000",
            match_id="B",
            stake=20.0,
            gross_pnl=20.0,
            commission=1.0,
            bankroll_after=1009.0,
            settled_at=SETTLE_B,
        )
        raw = RawBacktestOutput(ledger=(a, b), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)

        assert len(result.equity_curve) == 3
        assert result.equity_curve[0].timestamp == T0
        assert result.equity_curve[1].timestamp == SETTLE_A
        assert result.equity_curve[1].bankroll == 990.0
        assert result.equity_curve[2].timestamp == SETTLE_B
        assert result.equity_curve[2].bankroll == 1009.0
        assert result.summary_metrics.hit_rate == 0.5
        assert result.summary_metrics.turnover == 30.0
        assert result.summary_metrics.net_pnl == 9.0

    def test_gross_pnl_zero_is_not_a_win(self) -> None:
        # Lay edge case: laid selection wins with price exactly 1 + ε? Not
        # reachable with BetOrder.price > 1, but a push-like gross_pnl == 0
        # is still possible in weird commission configurations. Verify the
        # hit-rate rule documented in SummaryMetrics.
        bet = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=0.0,
            commission=0.0,
            bankroll_after=1000.0,
        )
        raw = RawBacktestOutput(ledger=(bet,), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)
        assert result.summary_metrics.hit_rate == 0.0

    def test_non_utc_t0_rejected(self) -> None:
        raw = RawBacktestOutput(ledger=(), rejections=())
        naive = datetime(2024, 1, 1, 12, 0)
        with pytest.raises(ValueError, match="UTC"):
            BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=naive)

    def test_determinism_same_inputs_byte_equal(self) -> None:
        bet1 = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=10.0,
            commission=0.5,
            bankroll_after=1009.5,
            settled_at=SETTLE_A,
        )
        bet2 = _make_bet(
            bet_id="B#0000",
            match_id="B",
            stake=20.0,
            gross_pnl=-20.0,
            commission=0.0,
            bankroll_after=989.5,
            settled_at=SETTLE_B,
        )
        raw = RawBacktestOutput(
            ledger=(bet1, bet2),
            rejections=(_make_rejection(),),
        )
        r1 = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)
        r2 = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)
        assert r1 == r2

    def test_bankroll_cross_check_catches_drift(self) -> None:
        # Tampered bankroll_after: running_net = +10 so expected = 1010,
        # but we set 1100. from_raw must flag this at the 4a/4b boundary.
        bet = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=10.0,
            commission=0.0,
            bankroll_after=1100.0,
        )
        raw = RawBacktestOutput(ledger=(bet,), rejections=())
        with pytest.raises(ValueError, match="exceeds tolerance"):
            BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)

    def test_bankroll_cross_check_tolerates_tiny_drift(self) -> None:
        # Drift well under 1e-6 is accepted: legitimate float accumulation.
        bet = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=10.0,
            commission=0.0,
            bankroll_after=1010.0 + 1e-9,
        )
        raw = RawBacktestOutput(ledger=(bet,), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)
        assert result.summary_metrics.n_bets == 1

    def test_bankroll_attribute_present_after_construction(self) -> None:
        raw = RawBacktestOutput(ledger=(), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)
        assert result.starting_bankroll == 1000.0
        assert result.t0 == T0


# ---------- max drawdown shape coverage ----------------------------------


class TestMaxDrawdown:
    def _result(self, *bets: SettledBet) -> BacktestResult:
        raw = RawBacktestOutput(ledger=tuple(bets), rejections=())
        return BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)

    def test_baseline_only_is_none(self) -> None:
        raw = RawBacktestOutput(ledger=(), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)
        assert result.summary_metrics.max_drawdown is None

    def test_monotonic_up_is_zero(self) -> None:
        a = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=10.0,
            commission=0.0,
            bankroll_after=1010.0,
            settled_at=SETTLE_A,
        )
        b = _make_bet(
            bet_id="B#0000",
            match_id="B",
            stake=10.0,
            gross_pnl=20.0,
            commission=0.0,
            bankroll_after=1030.0,
            settled_at=SETTLE_B,
        )
        assert self._result(a, b).summary_metrics.max_drawdown == 0.0

    def test_v_shape(self) -> None:
        # 1000 -> 980 -> 1010. Peak 1000, trough 980, drawdown 20.
        a = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=20.0,
            gross_pnl=-20.0,
            commission=0.0,
            bankroll_after=980.0,
            settled_at=SETTLE_A,
        )
        b = _make_bet(
            bet_id="B#0000",
            match_id="B",
            stake=30.0,
            gross_pnl=30.0,
            commission=0.0,
            bankroll_after=1010.0,
            settled_at=SETTLE_B,
        )
        assert self._result(a, b).summary_metrics.max_drawdown == 20.0

    def test_w_shape_second_trough_deeper(self) -> None:
        # 1000 -> 990 -> 1010 -> 980. New peak 1010 then deep drop to 980.
        a = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=-10.0,
            commission=0.0,
            bankroll_after=990.0,
            settled_at=SETTLE_A,
        )
        b = _make_bet(
            bet_id="B#0000",
            match_id="B",
            stake=20.0,
            gross_pnl=20.0,
            commission=0.0,
            bankroll_after=1010.0,
            settled_at=SETTLE_B,
        )
        c = _make_bet(
            bet_id="C#0000",
            match_id="C",
            stake=30.0,
            gross_pnl=-30.0,
            commission=0.0,
            bankroll_after=980.0,
            settled_at=SETTLE_C,
        )
        assert self._result(a, b, c).summary_metrics.max_drawdown == 30.0


# ---------- DataFrame helpers ---------------------------------------------


class TestDataFrames:
    def _nonempty_result(self) -> BacktestResult:
        bet = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=10.0,
            commission=0.5,
            bankroll_after=1009.5,
            side=Side.BACK,
            selection=Selection.HOME,
            outcome=Selection.HOME,
        )
        raw = RawBacktestOutput(
            ledger=(bet,),
            rejections=(_make_rejection("R1", RejectionReason.OFF_SNAPSHOT),),
        )
        return BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)

    def test_ledger_dataframe_columns(self) -> None:
        df = self._nonempty_result().ledger_dataframe()
        assert list(df.columns) == [
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
        assert len(df) == 1
        assert df.loc[0, "bet_id"] == "A#0000"
        assert df.loc[0, "selection"] == "home"
        assert df.loc[0, "side"] == "back"

    def test_ledger_dataframe_empty_has_columns(self) -> None:
        raw = RawBacktestOutput(ledger=(), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)
        df = result.ledger_dataframe()
        assert len(df) == 0
        assert "bet_id" in df.columns
        assert "bankroll_after" in df.columns

    def test_rejections_dataframe_flattens_order(self) -> None:
        df = self._nonempty_result().rejections_dataframe()
        assert list(df.columns) == [
            "rejected_at",
            "reason",
            "detail",
            "order_match_id",
            "order_selection",
            "order_side",
            "order_price",
            "order_stake",
        ]
        assert len(df) == 1
        assert df.loc[0, "reason"] == "off_snapshot"
        assert df.loc[0, "order_match_id"] == "R1"
        assert df.loc[0, "order_side"] == "back"
        assert df.loc[0, "order_price"] == 2.0
        assert df.loc[0, "order_stake"] == 5.0

    def test_rejections_dataframe_empty_has_columns(self) -> None:
        raw = RawBacktestOutput(ledger=(), rejections=())
        result = BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)
        df = result.rejections_dataframe()
        assert len(df) == 0
        assert "order_match_id" in df.columns

    def test_equity_curve_dataframe_columns(self) -> None:
        df = self._nonempty_result().equity_curve_dataframe()
        assert list(df.columns) == [
            "timestamp",
            "bankroll",
            "realised_pnl",
            "cumulative_turnover",
        ]
        assert len(df) == 2  # baseline + one match
        assert df.loc[0, "bankroll"] == 1000.0
        assert df.loc[1, "bankroll"] == 1009.5


# ---------- Parquet round-trip --------------------------------------------


class TestParquet:
    def _nonempty_result(self) -> BacktestResult:
        bet = _make_bet(
            bet_id="A#0000",
            match_id="A",
            stake=10.0,
            gross_pnl=10.0,
            commission=0.5,
            bankroll_after=1009.5,
        )
        raw = RawBacktestOutput(
            ledger=(bet,),
            rejections=(_make_rejection(),),
        )
        return BacktestResult.from_raw(raw, starting_bankroll=1000.0, t0=T0)

    def test_ledger_round_trip(self, tmp_path: Path) -> None:
        result = self._nonempty_result()
        path = tmp_path / "ledger.parquet"
        result.write_ledger_parquet(path)
        back = pd.read_parquet(path)
        assert len(back) == 1
        assert back.loc[0, "bet_id"] == "A#0000"
        assert back.loc[0, "stake"] == 10.0

    def test_rejections_round_trip(self, tmp_path: Path) -> None:
        result = self._nonempty_result()
        path = tmp_path / "rejections.parquet"
        result.write_rejections_parquet(path)
        back = pd.read_parquet(path)
        assert len(back) == 1
        assert back.loc[0, "reason"] == "off_snapshot"

    def test_equity_curve_round_trip(self, tmp_path: Path) -> None:
        result = self._nonempty_result()
        path = tmp_path / "equity.parquet"
        result.write_equity_curve_parquet(path)
        back = pd.read_parquet(path)
        assert len(back) == 2
        assert back.loc[0, "bankroll"] == 1000.0
        assert back.loc[1, "bankroll"] == 1009.5

    def test_write_all_parquet_creates_three_named_files(self, tmp_path: Path) -> None:
        result = self._nonempty_result()
        result.write_all_parquet(tmp_path)
        expected_files = {
            "ledger.parquet",
            "rejections.parquet",
            "equity_curve.parquet",
        }
        actual_files = {p.name for p in tmp_path.iterdir()}
        assert expected_files <= actual_files
