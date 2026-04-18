"""Tests for the commission module.

Covers ``SettledBetLine`` and ``CommissionBreakdown`` validation (including
the self-validating invariants on breakdown outputs), ``NetWinningsCommission``
constructor rejections, the full rule surface (all-losers, break-even,
single winner, multi-winner pro-rata, mixed winners/losers),
the arbitrage aggregation regression guard for module 10, edge cases (empty
input, duplicate bet ids, rate of 0), deterministic and seeded-randomised
invariants (attribution-sums-to-total, non-negativity, order-independence),
and structural conformance to the :class:`CommissionModel` protocol.
"""

from __future__ import annotations

import math
import random

import pytest
from pydantic import ValidationError

from betting_backtester.commission import (
    CommissionBreakdown,
    CommissionModel,
    NetWinningsCommission,
    SettledBetLine,
)


def _random_bets(rng: random.Random, n: int) -> list[SettledBetLine]:
    """Produce a deterministic batch of SettledBetLine instances.

    Gross P&L is drawn signed so roughly half of any large sample loses;
    stakes are positive. The RNG is passed in so each call advances shared
    state predictably under a fixed seed.
    """
    bets: list[SettledBetLine] = []
    for i in range(n):
        bets.append(
            SettledBetLine(
                bet_id=f"b-{i}",
                stake=rng.uniform(1.0, 200.0),
                gross_pnl=rng.uniform(-500.0, 500.0),
            )
        )
    return bets


class TestSettledBetLine:
    def test_valid_construction(self) -> None:
        line = SettledBetLine(bet_id="x", stake=10.0, gross_pnl=-5.0)
        assert line.bet_id == "x"
        assert line.stake == 10.0
        assert line.gross_pnl == -5.0

    def test_rejects_empty_bet_id(self) -> None:
        with pytest.raises(ValidationError):
            SettledBetLine(bet_id="", stake=10.0, gross_pnl=0.0)

    @pytest.mark.parametrize("bad", [0.0, -0.01, -100.0])
    def test_rejects_non_positive_stake(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            SettledBetLine(bet_id="x", stake=bad, gross_pnl=0.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_stake(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            SettledBetLine(bet_id="x", stake=bad, gross_pnl=0.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_gross_pnl(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            SettledBetLine(bet_id="x", stake=10.0, gross_pnl=bad)

    def test_allows_negative_gross_pnl(self) -> None:
        # Losers must be representable.
        line = SettledBetLine(bet_id="x", stake=10.0, gross_pnl=-10.0)
        assert line.gross_pnl == -10.0

    def test_is_frozen(self) -> None:
        line = SettledBetLine(bet_id="x", stake=10.0, gross_pnl=0.0)
        with pytest.raises(ValidationError):
            line.bet_id = "y"


class TestCommissionBreakdownValidator:
    def test_valid_construction(self) -> None:
        breakdown = CommissionBreakdown(total=3.0, per_bet={"a": 3.0, "b": 0.0})
        assert breakdown.total == 3.0
        assert breakdown.per_bet == {"a": 3.0, "b": 0.0}

    def test_zero_total_with_zero_attributions_ok(self) -> None:
        CommissionBreakdown(total=0.0, per_bet={"a": 0.0, "b": 0.0})

    def test_zero_total_with_empty_per_bet_ok(self) -> None:
        CommissionBreakdown(total=0.0, per_bet={})

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_per_bet_value(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="must be finite"):
            CommissionBreakdown(total=0.0, per_bet={"a": bad})

    def test_rejects_negative_per_bet_value(self) -> None:
        with pytest.raises(ValidationError, match="must be >= 0"):
            CommissionBreakdown(total=0.0, per_bet={"a": -0.5, "b": 0.5})

    def test_rejects_negative_total(self) -> None:
        with pytest.raises(ValidationError):
            CommissionBreakdown(total=-0.1, per_bet={})

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_total(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            CommissionBreakdown(total=bad, per_bet={})

    def test_rejects_positive_total_with_empty_per_bet(self) -> None:
        with pytest.raises(ValidationError, match="per_bet is empty"):
            CommissionBreakdown(total=1.0, per_bet={})

    def test_rejects_attribution_sum_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="sums to"):
            CommissionBreakdown(total=10.0, per_bet={"a": 5.0, "b": 3.0})

    def test_tolerates_tiny_float_drift(self) -> None:
        # A 1e-10 gap is well inside the 1e-9 invariant tolerance.
        CommissionBreakdown(total=1.0, per_bet={"a": 1.0 - 1e-10})

    def test_rejects_drift_beyond_tolerance(self) -> None:
        with pytest.raises(ValidationError, match="sums to"):
            CommissionBreakdown(total=1.0, per_bet={"a": 1.0 - 1e-6})

    def test_is_frozen(self) -> None:
        breakdown = CommissionBreakdown(total=0.0, per_bet={})
        with pytest.raises(ValidationError):
            breakdown.total = 1.0

    def test_per_bet_is_immutable(self) -> None:
        # ``frozen=True`` only blocks field reassignment; without the
        # MappingProxyType wrapper, ``breakdown.per_bet["a"] = 999`` would
        # silently violate the sum-to-total invariant. Lock the immutability
        # in with an explicit test.
        breakdown = CommissionBreakdown(total=3.0, per_bet={"a": 3.0, "b": 0.0})
        with pytest.raises(TypeError):
            breakdown.per_bet["a"] = 999.0  # type: ignore[index]

    def test_per_bet_is_isolated_from_caller_dict(self) -> None:
        # The AfterValidator copies the input via ``dict(...)`` before
        # wrapping, so later mutations to the caller's original dict must
        # not affect the stored mapping.
        source: dict[str, float] = {"a": 3.0, "b": 0.0}
        breakdown = CommissionBreakdown(total=3.0, per_bet=source)
        source["a"] = 999.0
        source["c"] = 42.0
        assert dict(breakdown.per_bet) == {"a": 3.0, "b": 0.0}


class TestNetWinningsCommissionConstructor:
    def test_default_rate(self) -> None:
        assert NetWinningsCommission().rate == 0.05

    def test_rate_is_readable(self) -> None:
        assert NetWinningsCommission(rate=0.02).rate == 0.02

    @pytest.mark.parametrize("bad", [-0.0001, -0.5, -1.0])
    def test_rejects_negative_rate(self, bad: float) -> None:
        with pytest.raises(ValueError, match="rate must be >= 0"):
            NetWinningsCommission(rate=bad)

    @pytest.mark.parametrize("bad", [1.0001, 1.5, 2.0])
    def test_rejects_rate_above_one(self, bad: float) -> None:
        with pytest.raises(ValueError, match="rate must be <= 1"):
            NetWinningsCommission(rate=bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_rate(self, bad: float) -> None:
        with pytest.raises(ValueError, match="rate must be finite"):
            NetWinningsCommission(rate=bad)

    def test_rate_zero_allowed(self) -> None:
        # Needed for zero-commission testing scenarios.
        NetWinningsCommission(rate=0.0)

    def test_rate_one_allowed(self) -> None:
        # 100% is permitted at the boundary even if unrealistic in practice.
        NetWinningsCommission(rate=1.0)


class TestCommissionForMarket:
    def test_all_losers_produces_zero(self) -> None:
        model = NetWinningsCommission(rate=0.05)
        bets = [
            SettledBetLine(bet_id="a", stake=10.0, gross_pnl=-20.0),
            SettledBetLine(bet_id="b", stake=10.0, gross_pnl=-10.0),
        ]
        result = model.commission_for_market(bets)
        assert result.total == 0.0
        assert result.per_bet == {"a": 0.0, "b": 0.0}

    def test_break_even_produces_zero(self) -> None:
        model = NetWinningsCommission(rate=0.05)
        bets = [
            SettledBetLine(bet_id="a", stake=10.0, gross_pnl=50.0),
            SettledBetLine(bet_id="b", stake=10.0, gross_pnl=-50.0),
        ]
        result = model.commission_for_market(bets)
        assert result.total == 0.0
        assert result.per_bet == {"a": 0.0, "b": 0.0}

    def test_single_winner(self) -> None:
        model = NetWinningsCommission(rate=0.05)
        bets = [SettledBetLine(bet_id="w", stake=10.0, gross_pnl=200.0)]
        result = model.commission_for_market(bets)
        assert result.total == pytest.approx(10.0)
        assert result.per_bet == {"w": pytest.approx(10.0)}

    def test_multiple_winners_split_pro_rata(self) -> None:
        model = NetWinningsCommission(rate=0.10)
        bets = [
            SettledBetLine(bet_id="a", stake=10.0, gross_pnl=60.0),
            SettledBetLine(bet_id="b", stake=10.0, gross_pnl=30.0),
            SettledBetLine(bet_id="c", stake=10.0, gross_pnl=10.0),
        ]
        # net = 100, total = 10; attribution pro-rata to gross.
        result = model.commission_for_market(bets)
        assert result.total == pytest.approx(10.0)
        assert result.per_bet["a"] == pytest.approx(6.0)
        assert result.per_bet["b"] == pytest.approx(3.0)
        assert result.per_bet["c"] == pytest.approx(1.0)

    def test_loser_gets_zero_when_net_positive(self) -> None:
        model = NetWinningsCommission(rate=0.05)
        bets = [
            SettledBetLine(bet_id="w", stake=10.0, gross_pnl=100.0),
            SettledBetLine(bet_id="l", stake=10.0, gross_pnl=-20.0),
        ]
        result = model.commission_for_market(bets)
        assert result.per_bet["l"] == 0.0

    def test_break_even_bet_in_positive_market_gets_zero(self) -> None:
        # A gross_pnl of exactly 0.0 is neither a winner nor a loser; it
        # must receive zero commission even when the market is in profit.
        model = NetWinningsCommission(rate=0.05)
        bets = [
            SettledBetLine(bet_id="w", stake=10.0, gross_pnl=100.0),
            SettledBetLine(bet_id="z", stake=10.0, gross_pnl=0.0),
        ]
        result = model.commission_for_market(bets)
        assert result.total == pytest.approx(5.0)
        assert result.per_bet["w"] == pytest.approx(5.0)
        assert result.per_bet["z"] == 0.0

    def test_arbitrage_commission_is_on_net_not_per_bet(self) -> None:
        # Module 10's arbitrage strategy depends on this exact behaviour:
        # a back + lay pair on the SAME market must be charged on NET
        # winnings, not on the winning leg in isolation.
        model = NetWinningsCommission(rate=0.05)
        back = SettledBetLine(bet_id="back", stake=100.0, gross_pnl=100.0)
        lay = SettledBetLine(bet_id="lay", stake=100.0, gross_pnl=-40.0)
        result = model.commission_for_market([back, lay])
        # Net = 60, total = 0.05 * 60 = 3.0. NOT 0.05 * 100 = 5.0.
        assert result.total == pytest.approx(3.0)
        assert result.per_bet["back"] == pytest.approx(3.0)
        assert result.per_bet["lay"] == 0.0
        # Explicit regression lock against the per-bet-in-isolation mistake.
        assert result.total != pytest.approx(5.0)

    def test_rate_zero_produces_zero_total_even_with_winner(self) -> None:
        model = NetWinningsCommission(rate=0.0)
        bets = [
            SettledBetLine(bet_id="a", stake=10.0, gross_pnl=100.0),
            SettledBetLine(bet_id="b", stake=10.0, gross_pnl=-30.0),
        ]
        result = model.commission_for_market(bets)
        assert result.total == 0.0
        assert result.per_bet == {"a": 0.0, "b": 0.0}

    def test_rate_one_siphons_entire_net(self) -> None:
        model = NetWinningsCommission(rate=1.0)
        bets = [
            SettledBetLine(bet_id="a", stake=10.0, gross_pnl=100.0),
            SettledBetLine(bet_id="b", stake=10.0, gross_pnl=-40.0),
        ]
        result = model.commission_for_market(bets)
        assert result.total == pytest.approx(60.0)
        assert result.per_bet["a"] == pytest.approx(60.0)
        assert result.per_bet["b"] == 0.0

    def test_empty_bet_list_returns_silent_zero(self) -> None:
        # Silent zero on empty input is an explicit design decision:
        # "no silent failures" targets swallowed errors, not handled
        # edge-case inputs. Documented here so any future flip to "raise"
        # is a conscious change.
        model = NetWinningsCommission(rate=0.05)
        result = model.commission_for_market([])
        assert result.total == 0.0
        assert result.per_bet == {}

    def test_rejects_duplicate_bet_id(self) -> None:
        model = NetWinningsCommission(rate=0.05)
        bets = [
            SettledBetLine(bet_id="dup", stake=10.0, gross_pnl=50.0),
            SettledBetLine(bet_id="dup", stake=10.0, gross_pnl=-20.0),
        ]
        with pytest.raises(ValueError, match="duplicate bet_id"):
            model.commission_for_market(bets)

    def test_every_input_id_appears_in_per_bet_output(self) -> None:
        # Attribution must cover every bet by id, not just the winners.
        model = NetWinningsCommission(rate=0.05)
        bets = [
            SettledBetLine(bet_id="w1", stake=10.0, gross_pnl=50.0),
            SettledBetLine(bet_id="w2", stake=10.0, gross_pnl=30.0),
            SettledBetLine(bet_id="l1", stake=10.0, gross_pnl=-20.0),
            SettledBetLine(bet_id="z1", stake=10.0, gross_pnl=0.0),
        ]
        result = model.commission_for_market(bets)
        assert set(result.per_bet.keys()) == {"w1", "w2", "l1", "z1"}


class TestInvariants:
    @pytest.mark.parametrize("rate", [0.0, 0.02, 0.05, 0.2, 1.0])
    def test_attribution_sums_to_total_randomised(self, rate: float) -> None:
        # Seeded random inputs; no hypothesis dependency. The
        # CommissionBreakdown validator also enforces this sum-invariant,
        # so a violation would surface as ValidationError during the call
        # rather than reaching the assert; the explicit assertion here
        # documents the contract at the test level.
        rng = random.Random(12345)
        model = NetWinningsCommission(rate=rate)
        for _ in range(100):
            bets = _random_bets(rng, n=rng.randint(1, 20))
            result = model.commission_for_market(bets)
            assert math.fsum(result.per_bet.values()) == pytest.approx(
                result.total, abs=1e-9
            )

    def test_total_non_negative_randomised(self) -> None:
        rng = random.Random(54321)
        model = NetWinningsCommission(rate=0.05)
        for _ in range(200):
            bets = _random_bets(rng, n=rng.randint(1, 20))
            result = model.commission_for_market(bets)
            assert result.total >= 0.0
            assert all(v >= 0.0 for v in result.per_bet.values())

    def test_order_independence_deterministic(self) -> None:
        # Two specific orderings of the same bet set produce equal output.
        # Paired with the randomised property below as belt-and-braces.
        model = NetWinningsCommission(rate=0.05)
        a = SettledBetLine(bet_id="a", stake=10.0, gross_pnl=80.0)
        b = SettledBetLine(bet_id="b", stake=10.0, gross_pnl=-30.0)
        c = SettledBetLine(bet_id="c", stake=10.0, gross_pnl=20.0)
        r1 = model.commission_for_market([a, b, c])
        r2 = model.commission_for_market([c, b, a])
        assert r1.total == pytest.approx(r2.total)
        assert r1.per_bet.keys() == r2.per_bet.keys()
        for key in r1.per_bet:
            assert r1.per_bet[key] == pytest.approx(r2.per_bet[key])

    def test_order_independence_randomised(self) -> None:
        rng = random.Random(99999)
        model = NetWinningsCommission(rate=0.05)
        for _ in range(50):
            bets = _random_bets(rng, n=rng.randint(2, 15))
            shuffled = list(bets)
            rng.shuffle(shuffled)
            r_orig = model.commission_for_market(bets)
            r_shuf = model.commission_for_market(shuffled)
            assert r_orig.total == pytest.approx(r_shuf.total, abs=1e-9)
            assert r_orig.per_bet.keys() == r_shuf.per_bet.keys()
            for key in r_orig.per_bet:
                assert r_orig.per_bet[key] == pytest.approx(
                    r_shuf.per_bet[key], abs=1e-9
                )


class TestCommissionModelProtocolConformance:
    def test_is_instance_of_commission_model(self) -> None:
        # Structural check via runtime_checkable Protocol, parallels the
        # EventSource conformance test in test_synthetic.py.
        assert isinstance(NetWinningsCommission(), CommissionModel)
