"""Tests for :mod:`betting_backtester.kelly`.

Organisation:

* ``TestBackKelly`` -- hand-computed values, zero-edge returns, input
  validation.
* ``TestLayKelly`` -- hand-computed values, zero-edge returns, input
  validation.
* ``TestRelationships`` -- cross-function sanity anchors.
"""

from __future__ import annotations

import math

import pytest

from betting_backtester.kelly import back_kelly, lay_kelly


# ---------- back_kelly -----------------------------------------------------


class TestBackKelly:
    @pytest.mark.parametrize(
        ("probability", "decimal_odds", "expected"),
        [
            # (0.6 * 2 - 1) / (2 - 1) = 0.2
            (0.6, 2.0, 0.2),
            # (0.5 * 3 - 1) / (3 - 1) = 0.25
            (0.5, 3.0, 0.25),
            # (0.9 * 1.5 - 1) / (1.5 - 1) = 0.7
            (0.9, 1.5, 0.7),
            # (0.55 * 2 - 1) / 1 = 0.1
            (0.55, 2.0, 0.1),
            # (0.75 * 2.5 - 1) / 1.5 = 7/12
            (0.75, 2.5, 7.0 / 12.0),
        ],
    )
    def test_hand_computed_values(
        self, probability: float, decimal_odds: float, expected: float
    ) -> None:
        assert back_kelly(probability, decimal_odds) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("probability", "decimal_odds"),
        [
            (0.5, 2.0),   # p * b == 1 exactly
            (0.4, 2.0),   # p * b = 0.8 < 1
            (0.3, 3.0),   # p * b = 0.9 < 1
            (0.01, 1.5),  # deep negative edge
        ],
    )
    def test_zero_or_negative_edge_returns_zero(
        self, probability: float, decimal_odds: float
    ) -> None:
        assert back_kelly(probability, decimal_odds) == 0.0

    @pytest.mark.parametrize(
        "bad",
        [0.0, 1.0, -0.1, 1.1, math.nan, math.inf, -math.inf],
    )
    def test_invalid_probability_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            back_kelly(bad, 2.0)

    @pytest.mark.parametrize(
        "bad",
        [1.0, 0.5, 0.0, -1.0, math.nan, math.inf, -math.inf],
    )
    def test_invalid_decimal_odds_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            back_kelly(0.5, bad)


# ---------- lay_kelly ------------------------------------------------------


class TestLayKelly:
    @pytest.mark.parametrize(
        ("probability", "decimal_odds", "expected"),
        [
            # 1 - 0.4 * 2 = 0.2
            (0.4, 2.0, 0.2),
            # 1 - 0.3 * 3 = 0.1
            (0.3, 3.0, 0.1),
            # 1 - 0.1 * 5 = 0.5
            (0.1, 5.0, 0.5),
            # 1 - 0.2 * 4 = 0.2
            (0.2, 4.0, 0.2),
            # 1 - 0.45 * 2 = 0.1
            (0.45, 2.0, 0.1),
        ],
    )
    def test_hand_computed_values(
        self, probability: float, decimal_odds: float, expected: float
    ) -> None:
        assert lay_kelly(probability, decimal_odds) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("probability", "decimal_odds"),
        [
            (0.5, 2.0),   # p * b == 1 exactly
            (0.6, 2.0),   # p * b = 1.2 > 1 (wrong side)
            (0.34, 3.0),  # p * b ~= 1.02 > 1
            (0.99, 1.5),  # extreme wrong side
        ],
    )
    def test_zero_or_wrong_side_edge_returns_zero(
        self, probability: float, decimal_odds: float
    ) -> None:
        assert lay_kelly(probability, decimal_odds) == 0.0

    @pytest.mark.parametrize(
        "bad",
        [0.0, 1.0, -0.1, 1.1, math.nan, math.inf, -math.inf],
    )
    def test_invalid_probability_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            lay_kelly(bad, 2.0)

    @pytest.mark.parametrize(
        "bad",
        [1.0, 0.5, 0.0, -1.0, math.nan, math.inf, -math.inf],
    )
    def test_invalid_decimal_odds_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            lay_kelly(0.5, bad)


# ---------- cross-function relationships -----------------------------------


class TestRelationships:
    @pytest.mark.parametrize("decimal_odds", [1.5, 2.0, 3.0, 5.0, 10.0])
    def test_market_implied_probability_yields_zero_from_both(
        self, decimal_odds: float
    ) -> None:
        # At the market-implied probability there is zero edge; both
        # functions should return exactly 0.0 (not just approx-zero).
        probability = 1.0 / decimal_odds
        assert back_kelly(probability, decimal_odds) == 0.0
        assert lay_kelly(probability, decimal_odds) == 0.0

    def test_fair_coin_at_two_is_zero(self) -> None:
        # Canonical hand-check in the :mod:`kelly` module docstring.
        assert back_kelly(0.5, 2.0) == 0.0
        assert lay_kelly(0.5, 2.0) == 0.0

    @pytest.mark.parametrize(
        ("probability", "decimal_odds"),
        [
            (0.6, 2.0),   # back-side edge
            (0.4, 2.0),   # lay-side edge
            (0.3, 3.0),   # lay-side edge
            (0.4, 3.0),   # back-side edge
            (0.1, 5.0),   # lay-side edge
            (0.25, 5.0),  # back-side edge
        ],
    )
    def test_at_most_one_side_has_positive_edge(
        self, probability: float, decimal_odds: float
    ) -> None:
        # For any (p, b) with p != 1/b, exactly one Kelly function
        # returns > 0. At p == 1/b both return 0 (tested above).
        back = back_kelly(probability, decimal_odds)
        lay = lay_kelly(probability, decimal_odds)
        assert (back > 0.0) != (lay > 0.0), (
            f"p={probability}, b={decimal_odds}: back={back}, lay={lay}"
        )
