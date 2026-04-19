"""Kelly stake sizing for back and lay bets on decimal-odds exchange markets.

Two pure, stateless functions live here:

* :func:`back_kelly` -- Kelly-optimal **stake** fraction for a back bet.
* :func:`lay_kelly`  -- Kelly-optimal **liability** fraction for a lay bet.

Asymmetry in return semantics
-----------------------------

The single most bug-prone detail in exchange bet sizing, so it gets its own
section:

* ``back_kelly(p, b)`` returns the **stake as a fraction of bankroll**.
  Multiply by bankroll to get stake in currency units. A back bet's
  liability equals its stake (you lose the stake if the selection loses),
  so the Kelly result is directly the ``stake`` field on the emitted
  :class:`~betting_backtester.backtester.BetOrder`.
* ``lay_kelly(p, b)`` returns the **liability as a fraction of bankroll**.
  The stake sent to the backer (and therefore the ``stake`` field on a
  lay :class:`~betting_backtester.backtester.BetOrder`) is
  ``liability / (b - 1)``. The "stake" on a lay bet is what the backer
  puts up, not what the layer risks; the layer's risk is the liability,
  and that is what Kelly sizes.

Both functions clip to ``[0, 1]`` on non-edge or over-confident inputs. A
zero return means "no bet" -- the caller must check and skip rather than
pass zero as a stake to :class:`~betting_backtester.backtester.BetOrder`,
which rejects non-positive stakes at construction.

Derivations
-----------

**Back** at decimal odds ``b`` with our probability ``p`` of winning.
Bet fraction ``f`` of bankroll; win ``(b-1)*f`` with prob ``p``, lose
``f`` with prob ``1-p``.

* ``E[log(W')] = p * log(1 + (b-1)*f) + (1-p) * log(1 - f)``
* ``f* = (p*b - 1) / (b - 1)``

**Lay** at decimal odds ``b`` with our probability ``p`` that the laid
selection wins. Equivalent to backing the complement at effective odds
``b' = b/(b-1)`` with our probability ``1 - p``; the back-bet derivation
in those coordinates and a substitution back give:

* ``f* = 1 - p*b``   (liability fraction of bankroll)

Sanity anchor
-------------

At the market-implied probability ``p = 1/b`` both formulas return 0.
That is precisely the zero-edge point: the ``edge_threshold`` gate in
:class:`~betting_backtester.strategies.xg_poisson.XgPoissonStrategy`
enforces ``p`` is on the profitable side of ``1/b`` before either Kelly
function is called, so a positive Kelly return implies a positive edge
by construction. Fair coin at ``b = 2.0`` with ``p = 0.5`` is the
canonical hand-check: ``f_back = 0``, ``f_lay = 0``.
"""

from __future__ import annotations

import math


def back_kelly(probability: float, decimal_odds: float) -> float:
    """Kelly-optimal back stake as a fraction of bankroll.

    Parameters
    ----------
    probability:
        Our believed probability that the backed selection wins. Must
        be finite and in the open interval ``(0, 1)``.
    decimal_odds:
        Decimal odds at which the back is offered. Must be finite and
        strictly greater than 1.

    Returns
    -------
    float
        Stake as a fraction of bankroll, clipped to ``[0, 1]``. Returns
        0 when our probability does not exceed the market-implied
        probability (``p * b <= 1``); in that case the back is not
        positive-edge and Kelly advises no bet.
    """
    _validate_probability(probability)
    _validate_decimal_odds(decimal_odds)
    numerator = probability * decimal_odds - 1.0
    if numerator <= 0.0:
        return 0.0
    fraction = numerator / (decimal_odds - 1.0)
    # Mathematically f* < 1 given p < 1 and b > 1; clip defensively so
    # callers never see > 1 under any float-precision pathology.
    return min(1.0, fraction)


def lay_kelly(probability: float, decimal_odds: float) -> float:
    """Kelly-optimal lay liability as a fraction of bankroll.

    Returns the **liability** (amount at risk if the laid selection
    wins), not the stake offered to the backer. The stake field on a
    lay :class:`~betting_backtester.backtester.BetOrder` must be
    ``liability / (decimal_odds - 1)``; see the module docstring for
    the derivation.

    Parameters
    ----------
    probability:
        Our believed probability that the laid selection wins (the
        *event* probability, not its complement). Must be finite and
        in ``(0, 1)``.
    decimal_odds:
        Decimal odds at which the lay is offered. Must be finite and
        strictly greater than 1.

    Returns
    -------
    float
        Liability as a fraction of bankroll, clipped to ``[0, 1]``.
        Returns 0 when our probability does not undercut the
        market-implied probability (``p * b >= 1``); in that case the
        lay is not positive-edge and Kelly advises no bet.
    """
    _validate_probability(probability)
    _validate_decimal_odds(decimal_odds)
    fraction = 1.0 - probability * decimal_odds
    if fraction <= 0.0:
        return 0.0
    # f* = 1 - p*b < 1 whenever p > 0, which the validator enforces.
    return fraction


def _validate_probability(p: float) -> None:
    if not math.isfinite(p):
        raise ValueError(f"probability must be finite, got {p!r}")
    if not (0.0 < p < 1.0):
        raise ValueError(f"probability must be in (0, 1), got {p}")


def _validate_decimal_odds(b: float) -> None:
    if not math.isfinite(b):
        raise ValueError(f"decimal_odds must be finite, got {b!r}")
    if b <= 1.0:
        raise ValueError(f"decimal_odds must be > 1, got {b}")


__all__ = ["back_kelly", "lay_kelly"]
