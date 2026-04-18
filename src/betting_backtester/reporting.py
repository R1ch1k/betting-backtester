"""Bootstrap confidence interval on yield for a :class:`BacktestResult`.

The per-bet distribution of returns on an exchange-style ledger violates
every assumption a parametric CI would need: heavy right-skew (mostly
small losses, occasional large wins at long odds), non-iid rows (stakes
vary under Kelly-like sizing, and back/lay pairs in arbitrage markets
are perfectly coupled), and small-to-moderate sample sizes. A
non-parametric bootstrap is the honest choice; this module implements
it per-match, not per-bet, to preserve the correlation structure of
multi-bet markets.

**Per-match vs per-bet resampling.** Resampling individual rows treats
every bet as independent. For a back/lay arbitrage market that is
catastrophically wrong: the back bet's ``gross_pnl`` and the lay bet's
return are mechanically coupled to the same match outcome, so breaking
them apart in a bootstrap inflates variance and yields a CI that is
neither conservative nor realistic. Resampling match-groups keeps
coupled bets together. For single-bet-per-match strategies per-match
and per-bet are equivalent, so per-match is strictly more general and
there is no downside.
"""

from __future__ import annotations

import math
import random

from pydantic import BaseModel, ConfigDict, Field, model_validator

from betting_backtester.backtest_result import (
    BacktestResult,
    group_ledger_by_match,
)


class YieldCI(BaseModel):
    """Bootstrap confidence interval on yield (``net_pnl / turnover``).

    The bootstrap mean can fall outside the percentile bounds when the
    yield distribution is heavily skewed (the per-resample mean pulls
    toward large wins, while the percentile bounds sit on the ordered
    sample), so the only structural guarantee is ``lower <= upper``.
    Do not read a tight inequality into it.

    ``n_valid_resamples`` is the count of resamples that contributed
    to the CI, i.e. ``n_resamples`` minus any resamples skipped for
    zero turnover. Under current backtester invariants (stake is
    strictly positive, ledger is non-empty) zero-turnover resamples
    are unreachable and ``n_valid_resamples == n_resamples``.
    ``n_match_groups`` is the sample size of the bootstrap in its own
    units, i.e. the number of distinct match-groups in the source
    ledger.
    """

    model_config = ConfigDict(frozen=True)

    mean: float = Field(allow_inf_nan=False)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    confidence: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    n_resamples: int = Field(ge=1)
    n_valid_resamples: int = Field(
        ge=0,
        description="Resamples with non-zero turnover that contributed to the CI.",
    )
    n_match_groups: int = Field(
        ge=0,
        description="Number of distinct match-level groups in the source ledger.",
    )

    @model_validator(mode="after")
    def _validate(self) -> YieldCI:
        if self.lower > self.upper:
            raise ValueError(f"lower ({self.lower}) must be <= upper ({self.upper})")
        return self


def _type7_quantile(sorted_values: list[float], q: float) -> float:
    """Type-7 quantile with linear interpolation between order statistics.

    Equivalent to ``numpy.quantile(sorted_values, q, method='linear')``
    -- the R default and the convention most statistical texts use.
    ``sorted_values`` must be non-empty and in ascending order; ``q``
    must lie in ``[0, 1]``.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = (n - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def compute_yield_ci(
    result: BacktestResult,
    seed: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> YieldCI:
    """Bootstrap CI on yield, resampling over match-groups.

    Procedure:

    1. Group the ledger into contiguous match-groups
       (:func:`group_ledger_by_match` enforces the 4a contiguity
       invariant).
    2. Resample the group indices ``n_resamples`` times with
       replacement using ``random.Random(seed)``. Stdlib ``random`` is
       chosen over numpy for cross-platform determinism: numpy's
       default RNG stream has changed between major releases in the
       past, while ``random.Random`` is stable.
    3. For each resample, compute yield as
       ``sum(bet.net_pnl) / sum(bet.stake)`` across all bets in the
       drawn groups. Resamples with zero turnover are skipped and do
       not contribute to the CI; under current backtester invariants
       (stake > 0 on every bet, ledger non-empty) this branch is
       unreachable, but it is retained defensively.
    4. Report the mean of kept yields and the lower / upper percentile
       bounds at ``alpha = (1 - confidence) / 2`` using type-7 linear
       interpolation, equivalent to ``numpy.quantile(..., method=
       'linear')``.

    Determinism: given the same ``(result, seed, n_resamples,
    confidence)`` this function returns a byte-identical
    :class:`YieldCI`.

    Raises
    ------
    ValueError
        If the ledger is empty; if ``n_resamples < 1``; if
        ``confidence`` is not in ``(0, 1)``; or if every resample had
        zero turnover (unreachable under current invariants).
    """
    if not result.ledger:
        raise ValueError("cannot bootstrap yield on an empty ledger")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    groups = group_ledger_by_match(result.ledger)
    n_groups = len(groups)
    # Pre-compute per-group turnover and net P&L so the inner loop is
    # O(n_groups) per resample instead of O(n_bets). Computed once;
    # ``sum`` is deterministic given a fixed input order.
    group_stake = [sum(b.stake for b in bets) for _, bets in groups]
    group_net = [sum(b.net_pnl for b in bets) for _, bets in groups]

    rng = random.Random(seed)
    indices = range(n_groups)
    yields: list[float] = []
    for _ in range(n_resamples):
        picks = rng.choices(indices, k=n_groups)
        # math.fsum is exact (Shewchuk's algorithm) and order-independent,
        # so the 10k-sample bootstrap is byte-identical across platforms
        # and Python builds. Plain sum drifts by ~1 ulp * N in adversarial
        # accumulation orders, which would break the determinism test.
        turnover = math.fsum(group_stake[i] for i in picks)
        if turnover == 0.0:
            continue
        net = math.fsum(group_net[i] for i in picks)
        yields.append(net / turnover)

    if not yields:
        raise ValueError(
            "all bootstrap resamples had zero turnover; the ledger is "
            "degenerate in a way that should be unreachable given the "
            "backtester's stake>0 invariant"
        )

    yields.sort()
    mean = math.fsum(yields) / len(yields)
    alpha = (1.0 - confidence) / 2.0
    lower = _type7_quantile(yields, alpha)
    upper = _type7_quantile(yields, 1.0 - alpha)

    return YieldCI(
        mean=mean,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_resamples=n_resamples,
        n_valid_resamples=len(yields),
        n_match_groups=n_groups,
    )


__all__ = ["YieldCI", "compute_yield_ci"]
