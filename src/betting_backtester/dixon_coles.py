"""Dixon-Coles-lite goal-rate model for 1X2 outcome probabilities.

Module 9a. Fits per-team attack and defence ratings plus a single global
home-advantage parameter by weighted, L2-regularised MLE on independent
Poisson goal counts:

    lambda_home = exp(alpha_home - beta_away + gamma)
    lambda_away = exp(alpha_away - beta_home)

Training-match log-likelihoods are weighted by an exponential time-decay
factor relative to a caller-supplied ``anchor_time`` (typically the start
of the evaluation window). Inference bins a 7x7 Poisson score grid into
HOME, DRAW, AWAY.

Scope and deviations from full Dixon-Coles
------------------------------------------

* **No low-score correction.** The 1997 Dixon-Coles paper adjusts the
  joint distribution around (0,0), (0,1), (1,0), (1,1) to correct for
  the independence assumption at low scores. V1 stops at independent
  Poissons. The mass on those cells is consequently slightly off from a
  true Poisson-dependence model; the low-score bump is a future
  extension.
* **Time-decay weighting.** Each match's log-likelihood contribution is
  multiplied by ``exp(-decay_rate * days_before_anchor)`` where
  ``days_before_anchor = max(0, (anchor_time - settled_at).days)``.
  Default ``decay_rate = 0.0019`` is a ~1-year half-life
  (``ln(2) / 365 ~= 0.0019``).
* **L2 regularisation on ratings only.** A penalty
  ``l2_reg * (sum(alpha^2) + sum(beta^2))`` is added to the negative
  log-likelihood. ``gamma`` is deliberately not penalised: there is one
  of it, and it carries a meaningful non-zero prior (home advantage is
  ~0.3 log-goals in top-tier football, not 0).

Parameter identifiability
-------------------------

``lambda_home = exp(alpha_h - beta_a + gamma)`` only depends on
``alpha_h - beta_a``; adding any constant ``c`` to every ``alpha`` or
every ``beta`` leaves the likelihood unchanged. To pin the gauge, after
L-BFGS-B returns we subtract the mean of the attack ratings and the
mean of the defence ratings separately from each. L2 already pulls
each mean toward zero so the correction is tiny; doing it explicitly
makes the fitted parameters directly comparable across fits.

Convexity
---------

The negative log-likelihood is convex in ``(alpha, beta, gamma)``:
``exp`` is convex, applied to a linear form in the parameters, and the
``y * eta`` term is linear. L2 is strictly convex. The combined
objective therefore has a unique global minimum; L-BFGS-B starting from
zeros converges deterministically. No multi-start needed.

Inference: Poisson grid
-----------------------

:meth:`DixonColesModel.predict` computes ``P(home=i, away=j) =
pmf(i; lambda_home) * pmf(j; lambda_away)`` for
``i, j in {0, ..., _MAX_GOALS}`` (``_MAX_GOALS = 6``) and bins into
HOME (i > j), DRAW (i == j), AWAY (i < j). The 7x7 grid truncates tail
mass past 6-6; at typical football rates (lambda ~ 1-2) the truncated
mass is ~1e-4. The grid is normalised to sum to 1 before binning --
equivalent to conditioning on ``max(goals) <= 6`` -- so the returned
probabilities pass the sum-to-one validator on
:class:`SelectionProbabilities`. ``_MAX_GOALS = 6`` is hidden because
it is an implementation detail of the Poisson truncation, not a
strategy knob; if experience shows 6 is too tight, the constant moves
here, not into every caller.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import minimize

# Truncation point on the score grid. See module docstring.
_MAX_GOALS: int = 6

# Sum-to-one tolerance on :class:`SelectionProbabilities`. Matches the
# convention used by :class:`~betting_backtester.synthetic.TrueProbabilities`
# (1e-9) scaled up to absorb the grid normalisation and binning
# accumulations without masking genuine bugs.
_PROB_SUM_EPSILON: float = 1e-6


class TrainingMatch(BaseModel):
    """One settled fixture passed to :meth:`DixonColesModel.fit`.

    A thin glue record joining the event-stream
    :class:`~betting_backtester.models.MatchResult` (which lacks team
    names) with the team identities the strategy layer carries in its
    ``match_directory``. Constructed by
    :class:`~betting_backtester.strategies.xg_poisson.XgPoissonStrategy`,
    not by the model itself.
    """

    model_config = ConfigDict(frozen=True)

    home_team: str = Field(min_length=1, description="Home team identifier.")
    away_team: str = Field(min_length=1, description="Away team identifier.")
    home_goals: int = Field(ge=0, description="Final home score.")
    away_goals: int = Field(ge=0, description="Final away score.")
    settled_at: datetime = Field(
        description="Settlement timestamp, timezone-aware UTC."
    )

    @model_validator(mode="after")
    def _validate(self) -> TrainingMatch:
        if self.home_team == self.away_team:
            raise ValueError(
                f"home_team and away_team must differ "
                f"(got {self.home_team!r} for both)"
            )
        if self.settled_at.utcoffset() != timedelta(0):
            raise ValueError("settled_at must be UTC (offset 0)")
        return self


class SelectionProbabilities(BaseModel):
    """Model probabilities for the three 1X2 selections.

    Each probability lies in ``(0, 1)`` and the three sum to ``1``
    within :data:`_PROB_SUM_EPSILON`. Returned by
    :meth:`DixonColesModel.predict`; compared against market-implied
    probabilities by the strategy.
    """

    model_config = ConfigDict(frozen=True)

    home: float = Field(gt=0.0, lt=1.0, description="P(home win).")
    draw: float = Field(gt=0.0, lt=1.0, description="P(draw).")
    away: float = Field(gt=0.0, lt=1.0, description="P(away win).")

    @model_validator(mode="after")
    def _sum_to_one(self) -> SelectionProbabilities:
        total = self.home + self.draw + self.away
        if abs(total - 1.0) > _PROB_SUM_EPSILON:
            raise ValueError(
                f"probabilities must sum to 1 within {_PROB_SUM_EPSILON}, "
                f"got {total!r} (home={self.home}, draw={self.draw}, "
                f"away={self.away})"
            )
        return self


class DixonColesModel:
    """Dixon-Coles-lite per-team goal-rate model.

    Construction sets hyperparameters only; :meth:`fit` learns ratings
    from a sequence of :class:`TrainingMatch`; :meth:`predict` returns
    a :class:`SelectionProbabilities` for any ordered team pair whose
    teams both appear in the training data.

    Re-fittable: each :meth:`fit` call replaces the previous fit
    entirely. No incremental update.
    """

    def __init__(self, l2_reg: float = 0.001, decay_rate: float = 0.0019) -> None:
        """Validate hyperparameters and initialise unfitted state.

        Parameters
        ----------
        l2_reg:
            L2 penalty coefficient on attack and defence ratings. Must
            be finite and non-negative. ``0`` disables regularisation
            (fine for synthetic recovery tests; typically too loose for
            real data).
        decay_rate:
            Exponential time-decay coefficient, in units of
            ``per-day``. Must be finite and non-negative. ``0`` applies
            equal weight to every training match regardless of age.
        """
        if not math.isfinite(l2_reg):
            raise ValueError(f"l2_reg must be finite, got {l2_reg!r}")
        if l2_reg < 0.0:
            raise ValueError(f"l2_reg must be >= 0, got {l2_reg}")
        if not math.isfinite(decay_rate):
            raise ValueError(f"decay_rate must be finite, got {decay_rate!r}")
        if decay_rate < 0.0:
            raise ValueError(f"decay_rate must be >= 0, got {decay_rate}")

        self._l2_reg: float = float(l2_reg)
        self._decay_rate: float = float(decay_rate)

        self._team_index: dict[str, int] = {}
        self._attack: npt.NDArray[np.float64] | None = None
        self._defence: npt.NDArray[np.float64] | None = None
        self._home_advantage: float = 0.0
        self._fitted: bool = False

    @property
    def l2_reg(self) -> float:
        """L2 penalty coefficient on attack and defence ratings."""
        return self._l2_reg

    @property
    def decay_rate(self) -> float:
        """Exponential time-decay coefficient, per-day."""
        return self._decay_rate

    @property
    def home_advantage(self) -> float:
        """Fitted global home advantage ``gamma`` in log-goals. ``0`` until fit()."""
        return self._home_advantage

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has succeeded at least once on this instance."""
        return self._fitted

    def known_teams(self) -> frozenset[str]:
        """Teams with at least one training match. Empty until :meth:`fit`."""
        return frozenset(self._team_index.keys())

    def fit(
        self,
        match_results: Sequence[TrainingMatch],
        anchor_time: datetime,
    ) -> None:
        """Fit attack/defence ratings and home advantage by weighted MLE.

        Parameters
        ----------
        match_results:
            Training matches. Must be non-empty; raises
            :class:`ValueError` otherwise.
        anchor_time:
            Reference time for time-decay weighting. Each match's
            log-likelihood contribution is multiplied by
            ``exp(-decay_rate * max(0, (anchor_time - settled_at).days))``.
            Typically the start of the evaluation window.

        Raises
        ------
        ValueError
            ``match_results`` is empty, or ``anchor_time`` is not UTC.
        RuntimeError
            L-BFGS-B failed to converge. The convex objective should
            always converge from zero initial point; a failure here
            indicates either extremely degenerate data or a numerical
            pathology worth inspecting.
        """
        if len(match_results) == 0:
            raise ValueError("fit() requires at least one TrainingMatch")
        if anchor_time.utcoffset() != timedelta(0):
            raise ValueError("anchor_time must be UTC (offset 0)")

        teams: list[str] = sorted(
            {m.home_team for m in match_results}
            | {m.away_team for m in match_results}
        )
        team_index: dict[str, int] = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)
        n_matches = len(match_results)

        home_idx: npt.NDArray[np.int64] = np.asarray(
            [team_index[m.home_team] for m in match_results], dtype=np.int64
        )
        away_idx: npt.NDArray[np.int64] = np.asarray(
            [team_index[m.away_team] for m in match_results], dtype=np.int64
        )
        home_goals: npt.NDArray[np.float64] = np.asarray(
            [m.home_goals for m in match_results], dtype=np.float64
        )
        away_goals: npt.NDArray[np.float64] = np.asarray(
            [m.away_goals for m in match_results], dtype=np.float64
        )
        weights: npt.NDArray[np.float64] = np.asarray(
            [
                _decay_weight(self._decay_rate, anchor_time, m.settled_at)
                for m in match_results
            ],
            dtype=np.float64,
        )

        l2_reg = self._l2_reg

        def objective(x: npt.NDArray[np.float64]) -> float:
            alpha = x[:n_teams]
            beta = x[n_teams : 2 * n_teams]
            gamma = float(x[2 * n_teams])
            log_lambda_h = alpha[home_idx] - beta[away_idx] + gamma
            log_lambda_a = alpha[away_idx] - beta[home_idx]
            lambda_h = np.exp(log_lambda_h)
            lambda_a = np.exp(log_lambda_a)
            # Poisson log-likelihood up to a score-independent constant:
            # ell = g * log(lambda) - lambda - log(g!); the last term is
            # free in theta so we drop it.
            per_match_ll = (
                home_goals * log_lambda_h
                - lambda_h
                + away_goals * log_lambda_a
                - lambda_a
            )
            nll = -float(np.dot(weights, per_match_ll))
            penalty = l2_reg * float(
                np.dot(alpha, alpha) + np.dot(beta, beta)
            )
            return nll + penalty

        x0: npt.NDArray[np.float64] = np.zeros(2 * n_teams + 1, dtype=np.float64)
        result = minimize(objective, x0, method="L-BFGS-B")
        if not bool(result.success):
            raise RuntimeError(
                f"DixonColesModel MLE did not converge: {result.message!r} "
                f"(nit={result.nit}, n_matches={n_matches}, "
                f"n_teams={n_teams})"
            )

        x_hat: npt.NDArray[np.float64] = np.asarray(result.x, dtype=np.float64)
        alpha_raw = x_hat[:n_teams]
        beta_raw = x_hat[n_teams : 2 * n_teams]
        alpha_hat = alpha_raw - float(np.mean(alpha_raw))
        beta_hat = beta_raw - float(np.mean(beta_raw))
        gamma_hat = float(x_hat[2 * n_teams])

        self._team_index = team_index
        self._attack = alpha_hat
        self._defence = beta_hat
        self._home_advantage = gamma_hat
        self._fitted = True

    def predict(self, home_team: str, away_team: str) -> SelectionProbabilities:
        """Return 1X2 probabilities for one fixture via the Poisson grid.

        Parameters
        ----------
        home_team, away_team:
            Team identifiers. Both must be in :meth:`known_teams`;
            otherwise :class:`KeyError`. The strategy layer is
            responsible for filtering unseen teams before calling
            :meth:`predict` and for surfacing the skip to its user.

        Raises
        ------
        RuntimeError
            :meth:`fit` has not been called on this instance.
        ValueError
            ``home_team == away_team``.
        KeyError
            Either team is unknown to the fitted model.
        """
        if not self._fitted:
            raise RuntimeError("predict() called before fit()")
        if home_team == away_team:
            raise ValueError(
                f"home_team and away_team must differ (got {home_team!r})"
            )
        if home_team not in self._team_index:
            raise KeyError(f"unseen team: {home_team!r}")
        if away_team not in self._team_index:
            raise KeyError(f"unseen team: {away_team!r}")

        attack = self._attack
        defence = self._defence
        if attack is None or defence is None:
            # Invariant: ``_fitted=True`` iff both arrays are populated.
            # Belt-and-braces under ``python -O`` where plain ``assert``
            # would disappear.
            raise RuntimeError(
                "DixonColesModel internal invariant broken: is_fitted=True "
                "but _attack or _defence is None"
            )

        i_h = self._team_index[home_team]
        i_a = self._team_index[away_team]
        lambda_h = math.exp(
            float(attack[i_h]) - float(defence[i_a]) + self._home_advantage
        )
        lambda_a = math.exp(float(attack[i_a]) - float(defence[i_h]))
        return _probabilities_from_goal_rates(lambda_h, lambda_a)


def _decay_weight(
    decay_rate: float, anchor: datetime, settled_at: datetime
) -> float:
    """Time-decay weight ``exp(-decay_rate * max(0, days_before_anchor))``.

    Clipping at 0 means a match with ``settled_at > anchor`` (shouldn't
    happen under the walk-forward evaluator's discipline, but defensive)
    receives weight 1.0 rather than amplified weight.
    """
    days_before = max(0.0, (anchor - settled_at).total_seconds() / 86400.0)
    return math.exp(-decay_rate * days_before)


def _probabilities_from_goal_rates(
    lambda_home: float, lambda_away: float
) -> SelectionProbabilities:
    """Bin a truncated, renormalised Poisson score grid into HOME/DRAW/AWAY.

    ``grid[i, j] = pmf(i; lambda_home) * pmf(j; lambda_away)`` with
    ``i`` = home goals and ``j`` = away goals over
    ``{0, ..., _MAX_GOALS}``. The 7x7 grid is normalised to sum to 1
    (absorbing the omitted tail mass past 6-6), then binned via
    lower-triangle (``i > j`` -> HOME), diagonal (``i == j`` -> DRAW),
    upper-triangle (``i < j`` -> AWAY). ``np.trace`` / ``np.tril`` /
    ``np.triu`` keep the binning close to the mathematical prose.
    """
    pmf_home = _poisson_pmf_vector(lambda_home, _MAX_GOALS)
    pmf_away = _poisson_pmf_vector(lambda_away, _MAX_GOALS)
    grid = np.outer(pmf_home, pmf_away)
    grid /= float(grid.sum())

    draw_mass = float(np.trace(grid))
    home_mass = float(np.tril(grid, k=-1).sum())
    away_mass = float(np.triu(grid, k=1).sum())

    return SelectionProbabilities(
        home=home_mass, draw=draw_mass, away=away_mass
    )


def _poisson_pmf_vector(lam: float, max_k: int) -> npt.NDArray[np.float64]:
    """Return ``[P(k=0; lam), ..., P(k=max_k; lam)]``.

    Computed iteratively from ``pmf(0) = exp(-lam)`` via
    ``pmf(k+1) = pmf(k) * lam / (k+1)``. Avoids ``scipy.stats.poisson``
    to keep the computation directly inspectable and side-step one more
    scipy sub-module dependency.
    """
    pmf = np.empty(max_k + 1, dtype=np.float64)
    pmf[0] = math.exp(-lam)
    for k in range(max_k):
        pmf[k + 1] = pmf[k] * lam / (k + 1)
    return pmf


__all__ = [
    "DixonColesModel",
    "SelectionProbabilities",
    "TrainingMatch",
]
