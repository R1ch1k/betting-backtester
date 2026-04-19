"""Tests for :mod:`betting_backtester.dixon_coles`.

Organisation:

* ``TestTrainingMatch`` -- input validation on the Pydantic glue record.
* ``TestSelectionProbabilities`` -- sum-to-one and per-field validators.
* ``TestDixonColesModelConstruction`` -- hyperparameter validation and
  initial unfitted state.
* ``TestFit`` -- fit-time input validation and refit semantics.
* ``TestPredict`` -- unseen team / degenerate input handling.
* ``TestSyntheticRecovery`` -- fit on matches sampled from known
  ratings, then verify the *predictions* match a ground-truth Poisson
  grid within tolerance. Avoids introspecting fitted ratings directly;
  the model is defined by what :meth:`predict` returns, and the test
  exercises that interface.
* ``TestL2Regularisation`` -- on thin data, an L2 penalty visibly
  shrinks predictions toward a common distribution (lower variance
  across team pairs).
* ``TestTimeDecay`` -- with a regime change in the training stream,
  a positive ``decay_rate`` pulls predictions toward the recent phase.
* ``TestDeterminism`` -- identical inputs yield byte-identical
  prediction sequences across two fresh models.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from betting_backtester.dixon_coles import (
    DixonColesModel,
    SelectionProbabilities,
    TrainingMatch,
)

UTC = timezone.utc

# Truncation point in the model's Poisson grid. Kept private there; we
# duplicate it here so the ground-truth predictions below use the same
# grid as the model.
_MAX_GOALS: int = 6


# ---------- helpers --------------------------------------------------------


def _training_match(
    home: str,
    away: str,
    hg: int,
    ag: int,
    ts: datetime,
) -> TrainingMatch:
    return TrainingMatch(
        home_team=home,
        away_team=away,
        home_goals=hg,
        away_goals=ag,
        settled_at=ts,
    )


def _truth_prediction(
    alpha_h: float,
    alpha_a: float,
    beta_h: float,
    beta_a: float,
    gamma: float,
) -> tuple[float, float, float]:
    """Ground-truth (home, draw, away) probabilities for known ratings.

    Uses the exact same truncated, renormalised 7x7 Poisson grid as
    :mod:`betting_backtester.dixon_coles`. Reimplemented here so the
    test does not depend on the module's private helpers.
    """
    lam_h = math.exp(alpha_h - beta_a + gamma)
    lam_a = math.exp(alpha_a - beta_h)
    pmf_h = [math.exp(-lam_h)]
    for k in range(_MAX_GOALS):
        pmf_h.append(pmf_h[-1] * lam_h / (k + 1))
    pmf_a = [math.exp(-lam_a)]
    for k in range(_MAX_GOALS):
        pmf_a.append(pmf_a[-1] * lam_a / (k + 1))
    total = sum(
        pmf_h[i] * pmf_a[j]
        for i in range(_MAX_GOALS + 1)
        for j in range(_MAX_GOALS + 1)
    )
    home = sum(
        pmf_h[i] * pmf_a[j]
        for i in range(_MAX_GOALS + 1)
        for j in range(_MAX_GOALS + 1)
        if i > j
    ) / total
    draw = sum(pmf_h[i] * pmf_a[i] for i in range(_MAX_GOALS + 1)) / total
    away = sum(
        pmf_h[i] * pmf_a[j]
        for i in range(_MAX_GOALS + 1)
        for j in range(_MAX_GOALS + 1)
        if i < j
    ) / total
    return home, draw, away


# ---------- TrainingMatch --------------------------------------------------


class TestTrainingMatch:
    def test_valid_construction(self) -> None:
        m = _training_match("A", "B", 2, 1, datetime(2024, 1, 1, tzinfo=UTC))
        assert m.home_team == "A"
        assert m.away_team == "B"
        assert m.home_goals == 2
        assert m.away_goals == 1

    def test_home_equals_away_raises(self) -> None:
        with pytest.raises(ValidationError, match="differ"):
            _training_match("A", "A", 1, 0, datetime(2024, 1, 1, tzinfo=UTC))

    def test_empty_home_team_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrainingMatch(
                home_team="",
                away_team="B",
                home_goals=1,
                away_goals=0,
                settled_at=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_negative_goals_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrainingMatch(
                home_team="A",
                away_team="B",
                home_goals=-1,
                away_goals=0,
                settled_at=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_non_utc_timestamp_raises(self) -> None:
        plus_two = timezone(timedelta(hours=2))
        with pytest.raises(ValidationError, match="UTC"):
            _training_match("A", "B", 1, 0, datetime(2024, 1, 1, tzinfo=plus_two))

    def test_frozen(self) -> None:
        m = _training_match("A", "B", 1, 0, datetime(2024, 1, 1, tzinfo=UTC))
        with pytest.raises(ValidationError):
            m.home_team = "C"  # type: ignore[misc]


# ---------- SelectionProbabilities ----------------------------------------


class TestSelectionProbabilities:
    def test_valid_sum_to_one(self) -> None:
        p = SelectionProbabilities(home=0.5, draw=0.3, away=0.2)
        assert p.home + p.draw + p.away == pytest.approx(1.0)

    def test_sum_too_far_from_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1"):
            SelectionProbabilities(home=0.4, draw=0.3, away=0.2)

    def test_zero_probability_raises(self) -> None:
        with pytest.raises(ValidationError):
            SelectionProbabilities(home=0.0, draw=0.5, away=0.5)

    def test_one_probability_raises(self) -> None:
        with pytest.raises(ValidationError):
            SelectionProbabilities(home=1.0, draw=0.0, away=0.0)


# ---------- DixonColesModel construction ----------------------------------


class TestDixonColesModelConstruction:
    def test_default_hyperparameters(self) -> None:
        m = DixonColesModel()
        assert m.l2_reg == 0.001
        assert m.decay_rate == 0.0019

    def test_initial_state_is_unfitted(self) -> None:
        m = DixonColesModel()
        assert m.is_fitted is False
        assert m.home_advantage == 0.0
        assert m.known_teams() == frozenset()

    @pytest.mark.parametrize("bad", [-0.01, math.nan, math.inf, -math.inf])
    def test_invalid_l2_reg_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            DixonColesModel(l2_reg=bad)

    @pytest.mark.parametrize("bad", [-0.01, math.nan, math.inf, -math.inf])
    def test_invalid_decay_rate_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            DixonColesModel(decay_rate=bad)


# ---------- fit -----------------------------------------------------------


class TestFit:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            DixonColesModel().fit([], anchor_time=datetime(2024, 1, 1, tzinfo=UTC))

    def test_non_utc_anchor_raises(self) -> None:
        model = DixonColesModel()
        m = _training_match("A", "B", 1, 0, datetime(2024, 1, 1, tzinfo=UTC))
        plus_two = timezone(timedelta(hours=2))
        with pytest.raises(ValueError, match="UTC"):
            model.fit([m], anchor_time=datetime(2024, 1, 2, tzinfo=plus_two))

    def test_is_fitted_flips_after_successful_fit(self) -> None:
        model = DixonColesModel()
        matches = [
            _training_match("A", "B", 2, 1, datetime(2024, 1, 1, tzinfo=UTC)),
            _training_match("B", "A", 1, 2, datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        model.fit(matches, anchor_time=datetime(2024, 1, 3, tzinfo=UTC))
        assert model.is_fitted is True
        assert model.known_teams() == frozenset({"A", "B"})

    def test_refit_replaces_previous_state(self) -> None:
        model = DixonColesModel()
        first = [_training_match("A", "B", 1, 0, datetime(2024, 1, 1, tzinfo=UTC))]
        model.fit(first, anchor_time=datetime(2024, 1, 2, tzinfo=UTC))
        second = [_training_match("X", "Y", 2, 1, datetime(2024, 2, 1, tzinfo=UTC))]
        model.fit(second, anchor_time=datetime(2024, 2, 2, tzinfo=UTC))
        assert model.known_teams() == frozenset({"X", "Y"})


# ---------- predict -------------------------------------------------------


class TestPredict:
    @pytest.fixture
    def fitted_model(self) -> DixonColesModel:
        model = DixonColesModel()
        matches = [
            _training_match("A", "B", 2, 1, datetime(2024, 1, 1, tzinfo=UTC)),
            _training_match("B", "A", 1, 2, datetime(2024, 1, 2, tzinfo=UTC)),
            _training_match("A", "C", 3, 0, datetime(2024, 1, 3, tzinfo=UTC)),
            _training_match("C", "A", 0, 2, datetime(2024, 1, 4, tzinfo=UTC)),
            _training_match("B", "C", 1, 1, datetime(2024, 1, 5, tzinfo=UTC)),
            _training_match("C", "B", 2, 0, datetime(2024, 1, 6, tzinfo=UTC)),
        ]
        model.fit(matches, anchor_time=datetime(2024, 1, 7, tzinfo=UTC))
        return model

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before fit"):
            DixonColesModel().predict("A", "B")

    def test_predict_same_team_raises(
        self, fitted_model: DixonColesModel
    ) -> None:
        with pytest.raises(ValueError, match="differ"):
            fitted_model.predict("A", "A")

    def test_predict_unseen_home_team_raises(
        self, fitted_model: DixonColesModel
    ) -> None:
        with pytest.raises(KeyError):
            fitted_model.predict("UNKNOWN", "A")

    def test_predict_unseen_away_team_raises(
        self, fitted_model: DixonColesModel
    ) -> None:
        with pytest.raises(KeyError):
            fitted_model.predict("A", "UNKNOWN")

    def test_predict_returns_valid_probabilities(
        self, fitted_model: DixonColesModel
    ) -> None:
        probs = fitted_model.predict("A", "B")
        assert isinstance(probs, SelectionProbabilities)
        assert 0.0 < probs.home < 1.0
        assert 0.0 < probs.draw < 1.0
        assert 0.0 < probs.away < 1.0
        assert probs.home + probs.draw + probs.away == pytest.approx(1.0)

    def test_home_and_away_swap_changes_prediction(
        self, fitted_model: DixonColesModel
    ) -> None:
        # Home advantage ``gamma`` is not symmetric under a swap, so the
        # prediction should differ; this catches the silly bug where the
        # team-index lookup is symmetric.
        ab = fitted_model.predict("A", "B")
        ba = fitted_model.predict("B", "A")
        assert ab != ba


# ---------- synthetic recovery --------------------------------------------


class TestSyntheticRecovery:
    def test_predictions_match_truth_on_dense_data(self) -> None:
        rng = np.random.default_rng(seed=0)
        n_teams = 10
        teams = [f"T{i:02d}" for i in range(n_teams)]
        alpha = rng.normal(0.0, 0.25, n_teams)
        alpha -= alpha.mean()
        beta = rng.normal(0.0, 0.25, n_teams)
        beta -= beta.mean()
        gamma = 0.3

        matches: list[TrainingMatch] = []
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        idx = 0
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                lam_h = math.exp(float(alpha[i] - beta[j] + gamma))
                lam_a = math.exp(float(alpha[j] - beta[i]))
                for _ in range(20):
                    hg = int(rng.poisson(lam_h))
                    ag = int(rng.poisson(lam_a))
                    matches.append(
                        _training_match(
                            home=teams[i],
                            away=teams[j],
                            hg=hg,
                            ag=ag,
                            ts=t0 + timedelta(hours=idx),
                        )
                    )
                    idx += 1

        model = DixonColesModel(l2_reg=0.0, decay_rate=0.0)
        model.fit(matches, anchor_time=t0 + timedelta(hours=idx + 1))

        squared_errors: list[float] = []
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                truth_home, truth_draw, truth_away = _truth_prediction(
                    float(alpha[i]),
                    float(alpha[j]),
                    float(beta[i]),
                    float(beta[j]),
                    gamma,
                )
                fitted = model.predict(teams[i], teams[j])
                squared_errors.extend(
                    [
                        (fitted.home - truth_home) ** 2,
                        (fitted.draw - truth_draw) ** 2,
                        (fitted.away - truth_away) ** 2,
                    ]
                )

        rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
        # With 20 replicas per pair across 90 ordered pairs (1800
        # matches), probability predictions should match ground truth
        # to within 0.03 RMSE. Threshold chosen conservatively: RMSE
        # should sit comfortably below this; a fit that's off by 0.1+
        # per probability indicates a real regression.
        assert rmse < 0.03, f"overall probability RMSE = {rmse:.4f}"


# ---------- L2 regularisation ---------------------------------------------


class TestL2Regularisation:
    def test_l2_shrinks_prediction_variance_on_thin_data(self) -> None:
        rng = np.random.default_rng(seed=0)
        n_teams = 6
        teams = [f"T{i:02d}" for i in range(n_teams)]
        matches: list[TrainingMatch] = []
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        idx = 0
        # Two matches per ordered (home, away) pair: thin-data regime
        # where MLE is under-identified and L2 should visibly shrink.
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                for _ in range(2):
                    hg = int(rng.poisson(1.5))
                    ag = int(rng.poisson(1.5))
                    matches.append(
                        _training_match(
                            home=teams[i],
                            away=teams[j],
                            hg=hg,
                            ag=ag,
                            ts=t0 + timedelta(hours=idx),
                        )
                    )
                    idx += 1

        anchor = t0 + timedelta(hours=idx + 1)
        model_nr = DixonColesModel(l2_reg=0.0, decay_rate=0.0)
        model_nr.fit(matches, anchor_time=anchor)
        model_r = DixonColesModel(l2_reg=0.1, decay_rate=0.0)
        model_r.fit(matches, anchor_time=anchor)

        home_probs_nr: list[float] = []
        home_probs_r: list[float] = []
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                home_probs_nr.append(model_nr.predict(teams[i], teams[j]).home)
                home_probs_r.append(model_r.predict(teams[i], teams[j]).home)

        var_nr = float(np.var(home_probs_nr))
        var_r = float(np.var(home_probs_r))
        # L2 pulls ratings toward 0, which collapses predictions toward
        # a single ``exp(gamma)`` common distribution -> smaller variance
        # across team pairs.
        assert var_r < var_nr, (
            f"expected L2 to shrink prediction variance: "
            f"var(l2=0) = {var_nr:.5f}, var(l2=0.1) = {var_r:.5f}"
        )


# ---------- time-decay ----------------------------------------------------


class TestTimeDecay:
    def test_positive_decay_pulls_prediction_toward_recent_regime(self) -> None:
        # Regime change: in the old half, DOM loses 0-3 at home to every
        # opponent; in the recent half, DOM wins 3-0 at home to every
        # opponent. A flat fit averages to zero net DOM strength; a
        # time-decayed fit should upweight the recent phase and predict
        # DOM to win at home.
        teams = ["DOM", "O0", "O1", "O2", "O3"]
        opponents = teams[1:]
        t0 = datetime(2024, 1, 1, tzinfo=UTC)

        matches: list[TrainingMatch] = []
        # Old regime (days 0-99): DOM weak at home.
        for day in range(100):
            opp = opponents[day % len(opponents)]
            matches.append(
                _training_match(
                    home="DOM",
                    away=opp,
                    hg=0,
                    ag=3,
                    ts=t0 + timedelta(days=day),
                )
            )
        # Recent regime (days 100-199): DOM strong at home.
        for day in range(100, 200):
            opp = opponents[day % len(opponents)]
            matches.append(
                _training_match(
                    home="DOM",
                    away=opp,
                    hg=3,
                    ag=0,
                    ts=t0 + timedelta(days=day),
                )
            )

        anchor = t0 + timedelta(days=200)
        model_flat = DixonColesModel(l2_reg=0.0, decay_rate=0.0)
        model_flat.fit(matches, anchor_time=anchor)
        model_decay = DixonColesModel(l2_reg=0.0, decay_rate=0.03)
        model_decay.fit(matches, anchor_time=anchor)

        p_flat = model_flat.predict("DOM", "O0")
        p_decay = model_decay.predict("DOM", "O0")
        assert p_decay.home > p_flat.home + 0.05, (
            f"decay should lift DOM home prob; flat={p_flat.home:.3f}, "
            f"decay={p_decay.home:.3f}"
        )


# ---------- determinism ---------------------------------------------------


class TestDeterminism:
    def test_two_fits_on_identical_data_yield_identical_predictions(self) -> None:
        matches = [
            _training_match("A", "B", 2, 1, datetime(2024, 1, 1, tzinfo=UTC)),
            _training_match("B", "A", 1, 2, datetime(2024, 1, 2, tzinfo=UTC)),
            _training_match("A", "C", 3, 0, datetime(2024, 1, 3, tzinfo=UTC)),
            _training_match("C", "A", 0, 2, datetime(2024, 1, 4, tzinfo=UTC)),
            _training_match("B", "C", 1, 1, datetime(2024, 1, 5, tzinfo=UTC)),
            _training_match("C", "B", 2, 0, datetime(2024, 1, 6, tzinfo=UTC)),
        ]
        anchor = datetime(2024, 1, 7, tzinfo=UTC)

        m1 = DixonColesModel()
        m1.fit(matches, anchor_time=anchor)
        m2 = DixonColesModel()
        m2.fit(matches, anchor_time=anchor)

        assert m1.home_advantage == m2.home_advantage
        assert m1.known_teams() == m2.known_teams()
        for home in ("A", "B", "C"):
            for away in ("A", "B", "C"):
                if home == away:
                    continue
                assert m1.predict(home, away) == m2.predict(home, away)
