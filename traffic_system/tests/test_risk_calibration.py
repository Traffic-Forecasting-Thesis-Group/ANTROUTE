import numpy as np
import pytest

from src.models.risk_calibration import (
    MIN_SAMPLES,
    Calibration,
    class_separation,
    fit,
    metrics,
)


def squashed(target: np.ndarray, centre: float = 0.57, spread: float = 0.06) -> np.ndarray:
    """A prediction with the right ordering but far too little range -- the v6 failure mode."""
    return centre + spread * (target - target.mean())


TARGET = np.repeat([0.0, 0.5, 1.0], 100)


def test_fit_recovers_the_affine_map_it_was_given():
    risk = (TARGET - 0.3) / 2.5          # exactly invertible: y = 2.5 * risk + 0.3
    calibration = fit(risk, TARGET)
    assert calibration.slope == pytest.approx(2.5)
    assert calibration.intercept == pytest.approx(0.3)
    assert calibration.n_samples == TARGET.size
    assert calibration.apply(risk) == pytest.approx(TARGET)


def test_calibration_clamps_to_the_unit_interval():
    # Risk is a probability and so is the target, so nothing outside [0, 1] is meaningful
    # however extreme the fitted slope.
    calibrated = Calibration(slope=5.0, intercept=-1.0, n_samples=999, fitted_on="train").apply(
        np.array([0.0, 0.2, 0.3, 0.9])
    )
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0
    assert calibrated.tolist() == [0.0, 0.0, 0.5, 1.0]


def test_calibration_beats_the_raw_score_it_was_fitted_on():
    risk = squashed(TARGET)
    calibration = fit(risk, TARGET)
    raw = metrics(TARGET, risk)
    calibrated = metrics(TARGET, calibration.apply(risk))
    assert calibrated["mae"] < raw["mae"]
    assert calibrated["rmse"] < raw["rmse"]
    assert calibrated["r2"] > raw["r2"]


def test_calibration_does_not_reorder_edges():
    """
    The point of an affine map: routes cannot change, because the ranking cannot change.

    A calibration with a negative slope would reverse it, so the ordering is asserted rather
    than assumed -- it is the property that makes it safe to leave routing on the raw column.
    """
    risk = squashed(TARGET)
    calibration = fit(risk, TARGET)
    assert calibration.slope > 0
    sample = np.linspace(0.3, 0.8, 50)
    assert np.all(np.diff(calibration.apply(sample)) >= 0)


def test_fit_refuses_too_few_samples():
    risk = np.linspace(0.4, 0.6, MIN_SAMPLES - 1)
    with pytest.raises(ValueError, match="need"):
        fit(risk, np.linspace(0.0, 1.0, MIN_SAMPLES - 1))


def test_fit_refuses_a_constant_prediction():
    # The case worth failing loudly on: a model that emits one number everywhere has no
    # slope to fit, and silently returning an identity would hide that.
    with pytest.raises(ValueError, match="constant"):
        fit(np.full(TARGET.size, 0.57), TARGET)


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="targets"):
        fit(np.zeros(10), np.zeros(11))


def test_fit_ignores_rows_with_no_score():
    risk = squashed(TARGET).copy()
    risk[:10] = np.nan
    assert fit(risk, TARGET).n_samples == TARGET.size - 10


def test_class_separation_reports_the_ordering():
    per_class, r = class_separation(TARGET, squashed(TARGET))
    assert sorted(per_class) == [0.0, 0.5, 1.0]
    assert per_class[0.0] < per_class[0.5] < per_class[1.0]
    assert r == pytest.approx(1.0)


def test_metrics_scores_a_perfect_prediction():
    assert metrics(TARGET, TARGET) == pytest.approx({"mae": 0.0, "rmse": 0.0, "r2": 1.0})
