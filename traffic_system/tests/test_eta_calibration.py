import numpy as np
import pytest

from src.routing.eta_calibration import cross_fitted_scale, fit_scale


def test_fit_scale_is_least_squares_through_the_origin():
    predicted = np.array([100.0, 200.0, 300.0])
    assert fit_scale(2.0 * predicted, predicted) == pytest.approx(2.0)


def test_fit_scale_rejects_nothing_to_fit():
    with pytest.raises(ValueError):
        fit_scale([100.0], [0.0])


def test_cross_fitted_scale_never_uses_a_trips_own_legs():
    # Trip "a" is off by 2x, trips "b" and "c" by 3x. Leave-one-trip-out means trip a is
    # scaled only by b and c (3x), so its calibrated value must NOT come back exact.
    actual = np.array([200.0, 300.0, 600.0])
    predicted = np.array([100.0, 100.0, 200.0])
    groups = ["a", "b", "c"]
    calibrated, k = cross_fitted_scale(actual, predicted, groups)
    assert calibrated[0] == pytest.approx(300.0)          # 100 * 3, fitted on b and c only
    assert k == pytest.approx(fit_scale(actual, predicted))


def test_cross_fitted_scale_keeps_a_trips_legs_together():
    # Both legs of trip "m" are held out together, so neither can inform the other.
    actual = np.array([200.0, 400.0, 300.0, 300.0])
    predicted = np.array([100.0, 200.0, 100.0, 100.0])
    calibrated, _ = cross_fitted_scale(actual, predicted, ["m", "m", "x", "y"])
    assert calibrated[:2] == pytest.approx([300.0, 600.0])


def test_cross_fitted_scale_needs_two_groups():
    with pytest.raises(ValueError, match="two"):
        cross_fitted_scale([100.0, 200.0], [50.0, 100.0], ["only", "only"])
