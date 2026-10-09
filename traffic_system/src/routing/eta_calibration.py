"""
Putting a system's ETA on the scale of the ground truth it is scored against, without
fitting on the trip being scored.

Both systems predict ETA as free-flow time x (1 + gamma * congestion), with free-flow from
OpenStreetMap speed limits and gamma = 1, so the prediction can never exceed twice the
speed-limit time. Apple Maps puts Metro Manila's peak-hour routes at roughly 3-4x that
(about 11 km/h against ~46 km/h of speed limits on the May 25 test trips). Uncorrected,
every ETA metric then mostly measures that ceiling: ANTROUTE's predictions were below
Apple's on 70 of 70 legs at ~45% of it, while its ranking of trips was as good as the
baseline's (r = 0.90 vs 0.90). The baseline lost less only because its observed camera
label sat at Heavy (multiplier 2.0) while ANTROUTE's predicted risk sat at ~0.69 (1.69).

So: one scale factor per system, least squares through the origin, cross-fitted by trip.
Each trip's legs are scaled by a factor fitted on every OTHER trip, so no trip is scored by
a calibration that has seen its own Apple Maps time -- the same discipline
src/models/risk_calibration.py applies by fitting on the training split only. A trip's legs
are held out together, since legs of one trip share a window and a corridor.

Kept beside the raw ETA rather than replacing it, so the thesis can report both and say
which is which. One parameter, not an affine or per-scenario map: with ~50 trips a richer
map would start fitting the test set, and a scale cannot change any ranking, so it adds no
information the system did not already have.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def fit_scale(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """k minimising sum((actual - k * predicted)^2)."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    keep = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[keep], predicted[keep]
    denominator = float(np.sum(predicted**2))
    if denominator <= 0:
        raise ValueError("no positive predicted ETA to fit a scale on")
    return float(np.sum(actual * predicted) / denominator)


def cross_fitted_scale(
    actual: Sequence[float], predicted: Sequence[float], groups: Sequence
) -> Tuple[np.ndarray, float]:
    """
    (calibrated predictions, scale fitted on everything).

    Each group's predictions are multiplied by the scale fitted on all other groups. The
    all-data scale is returned for reporting only; it is never applied to a scored value.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    groups = np.asarray(groups)
    if not (actual.shape == predicted.shape == groups.shape):
        raise ValueError("actual, predicted and groups must align one per leg")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("cross-fitting needs at least two trips: one to fit on, one to score")
    calibrated = np.full(predicted.shape, np.nan)
    for g in unique:
        held = groups == g
        calibrated[held] = fit_scale(actual[~held], predicted[~held]) * predicted[held]
    return calibrated, fit_scale(actual, predicted)
