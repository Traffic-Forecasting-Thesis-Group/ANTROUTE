"""
Putting the STGNN's risk scores back on the scale of the labels they are compared against.

The decoder ends in a sigmoid and is trained on a weak target of 0.0 / 0.5 / 1.0, but its
output stays bunched near the middle: on the v6 test split the predictions average 0.568
while the labels average 0.784. The ordering is right -- mean prediction rises Light 0.516,
Medium 0.566, Heavy 0.577 -- so the model has learned which roads are worse; it just states
it too quietly, and nearly all of the resulting MAE is that constant offset rather than a
wrong ranking. Uncorrected it loses to a constant "always predict the training mean", which
makes a genuinely informative model look worse than no model at all.

So: one affine map, slope and intercept, least squares. Fitted on the TRAINING split's
labelled camera edges only and then applied unchanged to val and test, which is what keeps
the held-out numbers honest -- a calibration fitted on the split it is scored on would be
circular in exactly the way scripts/evaluate_baseline_iaco.py was.

Two deliberate limits. Affine, not isotonic or Platt: with three distinct target values and
~6.6k training rows a monotone step function would chase the class proportions, and an
affine map cannot invent separation the model did not find. And it is a separate column,
not a replacement for `risk`: being monotone it leaves every edge's risk *ranking* untouched,
and keeping both lets the thesis report the raw and calibrated figures side by side instead of
quietly substituting one. It does NOT leave routes unchanged: the routing cost
W = d * (1 + lambda * Risk) mixes distance with risk, and an affine change of Risk changes how
much risk weighs against distance, so routing on the calibrated column (RISK_SCORES=calibrated
in the app) can pick different routes and needs its own routing evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

MIN_SAMPLES = 50  # below this a two-parameter fit is noise, not calibration


@dataclass(frozen=True)
class Calibration:
    """y_hat = clip(slope * risk + intercept, 0, 1), with the fit's provenance attached."""

    slope: float
    intercept: float
    n_samples: int
    fitted_on: str

    def apply(self, risk: np.ndarray) -> np.ndarray:
        return np.clip(self.slope * np.asarray(risk, dtype=np.float64) + self.intercept, 0.0, 1.0)

    def as_dict(self) -> dict:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "n_samples": self.n_samples,
            "fitted_on": self.fitted_on,
        }

    def __str__(self) -> str:
        return (f"risk_calibrated = clip({self.slope:.4f} * risk + {self.intercept:.4f}, 0, 1)"
                f"   [fitted on {self.n_samples} {self.fitted_on} edges]")


def fit(
    risk: np.ndarray,
    target: np.ndarray,
    fitted_on: str = "train",
) -> Calibration:
    """
    Least-squares affine fit of the weak target on the raw risk score.

    Raises rather than returning an identity when the fit cannot be trusted: a silent
    identity would be indistinguishable from a calibration that happened to be unnecessary,
    and the caller needs to know which it got.
    """
    risk = np.asarray(risk, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if risk.shape != target.shape:
        raise ValueError(f"{risk.size} risk values but {target.size} targets")
    keep = np.isfinite(risk) & np.isfinite(target)
    risk, target = risk[keep], target[keep]

    if risk.size < MIN_SAMPLES:
        raise ValueError(
            f"only {risk.size} usable {fitted_on} edges, need {MIN_SAMPLES} to fit a calibration"
        )
    if np.ptp(risk) < 1e-9:
        raise ValueError(
            f"the {fitted_on} risk scores are all but constant (spread {np.ptp(risk):.2e}); "
            "there is no slope to fit, and the model has not separated the edges at all"
        )

    slope, intercept = np.polyfit(risk, target, 1)
    return Calibration(float(slope), float(intercept), int(risk.size), fitted_on)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    """MAE, RMSE and R2 -- the same three app/evaluation.py reports."""
    target = np.asarray(target, dtype=np.float64)
    error = np.asarray(prediction, dtype=np.float64) - target
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": 1 - float(np.sum(error**2)) / ss_tot if ss_tot > 0 else float("nan"),
    }


def class_separation(target: np.ndarray, prediction: np.ndarray) -> Tuple[dict, float]:
    """
    Mean prediction per true class, and the Pearson r.

    This is the pair of numbers that distinguishes "miscalibrated" from "learned nothing",
    and calibration is only defensible when the per-class means are in the right order.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    per_class = {
        float(value): float(prediction[target == value].mean())
        for value in np.unique(target)
    }
    r = float(np.corrcoef(target, prediction)[0, 1]) if np.ptp(prediction) > 0 else float("nan")
    return per_class, r
