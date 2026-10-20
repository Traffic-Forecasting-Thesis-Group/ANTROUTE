"""CRITIC weights for one decision matrix (one study hour: alternatives are road segments).

D. Diakoulaki, G. Mavrotas and L. Papayannakis, "Determining objective weights in multiple
criteria problems: The CRITIC method", Computers & Operations Research 22(7), 763-770, 1995.
https://doi.org/10.1016/0305-0548(94)00059-H

Criteria are the ten additive criteria, all cost-type (higher is worse); reversing every column
leaves the weights unchanged, so no orientation step is needed. For each criterion j:
  r_j     = (x_j - min x_j) / (max x_j - min x_j)       within-matrix min-max normalization
  sigma_j = standard deviation of r_j (population form)    contrast intensity
  active  = criteria that vary in this matrix (max > min)
  I_j     = sigma_j * sum_{l in active} (1 - corr(x_j, x_l))
  w_j     = I_j / sum_{l in active} I_l,  and w_j = 0 for constant criteria
A constant criterion has no correlation with anything: it is left out of both correlation
dimensions and of every conflict sum, and its unavailable correlations are reported as None
(not as an artificial 0). Correlation is unchanged by min-max scaling, so it is taken on x.
When the information of the active criteria sums to zero (for example, perfectly concordant
varying columns), CRITIC is undefined: UndefinedCriticError is raised, never equal weights.

Statistics use centred cross-products merged chunk by chunk (Chan et al.), so streamed and
one-pass results agree without the cancellation of raw-moment formulas.
"""
from __future__ import annotations

import numpy as np

# A criterion is constant when its range is at most this fraction of its magnitude (floating-point noise).
CONSTANT_TOLERANCE = 1e-12
# Total information at most this fraction of the summed contrast is zero information.
INFORMATION_TOLERANCE = 1e-9


class UndefinedCriticError(ValueError):
    """CRITIC weights do not exist for this matrix (too few alternatives, no varying criterion, or zero information)."""


class CriticAccumulator:
    def __init__(self, criteria: tuple[str, ...]):
        self.criteria = criteria
        size = len(criteria)
        self.count = 0
        self.mean = np.zeros(size)
        self.cross = np.zeros((size, size))  # Sum of centred cross-products.
        self.minimum = np.full(size, np.inf)
        self.maximum = np.full(size, -np.inf)

    def add(self, rows) -> None:
        """rows: [alternatives, criteria] values in criteria order."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(self.criteria):
            raise ValueError("CRITIC rows must be a matrix with one column per criterion")
        if not np.isfinite(rows).all():
            raise ValueError("CRITIC rows must be finite")
        if not len(rows):
            return
        count, mean = len(rows), rows.mean(axis=0)
        centred = rows - mean
        cross = centred.T @ centred
        total = self.count + count
        delta = mean - self.mean
        self.cross += cross + np.outer(delta, delta) * self.count * count / total
        self.mean += delta * count / total
        self.count = total
        self.minimum = np.minimum(self.minimum, rows.min(axis=0))
        self.maximum = np.maximum(self.maximum, rows.max(axis=0))

    def result(self) -> dict:
        if self.count < 2:
            raise UndefinedCriticError("CRITIC needs at least two alternatives")
        names = self.criteria
        spread = self.maximum - self.minimum
        scale = np.maximum(np.abs(self.maximum), np.abs(self.minimum))
        active = spread > CONSTANT_TOLERANCE * np.maximum(scale, 1.0)
        if not active.any():
            raise UndefinedCriticError("CRITIC is undefined: every criterion is constant")
        covariance = self.cross / self.count
        std = np.sqrt(np.clip(np.diag(covariance), 0, None))
        index = np.flatnonzero(active)
        sub = covariance[np.ix_(index, index)] / np.outer(std[index], std[index])
        correlation = np.clip(sub, -1, 1)
        np.fill_diagonal(correlation, 1)
        sigma = std[index] / spread[index]
        conflict = (1 - correlation).sum(axis=1)
        information = sigma * conflict
        total = information.sum()
        # Correlations of exactly concordant columns compute as 1 - O(1e-16); treat that residue as zero information.
        if not np.isfinite(total) or total <= INFORMATION_TOLERANCE * max(sigma.sum(), 1e-300):
            raise UndefinedCriticError("CRITIC is undefined: the varying criteria carry no information (perfectly concordant)")
        weights = dict.fromkeys(names, 0.0)
        contrast, conflicts, informations = dict.fromkeys(names, 0.0), dict.fromkeys(names, None), dict.fromkeys(names, 0.0)
        matrix = {j: dict.fromkeys(names) for j in names}
        for position, j in enumerate(index):
            weights[names[j]] = float(information[position] / total)
            contrast[names[j]] = float(sigma[position])
            conflicts[names[j]] = float(conflict[position])
            informations[names[j]] = float(information[position])
            for other, l in enumerate(index):
                matrix[names[j]][names[l]] = float(correlation[position, other])
        return {"weights": weights, "active": {name: bool(active[k]) for k, name in enumerate(names)},
                "contrast_std": contrast, "conflict": conflicts, "information": informations,
                "correlation": matrix, "alternatives": self.count}


def critic_weights(rows, criteria: tuple[str, ...]) -> dict:
    accumulator = CriticAccumulator(criteria)
    accumulator.add(rows)
    return accumulator.result()
