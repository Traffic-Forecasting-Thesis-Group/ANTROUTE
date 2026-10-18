"""CRITIC weights for the ten routing features.

D. Diakoulaki, G. Mavrotas and L. Papayannakis, "Determining objective weights in multiple
criteria problems: The CRITIC method", Computers & Operations Research 22(7), 763-770, 1995.
https://doi.org/10.1016/0305-0548(94)00059-H

Alternatives are road segments at sampled study hours; criteria are the ten normalized
features (all cost-type: higher is worse). For each criterion j:
  r_j  = (x_j - min x_j) / (max x_j - min x_j)       min-max over all alternatives
  s_j  = standard deviation of r_j                     contrast intensity (population form)
  C_j  = s_j * sum_k (1 - corr(x_j, x_k))              information: contrast x conflict
  w_j  = C_j / sum_k C_k
Correlation is unchanged by min-max scaling, so it is taken on x directly. Statistics are
accumulated hour by hour, so the full sample never has to fit in memory.
"""
from __future__ import annotations

import numpy as np


class CriticAccumulator:
    def __init__(self, criteria: tuple[str, ...]):
        self.criteria = criteria
        size = len(criteria)
        self.count = 0
        self.sums = np.zeros(size)
        self.products = np.zeros((size, size))
        self.minimum = np.full(size, np.inf)
        self.maximum = np.full(size, -np.inf)

    def add(self, rows: np.ndarray) -> None:
        """rows: [alternatives, criteria] values in criteria order."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(self.criteria) or not np.isfinite(rows).all():
            raise ValueError("CRITIC rows must be finite with one column per criterion")
        self.count += len(rows)
        self.sums += rows.sum(axis=0)
        self.products += rows.T @ rows
        self.minimum = np.minimum(self.minimum, rows.min(axis=0, initial=np.inf))
        self.maximum = np.maximum(self.maximum, rows.max(axis=0, initial=-np.inf))

    def result(self) -> dict:
        if self.count < 2:
            raise ValueError("CRITIC needs at least two alternatives")
        mean = self.sums / self.count
        covariance = self.products / self.count - np.outer(mean, mean)
        std = np.sqrt(np.clip(np.diag(covariance), 0, None))
        spread = self.maximum - self.minimum
        contrast = np.divide(std, spread, out=np.zeros_like(std), where=spread > 0)
        denominator = np.outer(std, std)
        # A constant criterion has no contrast, so its weight is zero whatever its correlation.
        correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
        np.fill_diagonal(correlation, 1)
        correlation = np.clip(correlation, -1, 1)
        conflict = (1 - correlation).sum(axis=1)
        information = contrast * conflict
        if information.sum() <= 0:
            raise ValueError("CRITIC is undefined when every criterion is constant")
        weights = information / information.sum()
        name = dict(enumerate(self.criteria))
        return {
            "weights": {name[j]: float(weights[j]) for j in name},
            "contrast_std": {name[j]: float(contrast[j]) for j in name},
            "conflict": {name[j]: float(conflict[j]) for j in name},
            "information": {name[j]: float(information[j]) for j in name},
            "correlation": {name[j]: {name[k]: float(correlation[j, k]) for k in name} for j in name},
            "alternatives": self.count,
        }


def critic_weights(rows, criteria: tuple[str, ...]) -> dict:
    accumulator = CriticAccumulator(criteria)
    accumulator.add(np.asarray(rows, dtype=np.float64))
    return accumulator.result()
