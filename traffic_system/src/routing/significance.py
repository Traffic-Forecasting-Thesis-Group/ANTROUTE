"""
Paired significance testing of ANTROUTE against the baseline (thesis Section 3.9, Appendix 3).

Per metric, over trials paired by trip (same origin/destination, window and stops):

    d_i  = C_baseline,i - C_proposed,i                       (Computational Step 1)
    d̄, S_d                                                   (Equation 9)
    Shapiro-Wilk on d_i                                       (Equation 8)
    Wilcoxon signed-rank, W = min(W+, W-)                     (Equation 12)
    paired t-test, reported only when d_i is normal           (Equations 10-11)
    Δ% = (C_baseline - C_proposed) / C_baseline * 100         (Equation 7: MAE, RMSE, MSE, MAPE)
    Δ% = (C_proposed - C_baseline) / C_baseline * 100         (Equation 8: Route Optimality, R^2)

so a positive Δ% always means ANTROUTE did better, and a negative one that it did worse
(thesis Section 3.9, "Relative Difference"). HIGHER_IS_BETTER names the metrics Equation 8
applies to.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np
from scipy import stats

ALPHA = 0.05

# Equation 8 metrics; every other metric is lower-is-better (Equation 7).
HIGHER_IS_BETTER = frozenset({"route_optimality", "r_squared"})


@dataclass
class PairedComparison:
    metric: str
    n: int
    proposed_mean: float
    proposed_sd: float
    baseline_mean: float
    baseline_sd: float
    mean_difference: float              # d̄, with d = baseline - proposed
    sd_difference: float                # S_d
    relative_difference_pct: float      # Δ% of the means, Equation 7 or 8 (positive = proposed better)
    shapiro_w: float
    shapiro_p: float
    normal: bool                        # Shapiro-Wilk p >= alpha
    wilcoxon_w: float
    wilcoxon_p: float
    t_value: Optional[float]            # supplementary, only when normal
    t_p: Optional[float]
    significant: bool                   # Wilcoxon p < alpha


def relative_difference(baseline: float, proposed: float, higher_is_better: bool = False) -> float:
    """
    Δ%: Equation 7 for lower-is-better metrics, Equation 8 for higher-is-better ones.
    Positive means the proposed model did better. NaN when the baseline value is 0.
    """
    if baseline == 0:
        return float("nan")
    diff = (proposed - baseline) if higher_is_better else (baseline - proposed)
    return float(diff / abs(baseline) * 100.0)


def holm(p_values: Sequence[float]) -> List[float]:
    """
    Holm-Bonferroni adjusted p-values, in the input order; NaN stays NaN and is not counted.

    Six metrics are tested per scenario at alpha = 0.05, so by chance alone about one in
    four scenarios would show a "significant" metric. Holm controls that family-wise error
    without Bonferroni's loss of power. Compare the adjusted value against alpha as usual.
    """
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    running = 0.0
    for rank, i in enumerate(finite[np.argsort(p[finite], kind="stable")]):
        running = max(running, min(1.0, (m - rank) * p[i]))   # step-down, kept monotone
        out[i] = running
    return out.tolist()


def paired_bootstrap(
    metric: Callable[[np.ndarray, np.ndarray], float],
    actual_proposed: Sequence[float],
    predicted_proposed: Sequence[float],
    actual_baseline: Sequence[float],
    predicted_baseline: Sequence[float],
    groups: Sequence,
    higher_is_better: bool,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = ALPHA,
) -> Dict[str, float]:
    """
    Paired bootstrap of a POOLED metric (R^2, RMSE, MSE over every leg), resampling whole trips.

    Pooled metrics are one number per system, so there is nothing per trial for Wilcoxon to
    rank -- and per-trial R^2 is undefined for 1- and 2-leg trips. Resampling trips (all of a
    trip's legs together, for both systems at once) keeps the pairing and the within-trip
    correlation. `difference` is oriented so positive means the proposed model is better,
    matching relative_difference. The p-value is the two-sided percentile p for no difference.
    """
    ap, pp = np.asarray(actual_proposed, float), np.asarray(predicted_proposed, float)
    ab, pb = np.asarray(actual_baseline, float), np.asarray(predicted_baseline, float)
    groups = np.asarray(groups)
    if not (ap.shape == pp.shape == ab.shape == pb.shape == groups.shape):
        raise ValueError("both systems need one value per leg, aligned to the same groups")
    sign = 1.0 if higher_is_better else -1.0

    def diff(idx: np.ndarray) -> float:
        return sign * (metric(ap[idx], pp[idx]) - metric(ab[idx], pb[idx]))

    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, unique.size, unique.size)])
        try:
            draws[b] = diff(idx)
        except ValueError:          # e.g. R^2 on a resample whose actual values are all equal
            draws[b] = np.nan
    draws = draws[np.isfinite(draws)]
    everything = np.arange(groups.size)
    observed = diff(everything)
    p = 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0)) if draws.size else float("nan")
    return {
        "proposed": float(metric(ap, pp)),
        "baseline": float(metric(ab, pb)),
        "difference": float(observed),
        "ci_low": float(np.quantile(draws, alpha / 2)) if draws.size else float("nan"),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)) if draws.size else float("nan"),
        "p_value": float(min(1.0, p)),
        "n_trips": int(unique.size),
        "n_boot": int(draws.size),
    }


def _sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def compare_paired(
    metric: str, proposed: Sequence[float], baseline: Sequence[float], alpha: float = ALPHA
) -> PairedComparison:
    proposed = np.asarray(proposed, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if proposed.shape != baseline.shape:
        raise ValueError(f"{metric}: proposed {proposed.shape} and baseline {baseline.shape} are not paired")
    keep = np.isfinite(proposed) & np.isfinite(baseline)
    proposed, baseline = proposed[keep], baseline[keep]
    d = baseline - proposed
    n = int(d.size)
    nan = float("nan")

    shapiro_w = shapiro_p = nan
    if n >= 3 and np.ptp(d) > 0:                 # Shapiro-Wilk needs 3+ values that are not all equal
        shapiro_w, shapiro_p = (float(v) for v in stats.shapiro(d))
    normal = bool(np.isfinite(shapiro_p) and shapiro_p >= alpha)

    wilcoxon_w = wilcoxon_p = nan
    if n >= 1 and np.any(d != 0):                # undefined when every pair ties
        res = stats.wilcoxon(baseline, proposed)
        wilcoxon_w, wilcoxon_p = float(res.statistic), float(res.pvalue)

    t_value = t_p = None
    if normal:
        res = stats.ttest_rel(baseline, proposed)
        t_value, t_p = float(res.statistic), float(res.pvalue)

    proposed_mean = float(np.mean(proposed)) if n else nan
    baseline_mean = float(np.mean(baseline)) if n else nan
    return PairedComparison(
        metric=metric,
        n=n,
        proposed_mean=proposed_mean,
        proposed_sd=_sd(proposed),
        baseline_mean=baseline_mean,
        baseline_sd=_sd(baseline),
        mean_difference=float(np.mean(d)) if n else nan,
        sd_difference=_sd(d),
        relative_difference_pct=(
            relative_difference(baseline_mean, proposed_mean, metric in HIGHER_IS_BETTER) if n else nan
        ),
        shapiro_w=shapiro_w,
        shapiro_p=shapiro_p,
        normal=normal,
        wilcoxon_w=wilcoxon_w,
        wilcoxon_p=wilcoxon_p,
        t_value=t_value,
        t_p=t_p,
        significant=bool(np.isfinite(wilcoxon_p) and wilcoxon_p < alpha),
    )
