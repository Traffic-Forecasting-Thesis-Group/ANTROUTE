from app.schemas_route import (
    ComparisonMetricRow,
    ComparisonMetricsResponse,
    OptimalityPct,
)

# Real numbers, not placeholders -- computed by scripts/compute_comparison_metrics.py
# from the actual scored risk_edges.csv (checkpoint stgnn_edge_v6, confirmed via its own
# risk_summary.json), against the TEST split's real human-labelled Light/Medium/Heavy
# targets, over 1,679 labelled camera edges.
#
#   python scripts/compute_comparison_metrics.py
#
# Two things changed from the v4 figures that stood here before, and both matter.
#
# The baseline. It was "always predict Medium (0.5)", the reference used during training.
# That is a weak bar on this data: the test labels are 67% Heavy (mean 0.784), so a
# constant sitting at the TRAINING mean of 0.717 is far stronger, and it is the one
# reported here. ANTROUTE must beat that to have earned anything. For the record the
# 0.5 constant scores MAE 0.3862 / RMSE 0.4395 / R2 -0.7206.
#
# The prediction column. The STGNN's raw output is miscalibrated: it averages 0.568
# where the labels average 0.784, while still ordering the classes correctly on every
# split (mean prediction 0.516 Light, 0.566 Medium, 0.577 Heavy). Nearly all of its
# error is that constant offset, so an affine map fitted on the TRAIN split alone --
# risk_calibrated = clip(0.6901 * risk + 0.3225, 0, 1) -- is applied before scoring;
# see src/models/risk_calibration.py. The map is monotone, so no route changes.
# The uncalibrated figures are reported alongside rather than dropped, because the
# calibration is part of the method and a reader is entitled to see what it bought.
#
# Even calibrated this is a modest result: ANTROUTE beats the train-mean constant by
# about 2% MAE, and R2 is still slightly negative. Report it as such. Re-run the
# command above and update these numbers whenever risk_edges.csv is regenerated.
RAW_RESULTS = {
    "mae": (0.3049, 0.3121),
    "rmse": (0.3389, 0.3417),
    "r2": (-0.0231, -0.0401),
}

# The same model scored on its raw `risk` column, against the same baseline. Shown in the
# table so the calibration's effect is visible instead of silently folded in.
UNCALIBRATED_RESULTS = {
    "mae": (0.3626, 0.3121),
    "rmse": (0.3990, 0.3417),
    "r2": (-0.4181, -0.0401),
}


def _pct_change(ant: float, base: float) -> int:
    return round((ant - base) / base * 100)


def get_comparison_metrics() -> ComparisonMetricsResponse:
    # Derived from real MAE (0..1 scale, same scale as the risk targets), not a
    # separately-tracked metric: optimality% = (1 - MAE) * 100. This keeps the
    # schema's required route_optimality_pct field honest rather than inventing an
    # unrelated number for it.
    mae_a, mae_b = RAW_RESULTS["mae"]
    opt_a, opt_b = round((1 - mae_a) * 100, 1), round((1 - mae_b) * 100, 1)

    rows = [
        ComparisonMetricRow(
            metric="Risk Score Accuracy (1 - MAE)",
            antroute=f"{opt_a:g}%",
            baseline=f"{opt_b:g}%",
            improvement_pct=round(opt_a - opt_b),  # percentage points
            higher_is_better=True,
        )
    ]

    specs = [
        ("Congestion Risk MAE", "mae", False),
        ("Congestion Risk RMSE", "rmse", False),
        # R² can be negative, so a relative percent-of-base is undefined/misleading
        # here (a negative base flips the sign of an actual improvement) -- use
        # percentage points instead, same convention as the accuracy row above.
        ("R-squared", "r2", True),
    ]
    for label, key, higher in specs:
        a, b = RAW_RESULTS[key]
        improvement = round((a - b) * 100) if key == "r2" else _pct_change(a, b)
        rows.append(
            ComparisonMetricRow(
                metric=label,
                antroute=f"{a:g}",
                baseline=f"{b:g}",
                improvement_pct=improvement,
                higher_is_better=higher,
            )
        )

    # What the calibration bought, stated rather than absorbed: the raw score against the
    # same baseline. Without this row the table would read as if the model had always been
    # on the labels' scale.
    raw_mae, raw_base = UNCALIBRATED_RESULTS["mae"]
    rows.append(
        ComparisonMetricRow(
            metric="Congestion Risk MAE (uncalibrated)",
            antroute=f"{raw_mae:g}",
            baseline=f"{raw_base:g}",
            improvement_pct=_pct_change(raw_mae, raw_base),
            higher_is_better=False,
        )
    )

    return ComparisonMetricsResponse(
        route_optimality_pct=OptimalityPct(antroute=opt_a, baseline=opt_b),
        metrics=rows,
        baseline_name="Always 0.717 (training mean)",
    )
