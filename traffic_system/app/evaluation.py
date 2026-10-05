from app.schemas_route import (
    ComparisonMetricRow,
    ComparisonMetricsResponse,
    OptimalityPct,
)

# Real numbers, not placeholders -- computed by scripts/compute_comparison_metrics.py
# from the actual scored risk_edges.csv (checkpoint stgnn_edge_v4.pt, confirmed via
# its own risk_summary.json), against the TEST split's real human-labelled
# Light/Medium/Heavy targets. "Baseline" is the naive "always predict Medium (0.5)"
# constant -- the same reference point used throughout training.
#
#   python scripts/compute_comparison_metrics.py
#
# As of this run (1,456 labelled test-split camera edges): ANTROUTE barely beats
# the naive baseline (MAE 0.374 vs 0.378, ~1%). This is a real, held-out result,
# not a strong one -- it should be reported honestly as such, not framed to look
# more conclusive than it is. Re-run the command above and update these three
# numbers whenever risk_edges.csv is regenerated from a new checkpoint.
RAW_RESULTS = {
    "mae": (0.3744, 0.3781),
    "rmse": (0.4210, 0.4348),
    "r2": (-0.4632, -0.5609),
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

    return ComparisonMetricsResponse(
        route_optimality_pct=OptimalityPct(antroute=opt_a, baseline=opt_b),
        metrics=rows,
        baseline_name="Always 'Medium'",
    )
