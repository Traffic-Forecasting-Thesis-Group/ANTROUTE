from app.schemas_route import (
    ComparisonMetricRow,
    ComparisonMetricsResponse,
    OptimalityPct,
)

# Offline evaluation results from your test set: (ANTRoute, Baseline).
# Replace these with your real thesis numbers. Everything the app shows
# (formatting, improvement %) is computed from them, so this is the only
# place you need to edit.
RAW_RESULTS = {
    "route_optimality": (94.0, 84.0),  # percent
    "mae": (2.1, 4.6),                 # minutes
    "rmse": (3.0, 6.2),                # minutes
    "mse": (9.0, 38.4),
    "mape": (6.8, 14.9),               # percent
    "r2": (0.91, 0.74),
}


def _pct_change(ant: float, base: float) -> int:
    return round((ant - base) / base * 100)


def get_comparison_metrics() -> ComparisonMetricsResponse:
    opt_a, opt_b = RAW_RESULTS["route_optimality"]

    rows = [
        ComparisonMetricRow(
            metric="Route Optimality",
            antroute=f"{opt_a:g}%",
            baseline=f"{opt_b:g}%",
            improvement_pct=round(opt_a - opt_b),  # percentage points
            higher_is_better=True,
        )
    ]

    specs = [
        ("MAE", "mae", " min", "", False),
        ("RMSE", "rmse", " min", "", False),
        ("MSE", "mse", "", "", False),
        ("MAPE", "mape", "", "%", False),
        ("R²", "r2", "", "", True),
    ]
    for label, key, suffix, pct, higher in specs:
        a, b = RAW_RESULTS[key]
        rows.append(
            ComparisonMetricRow(
                metric=label,
                antroute=f"{a:g}{suffix}{pct}",
                baseline=f"{b:g}{suffix}{pct}",
                improvement_pct=_pct_change(a, b),
                higher_is_better=higher,
            )
        )

    return ComparisonMetricsResponse(
        route_optimality_pct=OptimalityPct(antroute=opt_a, baseline=opt_b),
        metrics=rows,
    )
