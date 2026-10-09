import math
import numpy as np
import pytest
from src.routing.metrics import (
    compute_metrics_by_scenario,
    compute_scenario_metrics,
    route_optimality,
    trial_eta_metrics,
)
from src.routing.route_trials import (
    ANTROUTE,
    BASELINE,
    MULTI_DESTINATION,
    RECOMMENDED,
    pick_waypoints,
    score_trip,
)
from src.routing.significance import compare_paired, holm, paired_bootstrap, relative_difference
from test_aco_routing import weighted_risky_direct


def trial(scenario, c_opt, c_pred, actual, predicted):
    return dict(
        scenario_type=scenario, c_optimal=c_opt, c_predicted=c_pred, actual_eta=actual, predicted_eta=predicted
    )


def test_route_optimality_is_optimum_over_chosen_cost():
    assert route_optimality([100, 90], [125, 90]).tolist() == [80.0, 100.0]


def test_route_optimality_rejects_non_positive_cost():
    with pytest.raises(ValueError):
        route_optimality([0.0], [0.0])


def test_metrics_split_by_scenario_with_overall_row():
    trials = [
        trial("pm_peak", 100, 110, 600, 650),
        trial("pm_peak", 90, 90, 500, 480),
        trial("off_peak", 200, 250, 900, 1000),
    ]
    results = compute_metrics_by_scenario(trials)
    assert set(results) == {"overall", "pm_peak", "off_peak"}
    assert results["overall"].n_trials == 3
    assert results["off_peak"].route_optimality_mean == pytest.approx(80.0)
    assert results["pm_peak"].mae == pytest.approx(35.0)


def test_missing_actual_eta_leaves_optimality_but_nans_eta_metrics():
    m = compute_scenario_metrics([100, 90], [110, 90], [math.nan, math.nan], [600, 500])
    assert m.n_trials == 2 and m.n_eta_trials == 0
    assert np.isfinite(m.route_optimality_mean)
    assert all(math.isnan(v) for v in (m.mae, m.rmse, m.mse, m.mape, m.r_squared))


def test_partial_actual_eta_scores_only_matched_trials():
    m = compute_scenario_metrics([1, 1, 1], [1, 1, 1], [100, math.nan, 200], [110, 999, 190])
    assert m.n_eta_trials == 2
    assert m.mae == pytest.approx(10.0)


def test_zero_actual_eta_nans_mape_instead_of_crashing():
    m = compute_scenario_metrics([1, 1], [1, 1], [0.0, 100.0], [10.0, 90.0])
    assert math.isnan(m.mape)
    assert m.mae == pytest.approx(10.0)


def test_empty_and_reserved_scenario_are_rejected():
    with pytest.raises(ValueError):
        compute_metrics_by_scenario([])
    with pytest.raises(ValueError, match="reserved"):
        compute_metrics_by_scenario([trial("overall", 1, 1, 1, 1)])




DETOUR = [100, 101, 102, 103]     # 3 x 200 m, risk 0.1 each
DIRECT = [100, 103]               # 500 m, risk 0.9


def free_flow(wg):
    return np.full(wg.n_edges, 30.0)


def test_optimality_uses_apple_on_both_sides_so_it_cannot_reward_a_low_eta():
    wg = weighted_risky_direct()
    # Baseline predicts a far lower ETA than ANTROUTE, but Apple timed its route slower:
    # optimality must follow Apple's timing of the routes, not the systems' own ETAs.
    ant = score_trip(wg, [DETOUR], free_flow(wg), 1.0, ANTROUTE, RECOMMENDED, [100.0], [110.0])
    base = score_trip(wg, [DIRECT], free_flow(wg), 0.0, BASELINE, RECOMMENDED, [100.0], [160.0])
    assert ant["predicted_eta"] == [pytest.approx(3 * 30.0 * 1.1)]
    assert base["predicted_eta"] == [pytest.approx(30.0)]
    ro = {t["system"]: compute_metrics_by_scenario([t])["overall"].route_optimality_mean for t in (ant, base)}
    assert ro[ANTROUTE] == pytest.approx(100.0 / 110.0 * 100)
    assert ro[BASELINE] == pytest.approx(100.0 / 160.0 * 100)


def test_trip_reports_appendix3_fields():
    wg = weighted_risky_direct()
    t = score_trip(wg, [DETOUR], free_flow(wg), 1.0, ANTROUTE, RECOMMENDED, [100.0], [110.0], "R01")
    assert t["actual_eta"] == [110.0] and t["c_predicted"] == 110.0 and t["c_optimal"] == 100.0
    assert t["distance_m"] == pytest.approx(600.0)
    assert t["risk_exposure"] == pytest.approx(0.3)
    assert t["eta_minutes"] == pytest.approx(99.0 / 60)
    assert t["stops"] == [100, 103] and t["trip_id"] == "R01"


def test_multi_destination_sums_cost_and_scores_eta_per_leg():
    wg = weighted_risky_direct()
    legs = [[100, 101], [101, 102, 103]]
    t = score_trip(wg, legs, free_flow(wg), 1.0, ANTROUTE, MULTI_DESTINATION, [40.0, 70.0], [50.0, 80.0])
    assert t["c_optimal"] == 110.0 and t["c_predicted"] == 130.0
    assert t["path"] == DETOUR and t["stops"] == [100, 101, 103]
    assert t["predicted_eta"] == [pytest.approx(33.0), pytest.approx(66.0)]
    pooled = compute_metrics_by_scenario([t])["overall"]
    assert pooled.n_trials == 1 and pooled.n_eta_trials == 2
    assert pooled.mae == pytest.approx((17.0 + 14.0) / 2)


def test_multi_destination_legs_must_connect_and_match_times():
    wg = weighted_risky_direct()
    with pytest.raises(ValueError, match="continue"):
        score_trip(wg, [[100, 101], [102, 103]], free_flow(wg), 1.0, ANTROUTE, MULTI_DESTINATION, [1, 1], [1, 1])
    with pytest.raises(ValueError, match="leg"):
        score_trip(wg, [DETOUR], free_flow(wg), 1.0, ANTROUTE, RECOMMENDED, [1, 1], [1])


def test_score_trip_rejects_missing_apple_time():
    wg = weighted_risky_direct()
    with pytest.raises(ValueError, match="Apple Maps"):
        score_trip(wg, [DETOUR], free_flow(wg), 1.0, ANTROUTE, RECOMMENDED, [100.0], [math.nan])


def test_score_trip_rejects_a_route_not_on_the_graph():
    wg = weighted_risky_direct()
    with pytest.raises(KeyError):
        score_trip(wg, [[100, 102]], free_flow(wg), 1.0, ANTROUTE, RECOMMENDED, [100.0], [110.0])


def test_per_trial_eta_metrics_leave_r2_undefined_for_one_observation():
    single = trial_eta_metrics([100.0], [90.0])
    assert single["mae"] == 10.0 and single["mse"] == 100.0 and single["mape"] == pytest.approx(10.0)
    assert math.isnan(single["r_squared"])
    assert np.isfinite(trial_eta_metrics([100.0, 200.0, 300.0], [110.0, 190.0, 310.0])["r_squared"])


def test_per_trial_r2_is_undefined_for_two_legs():
    # Two points always fit a line exactly; R^2 over them swings to -50 or -140 on a small
    # miss and says nothing about the model. The 3-stop multi-destination trips have 2 legs.
    two = trial_eta_metrics([1000.0, 1010.0], [500.0, 900.0])
    assert math.isnan(two["r_squared"])
    assert two["mae"] == pytest.approx(305.0)


def test_waypoints_are_inner_nodes_spaced_along_the_route():
    wg = weighted_risky_direct()
    assert pick_waypoints(wg, DETOUR, 1) == [101]     # 300 m mark: 101 (200 m) and 102 (400 m) tie, first wins
    assert pick_waypoints(wg, DETOUR, 2) == [101, 102]
    assert pick_waypoints(wg, DETOUR, 10) == [101, 102]
    assert pick_waypoints(wg, DIRECT, 3) == []
    assert pick_waypoints(wg, DETOUR, 0) == []


def test_relative_difference_follows_equations_7_and_8():
    # Equation 7 (MAE, RMSE, MSE, MAPE): lower is better, so a lower proposed value is positive
    assert relative_difference(200.0, 150.0) == pytest.approx(25.0)
    assert relative_difference(80.0, 90.0) == pytest.approx(-12.5)
    # Equation 8 (Route Optimality, R^2): higher is better, so a higher proposed value is positive
    assert relative_difference(80.0, 90.0, higher_is_better=True) == pytest.approx(12.5)
    assert relative_difference(0.8, 0.6, higher_is_better=True) == pytest.approx(-25.0)
    assert math.isnan(relative_difference(0.0, 1.0))


def test_paired_comparison_applies_the_right_equation_per_metric():
    better = [95.0, 97.0, 92.0]
    worse = [85.0, 88.0, 84.0]
    assert compare_paired("route_optimality", better, worse).relative_difference_pct > 0
    assert compare_paired("r_squared", better, worse).relative_difference_pct > 0
    assert compare_paired("mae", worse, better).relative_difference_pct > 0     # lower error = better


def test_paired_comparison_uses_baseline_minus_proposed():
    proposed = [95.0, 97.0, 92.0, 99.0, 94.0, 96.0, 98.0, 93.0, 97.5, 95.5]
    baseline = [85.0, 88.0, 84.0, 90.0, 83.0, 87.0, 89.0, 86.0, 88.5, 84.5]
    c = compare_paired("route_optimality", proposed, baseline)
    assert c.n == 10
    assert c.mean_difference == pytest.approx(np.mean(np.subtract(baseline, proposed)))
    assert c.wilcoxon_p < 0.05 and c.significant
    assert np.isfinite(c.shapiro_p)
    assert (c.t_value is not None) == c.normal


def test_paired_comparison_handles_ties_and_missing_values():
    c = compare_paired("mae", [1.0, 2.0, math.nan], [1.0, 2.0, 5.0])
    assert c.n == 2                                   # the NaN pair is dropped
    assert math.isnan(c.wilcoxon_p) and not c.significant
    assert math.isnan(c.shapiro_p) and c.t_value is None


def test_holm_adjusts_step_down_and_stays_monotone():
    adjusted = holm([0.01, 0.04, 0.03, math.nan])
    # sorted 0.01, 0.03, 0.04 over m=3 tests: 0.03, 0.06, 0.06 (monotone), NaN untouched
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert math.isnan(adjusted[3])


def test_paired_bootstrap_detects_a_pooled_r2_gap_and_resamples_whole_trips():
    rng = np.random.default_rng(1)
    actual = rng.uniform(600, 3000, 40)
    good = actual * rng.normal(1.0, 0.05, 40)
    bad = actual * rng.normal(1.0, 0.40, 40)
    groups = np.repeat(np.arange(20), 2)          # 20 two-leg trips
    from src.routing.metrics import r_squared
    out = paired_bootstrap(r_squared, actual, good, actual, bad, groups, higher_is_better=True, n_boot=500, seed=0)
    assert out["proposed"] > out["baseline"]
    assert out["ci_low"] > 0 and out["p_value"] < 0.05     # difference oriented: positive = proposed better


def test_paired_bootstrap_no_difference_is_not_significant():
    actual = np.linspace(500, 3000, 30)
    pred = actual * 0.9
    out = paired_bootstrap(lambda a, p: float(np.mean(np.abs(a - p))), actual, pred, actual, pred,
                           np.arange(30), higher_is_better=False, n_boot=300, seed=0)
    assert out["difference"] == pytest.approx(0.0) and out["p_value"] >= 0.05
