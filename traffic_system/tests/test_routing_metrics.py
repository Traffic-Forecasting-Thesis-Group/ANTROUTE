import math
import numpy as np
import pytest
from src.routing.metrics import compute_metrics_by_scenario, route_optimality
from src.routing.route_trials import (
    ANTROUTE,
    BASELINE,
    MULTI_DESTINATION,
    RECOMMENDED,
    pick_waypoints,
    score_trip,
)
from src.routing.significance import compare_paired, relative_difference
from test_aco_routing import weighted_risky_direct


def trial(scenario, c_opt, c_pred, congestion=1.0, event=0.0, distance=1000.0, avoidable=0.0):
    return dict(scenario_type=scenario, c_optimal=c_opt, c_predicted=c_pred, congestion_exposure=congestion,
                event_exposure=event, avoidable_event_exposure=avoidable, distance_m=distance)


def test_route_optimality_is_optimum_over_chosen_cost():
    assert route_optimality([100, 90], [125, 90]).tolist() == [80.0, 100.0]


def test_route_optimality_rejects_non_positive_cost():
    with pytest.raises(ValueError):
        route_optimality([0.0], [0.0])


def test_metrics_split_by_scenario_with_overall_row():
    trials = [
        trial("pm_peak", 100, 110, congestion=2.0, event=0.5),
        trial("pm_peak", 90, 90, congestion=4.0, event=0.0),
        trial("off_peak", 200, 250, congestion=1.0),
    ]
    results = compute_metrics_by_scenario(trials)
    assert set(results) == {"overall", "pm_peak", "off_peak"}
    assert results["overall"].n_trials == 3
    assert results["off_peak"].route_optimality_mean == pytest.approx(80.0)
    assert results["pm_peak"].congestion_exposure_mean == pytest.approx(3.0)
    assert results["pm_peak"].event_exposure_mean == pytest.approx(0.25)


def test_no_eta_metric_is_computed():
    m = compute_metrics_by_scenario([trial("x", 1, 1)])["overall"]
    assert not any(name in vars(m) for name in ("mae", "rmse", "mse", "mape", "r_squared"))


def test_empty_and_reserved_scenario_are_rejected():
    with pytest.raises(ValueError):
        compute_metrics_by_scenario([])
    with pytest.raises(ValueError, match="reserved"):
        compute_metrics_by_scenario([trial("overall", 1, 1)])


DETOUR = [100, 101, 102, 103]     # 3 x 200 m, risk 0.1 each
DIRECT = [100, 103]               # 500 m, risk 0.9


def test_optimality_is_timed_by_the_reference_on_both_sides():
    wg = weighted_risky_direct()
    ant = score_trip(wg, [DETOUR], ANTROUTE, RECOMMENDED, [100.0], [110.0])
    base = score_trip(wg, [DIRECT], BASELINE, RECOMMENDED, [100.0], [160.0])
    ro = {t["system"]: compute_metrics_by_scenario([t])["overall"].route_optimality_mean for t in (ant, base)}
    assert ro[ANTROUTE] == pytest.approx(100.0 / 110.0 * 100)
    assert ro[BASELINE] == pytest.approx(100.0 / 160.0 * 100)


def test_trip_reports_exposures_and_length_not_an_eta():
    wg = weighted_risky_direct()
    t = score_trip(wg, [DETOUR], ANTROUTE, RECOMMENDED, [100.0], [110.0], "R01")
    assert t["reference_route_seconds"] == [110.0] and t["c_predicted"] == 110.0 and t["c_optimal"] == 100.0
    assert t["distance_m"] == pytest.approx(600.0)
    assert t["congestion_exposure"] == pytest.approx(0.3)
    assert t["event_exposure"] == 0.0                  # no incident in force
    assert t["stops"] == [100, 103] and t["trip_id"] == "R01"
    assert not any("eta" in key for key in t)


def test_event_exposure_sums_the_incident_impact_on_the_route():
    wg = weighted_risky_direct()
    impact = np.zeros(wg.n_edges)
    impact[wg.edge_id(wg.index_of(100), wg.index_of(103))] = 0.8     # an incident on the direct road
    direct = score_trip(wg, [DIRECT], BASELINE, RECOMMENDED, [100.0], [120.0], impact=impact)
    detour = score_trip(wg, [DETOUR], ANTROUTE, RECOMMENDED, [100.0], [120.0], impact=impact)
    assert direct["event_exposure"] == pytest.approx(0.8)
    assert detour["event_exposure"] == 0.0
    # the direct road joins the trip's own stops, so that incident was never avoidable
    assert direct["avoidable_event_exposure"] == 0.0


def test_avoidable_event_exposure_counts_only_incidents_away_from_the_stops():
    wg = weighted_risky_direct()
    impact = np.zeros(wg.n_edges)
    impact[wg.edge_id(wg.index_of(101), wg.index_of(102))] = 0.6     # mid-route on the detour
    impact[wg.edge_id(wg.index_of(100), wg.index_of(101))] = 0.4     # at the origin
    t = score_trip(wg, [DETOUR], ANTROUTE, RECOMMENDED, [100.0], [120.0], impact=impact)
    assert t["event_exposure"] == pytest.approx(1.0)
    assert t["avoidable_event_exposure"] == pytest.approx(0.6)


def test_multi_destination_sums_the_reference_over_legs():
    wg = weighted_risky_direct()
    legs = [[100, 101], [101, 102, 103]]
    t = score_trip(wg, legs, ANTROUTE, MULTI_DESTINATION, [40.0, 70.0], [50.0, 80.0])
    assert t["c_optimal"] == 110.0 and t["c_predicted"] == 130.0
    assert t["path"] == DETOUR and t["stops"] == [100, 101, 103]


def test_multi_destination_legs_must_connect_and_match_times():
    wg = weighted_risky_direct()
    with pytest.raises(ValueError, match="continue"):
        score_trip(wg, [[100, 101], [102, 103]], ANTROUTE, MULTI_DESTINATION, [1, 1], [1, 1])
    with pytest.raises(ValueError, match="leg"):
        score_trip(wg, [DETOUR], ANTROUTE, RECOMMENDED, [1, 1], [1])


def test_score_trip_rejects_missing_reference_time():
    wg = weighted_risky_direct()
    with pytest.raises(ValueError, match="Google Maps"):
        score_trip(wg, [DETOUR], ANTROUTE, RECOMMENDED, [100.0], [math.nan])


def test_score_trip_rejects_a_route_not_on_the_graph():
    wg = weighted_risky_direct()
    with pytest.raises(KeyError):
        score_trip(wg, [[100, 102]], ANTROUTE, RECOMMENDED, [100.0], [110.0])


def test_waypoints_are_inner_nodes_spaced_along_the_route():
    wg = weighted_risky_direct()
    assert pick_waypoints(wg, DETOUR, 1) == [101]     # 300 m mark: 101 (200 m) and 102 (400 m) tie, first wins
    assert pick_waypoints(wg, DETOUR, 2) == [101, 102]
    assert pick_waypoints(wg, DETOUR, 10) == [101, 102]
    assert pick_waypoints(wg, DIRECT, 3) == []
    assert pick_waypoints(wg, DETOUR, 0) == []


def test_relative_difference_follows_equations_7_and_8():
    # Equation 7 (congestion and event exposure): lower is better, so a lower proposed value is positive
    assert relative_difference(200.0, 150.0) == pytest.approx(25.0)
    assert relative_difference(80.0, 90.0) == pytest.approx(-12.5)
    # Equation 8 (Route Optimality): higher is better, so a higher proposed value is positive
    assert relative_difference(80.0, 90.0, higher_is_better=True) == pytest.approx(12.5)
    assert relative_difference(0.8, 0.6, higher_is_better=True) == pytest.approx(-25.0)
    assert math.isnan(relative_difference(0.0, 1.0))


def test_paired_comparison_applies_the_right_equation_per_metric():
    better = [95.0, 97.0, 92.0]
    worse = [85.0, 88.0, 84.0]
    assert compare_paired("route_optimality", better, worse).relative_difference_pct > 0
    assert compare_paired("congestion_exposure", worse, better).relative_difference_pct > 0   # lower = better
    assert compare_paired("event_exposure", worse, better).relative_difference_pct > 0


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
    c = compare_paired("congestion_exposure", [1.0, 2.0, math.nan], [1.0, 2.0, 5.0])
    assert c.n == 2                                   # the NaN pair is dropped
    assert math.isnan(c.wilcoxon_p) and not c.significant
    assert math.isnan(c.shapiro_p) and c.t_value is None


def test_holm_adjusts_the_q3_family_and_never_lowers_a_p_value():
    from src.routing.significance import Q3_FAMILY, holm_adjust

    rng = np.random.default_rng(0)
    worse = rng.normal(5, 1, 40)
    better = worse - rng.normal(1.0, 0.5, 40)
    family = [compare_paired("congestion_exposure", better, worse), compare_paired("event_exposure", better, worse)]
    holm_adjust(family)
    assert Q3_FAMILY == ("congestion_exposure", "event_exposure")
    assert all(c.wilcoxon_p_holm >= c.wilcoxon_p for c in family)
    lo, hi = sorted(family, key=lambda c: c.wilcoxon_p)
    assert lo.wilcoxon_p_holm == pytest.approx(min(1.0, 2 * lo.wilcoxon_p))
    assert hi.wilcoxon_p_holm >= lo.wilcoxon_p_holm and all(c.significant_holm for c in family)


def test_holm_counts_a_tied_comparison_in_the_family_size():
    from src.routing.significance import holm_adjust

    rng = np.random.default_rng(1)
    worse = rng.normal(5, 1, 30)
    tested = compare_paired("congestion_exposure", worse - 1.0 - rng.random(30), worse)
    tied = compare_paired("event_exposure", [0.1] * 30, [0.1] * 30)
    holm_adjust([tested, tied])
    assert tested.wilcoxon_p_holm == pytest.approx(min(1.0, 2 * tested.wilcoxon_p))
    assert tied.wilcoxon_p_holm is None and tied.significant_holm is False
