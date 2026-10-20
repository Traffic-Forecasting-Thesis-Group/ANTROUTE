"""Frozen-manifest evaluator: paired units, failures, Wilcoxon and Holm (m = 2)."""
import statistics

import pytest

from scripts.evaluate_sop import analyse, holm, static_run, wilcoxon

SEEDS = [0, 1, 2]


def rows_for(request, antroute, cheng, status="converged"):
    rows = []
    for algorithm, values in (("antroute", antroute), ("cheng", cheng)):
        for seed, (cost, time) in zip(SEEDS, values):
            rows.append({"experiment": "static", "request": request, "algorithm": algorithm, "seed": seed, "status": status,
                         "path_cost": cost, "search_time_ms": time, "distance_travel_m": 1000,
                         "iterations_to_convergence": 21, "first_best_found_iteration": 1})
    return rows


def test_units_are_matched_seed_medians_and_failures_are_kept_out_of_pairs():
    rows = []
    for index in range(8):
        rows += rows_for(f"r{index}", [(1 + index, 10), (2 + index, 30), (100, 20)], [(2 + index, 40), (3 + index, 50), (200, 60)])
    failed = rows_for("r9", [(1, 1)] * 3, [(1, 1)] * 3)
    failed[0].update(status="search_time_limit", path_cost=None, search_time_ms=None)  # One seed failed: no route.
    result = analyse(rows + failed, SEEDS)
    assert result["requests"] == 9 and result["complete_requests"] == 8 and result["incomplete_requests"] == ["r9"]
    assert result["outcomes"]["antroute"]["search_time_limit"] == 1  # Kept in the all-request report, not zeroed.
    primary = result["primary"]
    # Request r0: ANTROUTE median cost 2, Cheng median 3 -> d = 1 for every request.
    assert primary["path_cost"]["median_difference"] == pytest.approx(1)
    assert primary["path_cost"]["proposed_median"] == pytest.approx(statistics.median([2 + i for i in range(8)]))
    assert primary["search_time_ms"]["median_difference"] == pytest.approx(30)  # 50 - 20
    assert {measure: test["p_holm"] is not None for measure, test in primary.items()} == {"path_cost": True, "search_time_ms": True}


def test_all_zero_differences_are_handled_explicitly():
    result = wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0 + 1e-12], precision=1e-9)  # Below the declared precision.
    assert result["test"] == "not tested (all differences zero)" and result["p"] == 1.0
    assert wilcoxon([1.0, 2.0], [2.0, 3.0], precision=1e-9)["p"] is None  # Too few pairs.


def test_wilcoxon_statistic_and_holm_over_two_outcomes():
    proposed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    baseline = [2.0, 2.5, 5.0, 4.5, 9.0, 6.25, 8.0, 7.5]  # One negative difference (8 -> 7.5).
    result = wilcoxon(proposed, baseline, precision=1e-9)
    assert result["W_plus"] + result["W_minus"] == pytest.approx(8 * 9 / 2)
    assert result["W"] == min(result["W_plus"], result["W_minus"])
    family = [{"p": 0.01}, {"p": 0.04}]
    holm(family)
    assert [test["p_holm"] for test in family] == pytest.approx([0.02, 0.04])
    assert [test["significant"] for test in family] == [True, True]
    family = [{"p": 0.03}, {"p": 0.02}]
    holm(family)
    assert [test["p_holm"] for test in family] == pytest.approx([0.04, 0.04])  # Monotone step-down.


def test_static_run_records_common_cost_and_outcomes(network):
    request = {"id": "r0", "origin": "a", "destination": "d", "departure_date": "2025-06-02", "hour": 8, "weather": "clear"}
    antroute, cheng = static_run(network, request, "antroute", 0), static_run(network, request, "cheng", 0)
    assert antroute["status"] == cheng["status"] == "converged"
    assert antroute["path_cost"] == pytest.approx(antroute["native_cost"])  # ANTROUTE's native cost is the common cost.
    assert cheng["path_cost"] == pytest.approx(antroute["path_cost"])  # Same route here, scored by the same contract.
    blocked = static_run(network, {**request, "origin": "d", "destination": "a"}, "antroute", 0)
    assert blocked["status"] == "no_feasible_route" and "path_cost" not in blocked
