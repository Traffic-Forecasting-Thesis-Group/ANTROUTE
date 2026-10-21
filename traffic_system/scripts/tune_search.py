"""Tune the shared ACO search settings on the tuning period (both algorithms, one setting).

Declared before running:
  requests   12 origin-destination pairs (3-10 km straight line, seed 2026) on random tuning-period dates and hours,
             historical weather; never development or test dates
  grid       q0 in {0, 0.3, 0.5, 0.7} x beta in {1, 2, 3, 5}; every other setting and the heuristic form stay declared
  measure    optimality gap = ACO route cost / exact lowest cost - 1, each algorithm under its own native objective
             (exact Dijkstra reference on the same snapshot graph; never given to ACO, never timed with it)
  seeds      0 and 1; the wall-time budget is disabled during tuning so time limits cannot bias the gap
  selection  lowest mean gap pooled over both algorithms; ties (within 1e-9) go to the lower median search time
  budget     wall_time_ms = 2 x the longest search time of the chosen setting, rounded up to the next second
Writes outputs/search_tuning.json. The chosen values are then frozen as SearchConfig defaults (aco_kernel.py).

--live runs the same rule on the live release: the requests take random saved captures of the live tuning
period (configs/live_study.json) instead of historical dates and hours, and the result goes to
outputs/live_search_tuning.json.

Run from traffic_system: python -m scripts.tune_search [--live]
"""
import argparse
import json
import math
import random
import statistics
from datetime import date, datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q0 = (0.0, 0.3, 0.5, 0.7)
BETA = (1.0, 2.0, 3.0, 5.0)
SEEDS = (0, 1)
REQUESTS = 12


def live_cases(rng):
    """Live release and requests on random saved captures of the live tuning period."""
    from app.main import get_live_network
    from scripts.calibrate_live import POLICY, load_split, period_of
    from scripts.evaluate_sop import _connected_pairs
    from src.live.snapshot import captures
    from src.routing.condition_update import live_conditions
    network = get_live_network()
    if network.cost_config.calibration_id.startswith("provisional-"):
        raise SystemExit("The live release is provisional; run python -m scripts.calibrate_live after the development week.")
    split = load_split(json.loads(POLICY.read_text(encoding="utf-8")))
    moments = [moment for moment in captures() if period_of(moment, split) == "tuning"]
    if not moments:
        raise SystemExit(f"No saved captures in the live tuning period {split['tuning'][0]} to {split['tuning'][1]}.")
    pairs = _connected_pairs(network, REQUESTS, rng)
    requests = []
    for origin, destination, km in pairs:
        moment = rng.choice(moments)
        requests.append({"origin": origin, "destination": destination, "straight_line_km": km, "live_at": moment.isoformat(),
                         "conditions": live_conditions(moment)})
    return network, requests, [d.isoformat() for d in split["tuning"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="tune on the live release and live tuning captures")
    live = parser.parse_args(argv).live
    from app.main import get_network
    from app.study_policy import require_historical_release
    from scripts.evaluate_sop import _connected_pairs
    from src.routing.aco_kernel import SearchConfig, aco_search, exact_shortest_path
    from src.routing.aco_routing import distance_reference, node_positions
    from src.routing.baseline_cheng import cheng_graph
    from src.routing.calibration import days, load_split
    from src.routing.condition_update import Conditions
    from src.routing.dynamic_weight import prepare_graph
    from src.routing.historical_traffic import traffic_for
    from src.routing.historical_weather import historical_snapshot

    rng = random.Random(2026)
    if live:
        network, requests, period = live_cases(rng)
    else:
        split = load_split(json.loads((ROOT / "configs/historical_study.json").read_text(encoding="utf-8")))
        require_historical_release()
        network = get_network()
        tuning_days = list(days(split["tuning"]))
        period = [d.isoformat() for d in split["tuning"]]
        pairs = _connected_pairs(network, REQUESTS, rng)
        requests = [{"origin": o, "destination": d, "straight_line_km": km, "departure_date": rng.choice(tuning_days).isoformat(),
                     "hour": rng.randrange(24)} for o, d, km in pairs]
    positions = node_positions(network)
    cases = []
    for request in requests:
        if live:
            conditions, weather, traffic = request.pop("conditions"), None, None
        else:
            conditions = Conditions(departure_date=date.fromisoformat(request["departure_date"]), time_of_day=request["hour"],
                                    weather="historical")
            weather = historical_snapshot(conditions.departure_date, conditions.time_of_day, network)[0]
            traffic = traffic_for(network, conditions)
        for model, graph in (("antroute", prepare_graph(network, conditions, weather, traffic)),
                             ("cheng", cheng_graph(network, conditions, weather, traffic))):
            exact = exact_shortest_path(graph, request["origin"], request["destination"])
            cases.append((request, model, graph, sum(edge.cost for edge in exact), distance_reference(network, model)))
    print(f"{len(requests)} tuning requests, {len(cases)} model graphs", flush=True)
    grid = []
    for q0, beta in product(Q0, BETA):
        runs = []
        for request, model, graph, optimum, reference in cases:
            for seed in SEEDS:
                config = SearchConfig(q0=q0, beta=beta, seed=seed, wall_time_ms=None)
                result = aco_search(graph, request["origin"], request["destination"], config, positions, reference)
                cost = sum(edge.cost for edge in result.roads)
                runs.append({"model": model, "gap": cost / optimum - 1, "search_time_ms": result.search_time_ms,
                             "stop_reason": result.stop_reason, "iterations": result.iterations_completed})
        summary = {"q0": q0, "beta": beta, "mean_gap": statistics.fmean(run["gap"] for run in runs),
                   **{f"mean_gap_{m}": statistics.fmean(r["gap"] for r in runs if r["model"] == m) for m in ("antroute", "cheng")},
                   "optimal_share": sum(run["gap"] < 1e-9 for run in runs) / len(runs),
                   "median_search_time_ms": statistics.median(run["search_time_ms"] for run in runs),
                   "max_search_time_ms": max(run["search_time_ms"] for run in runs),
                   "stop_reasons": {reason: sum(run["stop_reason"] == reason for run in runs) for reason in {r["stop_reason"] for r in runs}}}
        grid.append(summary)
        print(f"q0={q0} beta={beta}: mean gap {summary['mean_gap']:.3f} (ANTROUTE {summary['mean_gap_antroute']:.3f}, "
              f"Cheng {summary['mean_gap_cheng']:.3f}), optimal {summary['optimal_share']:.0%}, "
              f"median {summary['median_search_time_ms'] / 1000:.2f} s", flush=True)
    best = min(grid, key=lambda item: (round(item["mean_gap"], 9), item["median_search_time_ms"]))
    budget = math.ceil(2 * best["max_search_time_ms"] / 1000) * 1000
    result = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "mode": "live" if live else "historical",
              "calibration_id": network.cost_config.calibration_id, "tuning_period": period,
              "rule": __doc__.split("Declared before running:")[1].split("Writes")[0].strip(),
              "requests": requests, "seeds": list(SEEDS), "grid": grid,
              "chosen": {"q0": best["q0"], "beta": best["beta"], "wall_time_ms": budget}}
    target = ROOT / ("outputs/live_search_tuning.json" if live else "outputs/search_tuning.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Chosen: q0={best['q0']}, beta={best['beta']}, wall_time_ms={budget} -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
