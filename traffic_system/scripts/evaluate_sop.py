"""Compare ANTROUTE with the Cheng (2023) baseline under a frozen experiment manifest.

Primary evaluation (static, one snapshot per request; revised protocol):
  outcomes     1. common evaluation path cost (ten-criterion contract cost of the returned route)
               2. ACO search time (ms, the kernel's own monotonic timing)
  unit         one request (origin, destination, test-period date and hour, historical weather);
               each algorithm's value is the median over the complete matched seed set
  difference   d_i = C_baseline,i - C_ANTROUTE,i (positive favours ANTROUTE; lower is better for both outcomes)
  test         two-sided Wilcoxon signed-rank, zero_method="wilcox" (zero differences dropped), tied absolute
               differences given average ranks, scipy method="auto" (exact when n <= 50 without ties or zeros,
               otherwise the normal approximation without continuity correction), W = min(W+, W-)
  precision    differences are rounded before ranking: path cost to 1e-9, search time to 1e-3 ms
  correction   Holm over the two primary p-values (m = 2); significant when the adjusted p < alpha = 0.05
Failures (no feasible route, search exhaustion, time limit without a route, unavailable data) stay in the
all-request report and are never converted to zero cost or time; a request enters the paired analysis only
when both algorithms returned a route for every seed. Execution order is counterbalanced by request and seed.
Distance travelled and convergence behaviour are secondary and descriptive. Dynamic trips are a separate,
descriptive experiment (--dynamic).

Usage (from traffic_system):
  python -m scripts.evaluate_sop freeze --requests 30 --seeds 5   writes configs/experiment_manifest.json
  python -m scripts.evaluate_sop run [--dynamic]                    runs the frozen manifest
  python -m scripts.evaluate_sop analyse                            re-analyses outputs/sop_runs.csv
Writes outputs/sop_runs.csv, outputs/sop_results.json and outputs/sop_report.md.
"""
import argparse
import csv
import json
import math
import random
import statistics
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import networkx as nx
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/experiment_manifest.json"
ALPHA = 0.05
ALGORITHMS = ("antroute", "cheng")
NAMES = {"antroute": "ANTROUTE (proposed)", "cheng": "Cheng (2023) baseline"}
PRIMARY = {"path_cost": ("Common evaluation path cost", "", 1e-9), "search_time_ms": ("ACO search time", "ms", 1e-3)}
SECONDARY = {"distance_travel_m": ("Distance travelled", "m"), "iterations_to_convergence": ("N_conv (converged runs)", "iterations"),
             "first_best_found_iteration": ("First iteration that found the final best route", "iterations")}
ROUTED = {"converged", "iteration_limit", "time_limit"}  # Statuses that returned a route.
DYNAMIC_SCENARIOS = {
    "morning_peak_onset": "Morning peak begins during the trip (06:50 departure)",
    "evening_peak_onset": "Evening peak begins during the trip (16:50 departure)",
    "road_closure": "Road closure 2 min into the trip (08:10 departure)",
    "heavy_rain_onset": "Clear to heavy rain 3 min into the trip (08:10 departure)",
}


# ---------------------------------------------------------------- analysis (pure)

def wilcoxon(proposed: list[float], baseline: list[float], precision: float) -> dict:
    """Two-sided Wilcoxon signed-rank on d = baseline - proposed, rounded to `precision` before ranking."""
    differences = [round((b - a) / precision) * precision for a, b in zip(proposed, baseline)]
    result = {"n": len(differences), "proposed_median": statistics.median(proposed) if proposed else None,
              "baseline_median": statistics.median(baseline) if baseline else None,
              "mean_difference": statistics.fmean(differences) if differences else None,
              "median_difference": statistics.median(differences) if differences else None}
    nonzero = [d for d in differences if d != 0]
    if len(differences) < 3:
        return {**result, "test": "not tested (fewer than 3 complete pairs)", "p": None}
    if not nonzero:
        return {**result, "test": "not tested (all differences zero)", "n_nonzero": 0, "p": 1.0}
    ranks = stats.rankdata([abs(d) for d in nonzero])
    positive = float(sum(r for r, d in zip(ranks, nonzero) if d > 0))
    negative = float(sum(r for r, d in zip(ranks, nonzero) if d < 0))
    test = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto", correction=False)
    return {**result, "test": "Wilcoxon signed-rank (two-sided)", "n_nonzero": len(nonzero), "W_plus": positive,
            "W_minus": negative, "W": min(positive, negative), "p": float(test.pvalue),
            "rank_biserial": (positive - negative) / (positive + negative)}


def holm(tests: list[dict], alpha: float = ALPHA) -> None:
    """Holm step-down: p_adj,(i) = min(1, max_{j<=i} (m - j + 1) p_(j)) over the tested family."""
    tested = sorted((test for test in tests if test["p"] is not None), key=lambda test: test["p"])
    running = 0.0
    for j, test in enumerate(tested, start=1):
        running = max(running, (len(tested) - j + 1) * test["p"])
        test["p_holm"] = min(1.0, running)
    for test in tests:
        test.setdefault("p_holm", None)
        test["significant"] = None if test["p_holm"] is None else test["p_holm"] < alpha


def describe(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def analyse(rows: list[dict], seeds: list[int]) -> dict:
    """Primary paired analysis over complete requests, plus all-request outcome counts and secondary summaries."""
    static = [row for row in rows if row.get("experiment", "static") == "static"]
    by_request: dict[str, dict[str, dict[int, dict]]] = {}
    for row in static:
        by_request.setdefault(row["request"], {}).setdefault(row["algorithm"], {})[int(row["seed"])] = row
    outcomes = {algorithm: {} for algorithm in ALGORITHMS}
    for row in static:
        outcomes[row["algorithm"]][row["status"]] = outcomes[row["algorithm"]].get(row["status"], 0) + 1
    complete, incomplete = {}, {}
    for request, sides in sorted(by_request.items()):
        ok = all(all(seed in sides.get(algorithm, {}) and sides[algorithm][seed]["status"] in ROUTED for seed in seeds)
                 for algorithm in ALGORITHMS)
        (complete if ok else incomplete)[request] = sides
    primary = {}
    for measure, (_, _, precision) in PRIMARY.items():
        medians = {algorithm: [statistics.median(float(sides[algorithm][seed][measure]) for seed in seeds)
                               for sides in complete.values()] for algorithm in ALGORITHMS}
        primary[measure] = wilcoxon(medians["antroute"], medians["cheng"], precision)
    holm(list(primary.values()))
    secondary = {measure: {algorithm: describe(float(row[measure]) if row.get(measure) not in (None, "") else None
                                               for row in static if row["algorithm"] == algorithm and row["status"] in ROUTED)
                           for algorithm in ALGORITHMS} for measure in SECONDARY}
    stop_reasons = {algorithm: {} for algorithm in ALGORITHMS}
    for row in static:
        if row["status"] in ROUTED:
            stop_reasons[row["algorithm"]][row["status"]] = stop_reasons[row["algorithm"]].get(row["status"], 0) + 1
    return {"requests": len(by_request), "complete_requests": len(complete), "incomplete_requests": sorted(incomplete),
            "seeds": seeds, "outcomes": outcomes, "stop_reasons": stop_reasons, "primary": primary, "secondary": secondary}


# ---------------------------------------------------------------- manifest

def _connected_pairs(network, count, rng):
    graph = nx.DiGraph((road.source, road.target) for road in network.roads)
    component = sorted(max(nx.strongly_connected_components(graph), key=len))
    nodes = {node.id: node for node in network.nodes}
    pairs = []
    while len(pairs) < count:
        origin, destination = rng.sample(component, 2)
        a, b = nodes[origin], nodes[destination]
        kilometres = math.hypot((a.longitude - b.longitude) * math.cos(math.radians(a.latitude)), a.latitude - b.latitude) * 111.195
        if 3 <= kilometres <= 10:
            pairs.append((origin, destination, round(kilometres, 3)))
    return pairs


def freeze(requests: int, seeds: int, force: bool) -> None:
    """Freeze OD pairs, held-out test-period snapshots, seeds and order against the current release."""
    from app.main import get_network
    from app.study_policy import require_historical_release
    from src.routing.calibration import days, load_split
    if MANIFEST.exists() and not force:
        sys.exit(f"{MANIFEST.relative_to(ROOT)} already exists; it is frozen. Pass --force only to replace it deliberately.")
    policy = json.loads((ROOT / "configs/historical_study.json").read_text(encoding="utf-8"))
    split = load_split(policy)  # Blocked until declared.
    require_historical_release()
    network = get_network()
    rng = random.Random(2025)
    test_days = list(days(split["test"]))
    pairs = _connected_pairs(network, requests, rng)
    manifest = {
        "manifest_version": 1, "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release": {"calibration_id": network.cost_config.calibration_id, "contract_version": network.cost_config.contract_version},
        "seeds": list(range(seeds)),
        "order": "counterbalanced: ANTROUTE runs first when (request index + seed) is even, Cheng first otherwise",
        "requests": [{"id": f"r{index:03d}", "origin": origin, "destination": destination, "straight_line_km": km,
                      "departure_date": rng.choice(test_days).isoformat(), "hour": rng.randrange(24), "weather": "historical"}
                     for index, (origin, destination, km) in enumerate(pairs)],
        "analysis": {"primary": list(PRIMARY), "difference": "baseline - ANTROUTE", "alpha": ALPHA,
                     "test": "two-sided Wilcoxon signed-rank", "zero_method": "wilcox", "method": "auto", "correction": False,
                     "precision": {measure: precision for measure, (_, _, precision) in PRIMARY.items()},
                     "unit_summary": "median over the complete matched seed set", "multiplicity": "Holm, m = 2"},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Froze {len(manifest['requests'])} requests x {seeds} seeds -> {MANIFEST.relative_to(ROOT)}")


# ---------------------------------------------------------------- runs

def static_run(network, request: dict, algorithm: str, seed: int) -> dict:
    """One search on one snapshot. Failures are recorded with their outcome, never as zero."""
    from time import perf_counter
    from src.routing.aco_kernel import NoFeasibleRoute, SearchExhausted, SearchTimeLimit
    from src.routing.aco_routing import AntColonyConfig, ant_colony_route, distance_reference, node_positions
    from src.routing.baseline_cheng import ChengConfig, cheng_graph, cheng_route
    from src.routing.condition_update import Conditions
    from src.routing.criteria import CriterionInputError
    from src.routing.dynamic_weight import prepare_graph, segment_values
    from src.routing.historical_traffic import traffic_for
    from src.routing.historical_weather import historical_snapshot
    from src.routing.weather import MissingWeatherObservation
    row = {"experiment": "static", "request": request["id"], "algorithm": algorithm, "seed": seed}
    conditions = Conditions(departure_date=date.fromisoformat(request["departure_date"]), time_of_day=request["hour"],
                            weather=request["weather"])
    try:
        weather = historical_snapshot(conditions.departure_date, conditions.time_of_day, network)[0] if conditions.weather == "historical" else None
        traffic = traffic_for(network, conditions)
        started = perf_counter()
        if algorithm == "cheng":
            config = ChengConfig(seed=seed)
            result = cheng_route(cheng_graph(network, conditions, weather, traffic), request["origin"], request["destination"],
                                 config, node_positions(network), distance_reference(network, "cheng"), None, started)
        else:
            config = AntColonyConfig(seed=seed)
            result = ant_colony_route(prepare_graph(network, conditions, weather, traffic), request["origin"], request["destination"],
                                      config, node_positions(network), distance_reference(network, "antroute"), started)
    except NoFeasibleRoute:
        return {**row, "status": "no_feasible_route"}
    except SearchTimeLimit:
        return {**row, "status": "search_time_limit"}
    except SearchExhausted:
        return {**row, "status": "search_exhausted"}
    except (CriterionInputError, MissingWeatherObservation, KeyError, ValueError) as error:
        return {**row, "status": f"data_unavailable: {error}"}
    nodes = {node.id: node for node in network.nodes}
    common = sum(segment_values(network, edge.road, conditions, nodes, weather, traffic).cost for edge in result.roads)
    return {**row, "status": result.stop_reason, "path_cost": common, "native_cost": sum(edge.cost for edge in result.roads),
            "search_time_ms": result.search_time_ms, "distance_travel_m": sum(edge.road.distance_m for edge in result.roads),
            "iterations_to_convergence": result.converged_at, "first_best_found_iteration": result.first_best_found_iteration,
            "iterations_completed": result.iterations_completed}


def shortest_distance_closure(network, request: dict) -> frozenset:
    """The middle road of the shortest-distance route, which neither algorithm routes by; same closure for both."""
    from src.routing.aco_kernel import exact_shortest_path
    from src.routing.dynamic_weight import WeightedRoad
    adjacency = {node.id: [] for node in network.nodes}
    for road in network.roads:
        adjacency[road.source].append(WeightedRoad(road, road.distance_m))
    route = exact_shortest_path(adjacency, request["origin"], request["destination"]) or []
    return frozenset({route[len(route) // 2].road.id}) if route else frozenset()


def dynamic_runs(network, manifest: dict) -> list[dict]:
    """Secondary, descriptive: the same trips under scenario changes, both algorithms, common trigger policy."""
    from src.routing.condition_update import MANILA
    from src.routing.dynamic_trip import ReplanConfig, TripEvent, simulate_trip
    rows = []
    for index, request in enumerate(manifest["requests"]):
        day = date.fromisoformat(request["departure_date"])
        for scenario in DYNAMIC_SCENARIOS:
            hour, minute = {"morning_peak_onset": (6, 50), "evening_peak_onset": (16, 50)}.get(scenario, (8, 10))
            weather, events = "historical", []
            if scenario == "heavy_rain_onset":
                weather, events = "clear", [TripEvent(at_minute=3, weather="heavy_rain")]
            for seed in manifest["seeds"]:
                for algorithm in (ALGORITHMS if (index + seed) % 2 == 0 else ALGORITHMS[::-1]):
                    row = {"experiment": f"dynamic:{scenario}", "request": request["id"], "algorithm": algorithm, "seed": seed}
                    departure = datetime(day.year, day.month, day.day, hour, minute, tzinfo=MANILA)
                    try:
                        trip_events = events
                        if scenario == "road_closure":
                            trip_events = [TripEvent(at_minute=2, close_road_ids=shortest_distance_closure(network, request))]
                        from src.routing.aco_routing import AntColonyConfig
                        from src.routing.baseline_cheng import ChengConfig
                        result = simulate_trip(network, request["origin"], request["destination"], departure, weather, frozenset(), trip_events,
                                               algorithm, AntColonyConfig(seed=seed), ChengConfig(seed=seed), ReplanConfig())
                        evaluation = result.evaluation()
                        rows.append({**row, "status": evaluation["convergence"]["stop_reason"], "path_cost": evaluation["path_cost"],
                                     "search_time_ms": evaluation["search_time_ms"], "distance_travel_m": evaluation["distance_travel_m"],
                                     "recalculation_time_ms": evaluation["recalculation_time_ms"], "reroutes": evaluation["reroutes"],
                                     "routes_adopted": evaluation["routes_adopted"]})
                    except Exception as error:  # A failed trip is recorded with its type, never hidden or zeroed.
                        rows.append({**row, "status": f"{type(error).__name__}: {error}"})
    return rows


def run(dynamic: bool) -> list[dict]:
    from app.main import get_network
    from app.study_policy import require_historical_release
    if not MANIFEST.exists():
        sys.exit("No frozen manifest. Run: python -m scripts.evaluate_sop freeze")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require_historical_release()
    network = get_network()
    if network.cost_config.calibration_id != manifest["release"]["calibration_id"]:
        sys.exit("The release calibration differs from the frozen manifest; re-freeze deliberately or restore the release.")
    rows = []
    for index, request in enumerate(manifest["requests"]):
        for seed in manifest["seeds"]:
            order = ALGORITHMS if (index + seed) % 2 == 0 else ALGORITHMS[::-1]  # Counterbalanced.
            for position, algorithm in enumerate(order):
                rows.append({**static_run(network, request, algorithm, seed), "order_position": position})
                print(f"{request['id']} seed {seed} {algorithm:8} {rows[-1]['status'][:40]}", flush=True)
    if dynamic:
        rows += dynamic_runs(network, manifest)
    return rows


# ---------------------------------------------------------------- reporting

def report(results: dict) -> str:
    p_text = lambda p: "–" if p is None else "< 0.0001" if p < 0.0001 else f"{p:.4f}"
    number = lambda value, digits=4: "–" if value is None else f"{value:,.{digits}f}"
    analysis = results["analysis"]
    lines = ["# ANTROUTE vs Cheng (2023): frozen-manifest comparison", "",
             f"Generated {results['generated_at']} · manifest {results['manifest']['frozen_at']} · calibration "
             f"{results['manifest']['calibration_id'][:12]} · scipy {results['scipy_version']}", "",
             "Software verification and modeled-scenario evaluation only: traffic is an assumed congestion scenario, "
             "so these results are not external empirical validation.", "",
             "## Primary outcomes", "",
             f"Requests: {analysis['requests']}; complete matched pairs: {analysis['complete_requests']} "
             f"(seeds {analysis['seeds']}, median per request). d = Cheng − ANTROUTE; positive favours ANTROUTE. "
             f"Two-sided Wilcoxon signed-rank, Holm over the two primary outcomes, α = {ALPHA}.", "",
             "| Outcome | n | ANTROUTE median | Cheng median | Median d | W+ | W− | W | p | Holm p | Significant |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for measure, (label, unit, _) in PRIMARY.items():
        t = analysis["primary"][measure]
        ranks = " | ".join(f"{t[key]:.1f}" if key in t else "–" for key in ("W_plus", "W_minus", "W"))
        significant = {True: "**Yes**", False: "No", None: "–"}[t["significant"]]
        lines.append(f"| {label}{f' ({unit})' if unit else ''} | {t['n']} | {number(t['proposed_median'])} | {number(t['baseline_median'])} | "
                     f"{number(t['median_difference'])} | {ranks} | {p_text(t['p'])} | {p_text(t['p_holm'])} | {significant} |")
    lines += ["", "## All requests: outcomes", "", "| Algorithm | Outcome | Runs |", "|---|---|---|"]
    for algorithm, counts in analysis["outcomes"].items():
        lines += [f"| {NAMES[algorithm]} | {status} | {count} |" for status, count in sorted(counts.items())]
    if analysis["incomplete_requests"]:
        lines += ["", f"Requests excluded from the paired analysis (a failure on some seed): {', '.join(analysis['incomplete_requests'])}."]
    lines += ["", "## Secondary (descriptive)", "", "| Measure | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |", "|---|---|---|"]
    for measure, (label, unit) in SECONDARY.items():
        cells = []
        for algorithm in ALGORITHMS:
            d = analysis["secondary"][measure][algorithm]
            cells.append("–" if not d["n"] else f"{number(d['mean'], 2)} ± {number(d['sd'], 2)} [{number(d['min'], 2)}–{number(d['max'], 2)}] (n={d['n']})")
        lines.append(f"| {label} ({unit}) | {cells[0]} | {cells[1]} |")
    lines += ["", "Stop reasons of runs that returned a route: " + "; ".join(
        f"{NAMES[a]}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) for a, c in analysis["stop_reasons"].items())]
    return "\n".join(lines) + "\n"


def dynamic_summary(rows: list[dict]) -> dict:
    """Descriptive dynamic-trip results by scenario and algorithm; failed trips are counted, not averaged."""
    summary = {}
    for scenario in DYNAMIC_SCENARIOS:
        summary[scenario] = {}
        for algorithm in ALGORITHMS:
            selected = [row for row in rows if row["experiment"] == f"dynamic:{scenario}" and row["algorithm"] == algorithm]
            routed = [row for row in selected if row["status"] in ROUTED]
            summary[scenario][algorithm] = {
                "runs": len(selected), "failed": len(selected) - len(routed),
                **{measure: describe(float(row[measure]) for row in routed) for measure in ("path_cost", "recalculation_time_ms")}}
    return summary


def write(rows: list[dict]) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "scipy_version": scipy.__version__,
               "manifest": {"frozen_at": manifest["frozen_at"], **manifest["release"]}, "alpha": ALPHA,
               "primary_measures": {measure: {"label": label, "unit": unit, "precision": precision} for measure, (label, unit, precision) in PRIMARY.items()},
               "algorithms": NAMES, "analysis": analyse(rows, manifest["seeds"]), "dynamic": dynamic_summary(rows), "runs": rows}
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    fields = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with (output / "sop_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "sop_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "sop_report.md").write_text(report(results), encoding="utf-8")
    print(f"Saved {len(rows)} runs: outputs/sop_runs.csv, outputs/sop_results.json, outputs/sop_report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--requests", type=int, default=30)
    frozen.add_argument("--seeds", type=int, default=5)
    frozen.add_argument("--force", action="store_true")
    running = commands.add_parser("run")
    running.add_argument("--dynamic", action="store_true", help="also run the descriptive dynamic-trip experiment")
    commands.add_parser("analyse")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.requests, args.seeds, args.force)
    elif args.command == "run":
        write(run(args.dynamic))
    else:
        with (ROOT / "outputs/sop_runs.csv").open(encoding="utf-8") as stream:
            write(list(csv.DictReader(stream)))


if __name__ == "__main__":
    main()
