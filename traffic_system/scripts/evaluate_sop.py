"""Answer the statement of the problem (SOP) with simulated trips of ANTROUTE and the Cheng (2023) baseline.

  SOP 1  Routing performance of the proposed system under selected traffic conditions:
         distance travelled and path cost.
  SOP 2  Computational performance of the proposed system: computational time, initial routing
         time, dynamic re-routing time and convergence rate (iterations to convergence).
  SOP 3  Significant difference between ANTROUTE and the baseline in distance travelled, path
         cost, computational time and convergence rate.

Both algorithms drive the same trips, departures, events and seeds through dynamic_trip.py. Path
cost is the CRITIC ten-feature cost of each road when entered, for both. Convergence rate is N_conv,
the iteration at which the initial search's best path cost had not improved for 20 consecutive
iterations (Aadil et al., 2016; fewer = faster). Both algorithms stop on this rule; a run that reaches
the iteration limit first is recorded as not converged, summarised by count, and its pair is left out
of the convergence-rate test.

SOP 3 follows the statistical treatment (Table 5). Results are paired per trip and scenario,
averaged over seeds, and the paired difference is d_i = C_baseline,i - C_ANTROUTE,i, so a positive
difference favours ANTROUTE for every measure. Each measure is tested with the two-sided Wilcoxon
signed-rank test (zero differences excluded, tied absolute differences given average ranks,
W = min(W+, W-)). The four p-values of each family (all conditions, and each traffic condition) are
Holm-adjusted, p_adj,(i) = min(1, max_{j<=i} (m - j + 1) p_(j)) with m = 4, and a difference is
significant when p_adj < alpha = 0.05.

Closures are placed on the shortest-distance route, which neither algorithm routes by; the closed
stretch is the one nearest the route's middle that still leaves a route from origin to destination.

Usage: python -m scripts.evaluate_sop [--trips 10] [--seeds 2]
       python -m scripts.evaluate_sop --from-runs   (re-analyse outputs/sop_runs.csv without re-running trips)
Writes outputs/sop_runs.csv, outputs/sop_results.json and outputs/sop_report.md.
"""
import argparse
import csv
import heapq
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import networkx as nx
from scipy import stats

from app.main import get_network
from app.study_policy import require_historical_release
from src.routing.aco_routing import CONVERGENCE_PATIENCE, AntColonyConfig
from src.routing.baseline_cheng import ChengConfig
from src.routing.dynamic_trip import TripEvent, simulate_trip

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
ALGORITHMS = ("antroute", "cheng")
NAMES = {"antroute": "ANTROUTE (proposed)", "cheng": "Cheng (2023) baseline"}
SCENARIOS = {
    "off_peak": "Off-peak, no change (Tue 10:10, historical weather)",
    "morning_peak_onset": "Morning peak begins during the trip (Tue 06:50)",
    "evening_peak_onset": "Evening peak begins during the trip (Tue 16:50)",
    "road_closure": "Road closure 2 min into the trip (Tue 08:10)",
    "heavy_rain_onset": "Clear to heavy rain 3 min into the trip (Tue 08:10)",
}
# measure: (label, unit, scale from the stored value)
MEASURES = {
    "distance_travel_m": ("Distance travelled", "km", 0.001),
    "path_cost": ("Path cost", "", 1),
    "computational_time_ms": ("Computational time", "s", 0.001),
    "initial_routing_time_ms": ("Initial routing time", "s", 0.001),
    "mean_rerouting_time_ms": ("Dynamic re-routing time (per re-route)", "s", 0.001),
    "dynamic_rerouting_time_ms": ("Dynamic re-routing time (per trip)", "s", 0.001),
    "reroutes": ("Re-routes per trip", "", 1),
    "iterations_to_convergence": ("Convergence rate: iterations to convergence (initial search)", "iterations", 1),
    "mean_iterations_to_convergence": ("Iterations to convergence (all searches)", "iterations", 1),
}
SOP = {1: ("distance_travel_m", "path_cost"),
       2: ("computational_time_ms", "initial_routing_time_ms", "mean_rerouting_time_ms", "dynamic_rerouting_time_ms",
           "reroutes", "iterations_to_convergence", "mean_iterations_to_convergence"),
       3: ("distance_travel_m", "path_cost", "computational_time_ms", "iterations_to_convergence")}


def scenario_setup(name, closure_roads):
    tuesday = datetime(2025, 6, 3)
    return {
        "off_peak": (tuesday.replace(hour=10, minute=10), "historical", []),
        "morning_peak_onset": (tuesday.replace(hour=6, minute=50), "historical", []),
        "evening_peak_onset": (tuesday.replace(hour=16, minute=50), "historical", []),
        "road_closure": (tuesday.replace(hour=8, minute=10), "historical",
                         [TripEvent(at_minute=2, close_road_ids=frozenset(closure_roads))]),
        "heavy_rain_onset": (tuesday.replace(hour=8, minute=10), "clear", [TripEvent(at_minute=3, weather="heavy_rain")]),
    }[name]


def shortest_distance_route(network, origin, destination):
    adjacency = {}
    for road in network.roads:
        adjacency.setdefault(road.source, []).append(road)
    best, previous, queue = {origin: 0.0}, {}, [(0.0, origin)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node == destination:
            break
        if cost > best[node]:
            continue
        for road in adjacency.get(node, []):
            if cost + road.distance_m < best.get(road.target, math.inf):
                best[road.target], previous[road.target] = cost + road.distance_m, road
                heapq.heappush(queue, (best[road.target], road.target))
    route, node = [], destination
    while node != origin:
        route.append(previous[node])
        node = previous[node].source
    return route[::-1]


def closure_on(network, route, origin, destination, length=3):
    """Consecutive roads nearest the middle of route whose closure still leaves origin -> destination."""
    adjacency = {}
    for road in network.roads:
        adjacency.setdefault(road.source, []).append(road)
    middle = len(route) // 2
    for start in sorted(range(max(len(route) - length + 1, 1)), key=lambda index: abs(index + length // 2 - middle)):
        closed = {road.id for road in route[start:start + length]}
        seen, pending = {origin}, [origin]
        while pending:
            for road in adjacency.get(pending.pop(), []):
                if road.id not in closed and road.target not in seen:
                    seen.add(road.target)
                    pending.append(road.target)
        if destination in seen:
            return closed
    raise ValueError(f"No closure on the shortest-distance route keeps {destination} reachable from {origin}")


def trip_pairs(network, count, rng):
    graph = nx.DiGraph((road.source, road.target) for road in network.roads)
    component = sorted(max(nx.strongly_connected_components(graph), key=len))
    nodes = {node.id: node for node in network.nodes}
    pairs = []
    while len(pairs) < count:
        origin, destination = rng.sample(component, 2)
        a, b = nodes[origin], nodes[destination]
        kilometres = math.hypot((a.longitude - b.longitude) * math.cos(math.radians(a.latitude)), a.latitude - b.latitude) * 111.195
        if 3 <= kilometres <= 10:
            pairs.append((origin, destination, kilometres))
    return pairs


def describe(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def compare(proposed, baseline):
    """Wilcoxon signed-rank test on the paired differences d_i = baseline_i - proposed_i."""
    differences = [b - a for a, b in zip(proposed, baseline)]
    result = {"n": len(differences), "proposed_mean": statistics.fmean(proposed), "baseline_mean": statistics.fmean(baseline),
              "mean_difference": statistics.fmean(differences), "median_difference": statistics.median(differences)}
    nonzero = [difference for difference in differences if difference != 0]
    if len(differences) < 3:
        return {**result, "test": "not tested (fewer than 3 pairs)", "p": None}
    if not nonzero:
        return {**result, "test": "not tested (no differences)", "n_nonzero": 0, "p": 1.0}
    # Zero differences excluded; tied absolute differences share their average rank.
    ranks = stats.rankdata([abs(difference) for difference in nonzero])
    positive = float(sum(rank for rank, difference in zip(ranks, nonzero) if difference > 0))
    negative = float(sum(rank for rank, difference in zip(ranks, nonzero) if difference < 0))
    test = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    return {**result, "test": "Wilcoxon signed-rank", "n_nonzero": len(nonzero), "W_plus": positive, "W_minus": negative,
            "W": min(positive, negative), "p": float(test.pvalue),
            "rank_biserial": (positive - negative) / (positive + negative)}


def holm(tests):
    """Holm step-down adjustment of one family: p_adj,(i) = min(1, max_{j<=i} (m - j + 1) p_(j))."""
    tested = sorted((test for test in tests if test["p"] is not None), key=lambda test: test["p"])
    running = 0.0
    for j, test in enumerate(tested, start=1):
        running = max(running, (len(tested) - j + 1) * test["p"])
        test["p_holm"] = min(1.0, running)
    for test in tests:
        test.setdefault("p_holm", None)
        test["significant"] = None if test["p_holm"] is None else test["p_holm"] < ALPHA


def number(value, scale=1, digits=2):
    return "–" if value is None else f"{value * scale:,.{digits}f}"


def report(results):
    lines = [f"# SOP results: ANTROUTE vs Cheng (2023)", "",
             f"Generated {results['generated_at']} · {results['trips']} trips × {len(SCENARIOS)} scenarios × "
             f"{results['seeds']} seeds · network built {results['network_built_at']}", "",
             results["note"], ""]
    for number_, title in ((1, "SOP 1. Routing performance under selected traffic conditions"),
                           (2, "SOP 2. Computational performance")):
        lines += [f"## {title}", ""]
        for measure in SOP[number_]:
            label, unit, scale = MEASURES[measure]
            lines += [f"### {label}{f' ({unit})' if unit else ''}", "",
                      "| Traffic condition | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |", "|---|---|---|"]
            for scenario in (*SCENARIOS, "all"):
                cells = []
                for algorithm in ALGORITHMS:
                    d = results["descriptive"][scenario][algorithm][measure]
                    cells.append("–" if not d["n"] else f"{number(d['mean'], scale)} ± {number(d['sd'], scale)} "
                                                        f"[{number(d['min'], scale)}–{number(d['max'], scale)}] (n={d['n']})")
                lines.append(f"| {SCENARIOS.get(scenario, 'All conditions')} | {cells[0]} | {cells[1]} |")
            if measure == "iterations_to_convergence":
                counts = results["not_converged"]["all"]
                lines += ["", f"Runs that reached the iteration limit without converging (not included above): "
                              f"ANTROUTE {counts['antroute']}, Cheng {counts['cheng']}."]
            lines.append("")
    lines += ["## SOP 3. Significant difference between ANTROUTE and the baseline", "",
              f"Paired by trip and traffic condition (seeds averaged). d = Cheng − ANTROUTE; positive favours ANTROUTE for every measure. "
              f"Two-sided Wilcoxon signed-rank test, W = min(W+, W−); the four p-values of each condition are Holm-adjusted. "
              f"Significant when the Holm-adjusted p < α = {ALPHA}.", ""]
    p_text = lambda p: "–" if p is None else "< 0.0001" if p < 0.0001 else f"{p:.4f}"
    for scenario in ("all", *SCENARIOS):
        lines += [f"### {SCENARIOS.get(scenario, 'All conditions')}", "",
                  "| Measure | n | ANTROUTE mean | Cheng mean | Mean d | W+ | W− | W | p | Holm p | Significant |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for measure in SOP[3]:
            label, unit, scale = MEASURES[measure]
            t = results["sop3"][scenario][measure]
            ranks = [f"{t[key]:.1f}" if key in t else "–" for key in ("W_plus", "W_minus", "W")]
            significant = {True: "**Yes**", False: "No", None: "–"}[t["significant"]]
            lines.append(f"| {label.split(':')[0]} | {t['n']} | {number(t.get('proposed_mean'), scale)} | "
                         f"{number(t.get('baseline_mean'), scale)} | {number(t.get('mean_difference'), scale)} | "
                         f"{' | '.join(ranks)} | {p_text(t['p'])} | {p_text(t['p_holm'])} | {significant} |")
        excluded = results["sop3"][scenario]["iterations_to_convergence"].get("pairs_excluded_not_converged")
        if excluded:
            lines += ["", f"Convergence rate: {excluded} pair(s) excluded because a run did not converge."]
        lines.append("")
    failed = [row for row in results["runs"] if row["status"] != "ok"]
    lines += ["## Runs that failed", "", "None." if not failed else
              "\n".join(f"- trip {row['trip']}, {row['scenario']}, seed {row['seed']}, {row['algorithm']}: {row['status']}" for row in failed), ""]
    return "\n".join(lines)


def analyse(rows):
    ok = [row for row in rows if row["status"] == "ok"]
    descriptive = {}
    for scenario in (*SCENARIOS, "all"):
        descriptive[scenario] = {algorithm: {measure: describe(
            row[measure] for row in ok if row["algorithm"] == algorithm and scenario in ("all", row["scenario"]))
            for measure in MEASURES} for algorithm in ALGORITHMS}
    # SOP 3 unit: one pair per trip and scenario, each side averaged over its successful seeds.
    units = {}
    for row in ok:
        units.setdefault((row["trip"], row["scenario"]), {}).setdefault(row["algorithm"], []).append(row)
    sop3 = {}
    for scenario in ("all", *SCENARIOS):
        sop3[scenario] = {}
        for measure in SOP[3]:
            proposed, baseline, excluded = [], [], 0
            for (trip, unit_scenario), sides in sorted(units.items()):
                if scenario in ("all", unit_scenario) and all(sides.get(algorithm) for algorithm in ALGORITHMS):
                    # A run that did not converge has no N_conv; a pair missing either side is excluded.
                    values = [[row[measure] for row in sides[algorithm] if row[measure] is not None] for algorithm in ALGORITHMS]
                    if not all(values):
                        excluded += 1
                        continue
                    proposed.append(statistics.fmean(values[0]))
                    baseline.append(statistics.fmean(values[1]))
            test = compare(proposed, baseline) if proposed else {"n": 0, "test": "no pairs", "p": None}
            sop3[scenario][measure] = {**test, "pairs_excluded_not_converged": excluded}
        holm(list(sop3[scenario].values()))
    not_converged = {scenario: {algorithm: sum(1 for row in ok if row["algorithm"] == algorithm and scenario in ("all", row["scenario"])
                                                and row["iterations_to_convergence"] is None) for algorithm in ALGORITHMS}
                     for scenario in (*SCENARIOS, "all")}
    return descriptive, sop3, not_converged


def load_runs(path):
    """Rows of a previous sop_runs.csv with their numbers restored."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["trip"], row["seed"] = int(row["trip"]), int(row["seed"])
        for key in (*MEASURES, "straight_line_km", "travel_time_s"):
            row[key] = float(row[key]) if row.get(key) else None
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trips", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--from-runs", action="store_true", help="re-analyse outputs/sop_runs.csv instead of running trips")
    args = parser.parse_args()
    output = ROOT / "outputs"
    if args.from_runs:
        rows = load_runs(output / "sop_runs.csv")
        previous = output / "sop_results.json"
        built_at = json.loads(previous.read_text(encoding="utf-8")).get("network_built_at") if previous.exists() else None
        write(output, rows, len({row["trip"] for row in rows}), len({row["seed"] for row in rows}), built_at or "unknown",
              save_runs=False)
        return
    require_historical_release()
    network = get_network()
    rows = []
    for trip, (origin, destination, kilometres) in enumerate(trip_pairs(network, args.trips, random.Random(2025))):
        closure_roads = closure_on(network, shortest_distance_route(network, origin, destination), origin, destination)
        for scenario in SCENARIOS:
            departure, weather, events = scenario_setup(scenario, closure_roads)
            for seed in range(args.seeds):
                for algorithm in ALGORITHMS:
                    row = {"trip": trip, "origin": origin, "destination": destination, "straight_line_km": round(kilometres, 3),
                           "scenario": scenario, "seed": seed, "algorithm": algorithm}
                    started = perf_counter()
                    try:
                        result = simulate_trip(network, origin, destination, departure, weather, frozenset(), events,
                                               algorithm, AntColonyConfig(seed=seed), ChengConfig(seed=seed))
                        evaluation = result.evaluation()
                        row.update({measure: evaluation.get(measure, evaluation["convergence"].get(measure)) for measure in MEASURES},
                                   travel_time_s=sum(result.travel_time_s), status="ok")
                    except Exception as exc:  # A failed run is recorded, not hidden.
                        row.update(status=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
                    print(f"trip {trip} {scenario:20} seed {seed} {algorithm:8} {row['status'][:40]:40} "
                          f"{perf_counter() - started:6.1f} s", flush=True)
    write(output, rows, args.trips, args.seeds, network.provenance["built_at"])


def write(output, rows, trips, seeds, network_built_at, save_runs=True):
    descriptive, sop3, not_converged = analyse(rows)
    results = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "network_built_at": network_built_at, "trips": trips, "seeds": seeds,
               "alpha": ALPHA, "test": "Wilcoxon signed-rank (two-sided), Holm-adjusted per condition",
               "difference": "baseline - ANTROUTE", "scenarios": SCENARIOS, "algorithms": NAMES,
               "measures": {measure: {"label": label, "unit": unit, "scale": scale} for measure, (label, unit, scale) in MEASURES.items()},
               "sop_measures": SOP,
               "note": "Path cost is the CRITIC ten-feature cost of each road when entered, for both algorithms. "
                       f"Convergence rate = N_conv, the iteration at which the initial search's best path cost had not "
                       f"improved for {CONVERGENCE_PATIENCE} consecutive iterations (Aadil et al., 2016); both algorithms stop "
                       f"on this rule, and a run that reaches the iteration limit first is recorded as not converged. "
                       "Traffic is modeled, not measured. Lower is better for every measure.",
               "descriptive": descriptive, "not_converged": not_converged, "sop3": sop3, "runs": rows}
    output.mkdir(exist_ok=True)
    if save_runs:
        fields = list(rows[0])
        for row in rows:
            fields += [key for key in row if key not in fields]
        with (output / "sop_runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output / "sop_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "sop_report.md").write_text(report(results), encoding="utf-8")
    print(f"Saved {len(rows)} runs{': outputs/sop_runs.csv,' if save_runs else ' (re-analysed):'} outputs/sop_results.json, outputs/sop_report.md")


if __name__ == "__main__":
    main()
