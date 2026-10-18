"""Answer the statement of the problem (SOP) with simulated trips of ANTROUTE and the Cheng (2023) baseline.

  SOP 1  Routing performance of the proposed system under selected traffic conditions:
         distance travelled and path cost.
  SOP 2  Computational performance of the proposed system: computational time, initial routing
         time, dynamic re-routing time and convergence rate (iterations to convergence).
  SOP 3  Significant difference between ANTROUTE and the baseline in distance travelled, path
         cost, computational time and convergence rate.

Both algorithms drive the same trips, departures, events and seeds through dynamic_trip.py. Path
cost is the CRITIC ten-feature cost of each road when entered, for both. Convergence rate is the
iteration at which the initial search first found its final best route (fewer = faster).

SOP 3 compares paired results per trip and scenario, averaged over seeds. For each measure the
paired differences are tested for normality (Shapiro-Wilk); if normal (p >= 0.05) a paired t-test
is used, otherwise the Wilcoxon signed-rank test. alpha = 0.05.

Closures are placed on the shortest-distance route, which neither algorithm routes by; the closed
stretch is the one nearest the route's middle that still leaves a route from origin to destination.

Usage: python -m scripts.evaluate_sop [--trips 10] [--seeds 2]
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
from src.routing.aco_routing import AntColonyConfig
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
    """Paired test of proposed - baseline: Shapiro-Wilk on the differences, then paired t or Wilcoxon."""
    differences = [a - b for a, b in zip(proposed, baseline)]
    result = {"n": len(differences), "proposed_mean": statistics.fmean(proposed), "baseline_mean": statistics.fmean(baseline),
              "mean_difference": statistics.fmean(differences), "median_difference": statistics.median(differences)}
    if len(differences) < 3:
        return {**result, "test": "not tested (fewer than 3 pairs)", "p": None, "significant": None}
    if all(abs(difference) < 1e-12 for difference in differences):
        return {**result, "test": "not tested (no differences)", "p": 1.0, "significant": False}
    normal_p = float(stats.shapiro(differences).pvalue)
    if normal_p >= ALPHA:
        test = stats.ttest_rel(proposed, baseline)
        sd = statistics.stdev(differences)
        effect = {"cohens_dz": result["mean_difference"] / sd if sd else None}
        name, statistic = "paired t-test", {"t": float(test.statistic), "df": len(differences) - 1}
    else:
        test = stats.wilcoxon(proposed, baseline)
        ranks = stats.rankdata([abs(difference) for difference in differences if difference])
        signed = [difference for difference in differences if difference]
        positive = sum(rank for rank, difference in zip(ranks, signed) if difference > 0)
        negative = sum(rank for rank, difference in zip(ranks, signed) if difference < 0)
        effect = {"rank_biserial": (positive - negative) / (positive + negative)}
        name, statistic = "Wilcoxon signed-rank", {"W": float(test.statistic)}
    p = float(test.pvalue)
    return {**result, "normality_shapiro_p": normal_p, "test": name, **statistic, "p": p,
            "significant": p < ALPHA, **effect}


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
                      "| Traffic condition | ANTROUTE mean ± SD | Cheng mean ± SD |", "|---|---|---|"]
            for scenario in (*SCENARIOS, "all"):
                cells = []
                for algorithm in ALGORITHMS:
                    d = results["descriptive"][scenario][algorithm][measure]
                    cells.append("–" if not d["n"] else f"{number(d['mean'], scale)} ± {number(d['sd'], scale)} (n={d['n']})")
                lines.append(f"| {SCENARIOS.get(scenario, 'All conditions')} | {cells[0]} | {cells[1]} |")
            lines.append("")
    lines += ["## SOP 3. Significant difference between ANTROUTE and the baseline", "",
              f"Paired by trip and traffic condition (seeds averaged). Difference = ANTROUTE − Cheng; negative favours ANTROUTE for every measure. α = {ALPHA}.", ""]
    for scenario in ("all", *SCENARIOS):
        lines += [f"### {SCENARIOS.get(scenario, 'All conditions')}", "",
                  "| Measure | n | ANTROUTE | Cheng | Mean difference | Test | Statistic | p | Significant |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for measure in SOP[3]:
            label, unit, scale = MEASURES[measure]
            t = results["sop3"][scenario][measure]
            statistic = (f"t({t['df']}) = {t['t']:.2f}" if "t" in t else f"W = {t['W']:.1f}" if "W" in t else "–")
            significant = {True: "**Yes**", False: "No", None: "–"}[t["significant"]]
            p = "–" if t["p"] is None else "< 0.0001" if t["p"] < 0.0001 else f"{t['p']:.4f}"
            lines.append(f"| {label.split(':')[0]} | {t['n']} | {number(t.get('proposed_mean'), scale)} | "
                         f"{number(t.get('baseline_mean'), scale)} | {number(t.get('mean_difference'), scale)} | "
                         f"{t['test']} | {statistic} | {p} | {significant} |")
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
            proposed, baseline = [], []
            for (trip, unit_scenario), sides in sorted(units.items()):
                if scenario in ("all", unit_scenario) and all(sides.get(algorithm) for algorithm in ALGORITHMS):
                    proposed.append(statistics.fmean(row[measure] for row in sides["antroute"]))
                    baseline.append(statistics.fmean(row[measure] for row in sides["cheng"]))
            sop3[scenario][measure] = compare(proposed, baseline) if proposed else {"n": 0, "test": "no pairs", "p": None, "significant": None}
    return descriptive, sop3


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trips", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
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
    descriptive, sop3 = analyse(rows)
    results = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "network_built_at": network.provenance["built_at"], "trips": args.trips, "seeds": args.seeds,
               "alpha": ALPHA, "scenarios": SCENARIOS, "algorithms": NAMES,
               "measures": {measure: {"label": label, "unit": unit, "scale": scale} for measure, (label, unit, scale) in MEASURES.items()},
               "sop_measures": SOP,
               "note": "Path cost is the CRITIC ten-feature cost of each road when entered, for both algorithms. "
                       "Convergence rate = iterations until the initial search first found its final best route. "
                       "Traffic is modeled, not measured. Lower is better for every measure.",
               "descriptive": descriptive, "sop3": sop3, "runs": rows}
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    fields = list(rows[0])
    for row in rows:
        fields += [key for key in row if key not in fields]
    with (output / "sop_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "sop_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "sop_report.md").write_text(report(results), encoding="utf-8")
    print(f"Saved {len(rows)} runs: outputs/sop_runs.csv, outputs/sop_results.json, outputs/sop_report.md")


if __name__ == "__main__":
    main()
