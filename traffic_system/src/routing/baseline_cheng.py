"""Baseline: the improved ant colony algorithm of Cheng (2023), on the shared search kernel.

J. Cheng, "Dynamic Path Optimization Based on Improved Ant Colony Algorithm",
Journal of Advanced Transportation, 2023, 7651100. https://doi.org/10.1155/2023/7651100

Runs on the same road graph, eligibility (closures), date/hour scenario traffic and weather as ANTROUTE,
with the paper's own edge cost:
  - Eq. 5-6: s_ij = 0.637 * length/L_ref + 0.258 * travel time/T_ref + 0.105 * flow/F_ref, with the declared
    rounded AHP weights of Table 3 and fixed development references (metres, seconds, veh/h). References never
    depend on which roads are open, so an unrelated closure leaves every remaining road's score unchanged.
  - Eq. 8, 10-14 and Step 7: the shared kernel (aco_kernel.py) with Cheng's published q0, alpha, beta, rho, xi.

Inputs and adaptations, stated so results are not over-read:
  - Traffic flow: no vehicle counts or independent demand exist. Flow is a simulation adaptation: the assumed
    congestion scenario's BPR index is inverted (x) and multiplied by capacity. It is not a measurement and not
    independent demand; results label it "simulation_adaptation_inverse_bpr".
  - d_jD is divided by a frozen development reference (the longest development road), the divisor of the
    length term, so d_ij and d_jD share one scale as eq. 7 requires. The paper does not give it.
  - xi (global evaporation) is not given in the paper; 0.3, equal to rho.
  - Ants: the paper uses 1.5 x nodes (18,597 here); 20 keeps run time comparable, as for ANTROUTE.
  - Stopping: the study's shared convergence rule and time budget within NC_max (the paper runs all NC_max).
  - Dead ends: ants only enter nodes that can still reach D and step back from dead ends, as for ANTROUTE.
  - Eq. 9 (pheromone rescaling when road information changes) is applied, as printed, only in the separately
    labelled mode "cheng_reference_printed": a road that becomes slower gains pheromone there. The controlled
    comparison re-plans from fresh pheromone, like ANTROUTE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from .aco_kernel import SearchConfig, SearchResult, aco_search
from .condition_update import Conditions, validate_conditions
from .dynamic_weight import WeightedRoad, segment_values
from .network import RoadNetwork, WeatherEffect

REFERENCE = {"name": "Improved ant colony algorithm (Cheng, 2023)", "doi": "10.1155/2023/7651100",
             "citation": "J. Cheng, Dynamic Path Optimization Based on Improved Ant Colony Algorithm, "
                         "Journal of Advanced Transportation, 2023, 7651100.",
             "flow_provenance": "simulation_adaptation_inverse_bpr"}
# Table 3: path length, travel time and traffic flow compared on Saaty's scale (Table 1).
JUDGMENT = ((1, 3, 5), (1 / 3, 1, 3), (1 / 5, 1 / 3, 1))
# Table 2: average random consistency index by matrix order.
RANDOM_INDEX = (0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
# The declared, rounded weights used for routing (the paper's reported values).
AHP_WEIGHTS = (0.637, 0.258, 0.105)
Mode = Literal["controlled", "cheng_reference_printed"]


class ChengConfig(SearchConfig):
    """Shared search settings; mode selects fresh pheromone (controlled) or the printed eq. 9 (reference)."""
    mode: Mode = "controlled"


BaselineResult = SearchResult


@dataclass(frozen=True)
class Ahp:
    weights: tuple[float, ...]
    lambda_max: float
    consistency_ratio: float


def ahp(matrix=JUDGMENT) -> Ahp:
    """Sec. 3.2.3 steps 1-4 (row geometric mean) and the consistency ratio of eq. 4 (reported; routing uses AHP_WEIGHTS)."""
    n = len(matrix)
    roots = [math.prod(row) ** (1 / n) for row in matrix]
    weights = [root / sum(roots) for root in roots]
    lambda_max = sum(sum(a * w for a, w in zip(row, weights)) / (n * weight) for row, weight in zip(matrix, weights))
    ratio = (lambda_max - n) / (n - 1) / RANDOM_INDEX[n - 1] if n > 2 else 0.0
    return Ahp(tuple(weights), lambda_max, ratio)


def estimated_flow_vph(congestion: float, capacity_vph: float, bpr_alpha: float, bpr_beta: float) -> float:
    """Simulation adaptation: invert congestion = 1 - 1/(1 + a x^b) for x, then x * capacity. Not a measured flow."""
    congestion = min(congestion, 1 - 1e-9)
    if congestion <= 0:
        return 0.0
    return ((1 / (1 - congestion) - 1) / bpr_alpha) ** (1 / bpr_beta) * capacity_vph


def references(network: RoadNetwork) -> tuple[float, float, float]:
    """Fixed development references (m, s, veh/h) from the calibration."""
    try:
        frozen = network.provenance["cheng_references"]
        values = (float(frozen["distance_m"]), float(frozen["travel_time_s"]), float(frozen["flow_vph"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("The network has no frozen Cheng references; rebuild the calibrated release") from None
    if not all(value > 0 for value in values):
        raise ValueError("Cheng references must be positive")
    return values


def comprehensive_costs(criteria: list[tuple[float, float, float]], fixed: tuple[float, float, float],
                        weights: tuple[float, float, float] = AHP_WEIGHTS) -> list[float]:
    """Eq. 5-6 for (length m, travel time s, flow veh/h) rows, each divided by its fixed development reference."""
    return [sum(w * value / reference for w, value, reference in zip(weights, row, fixed)) for row in criteria]


def cheng_graph(network: RoadNetwork, conditions: Conditions, weather_by_city: dict[str, WeatherEffect] | None = None,
                traffic: dict[str, tuple[float, float]] | None = None) -> dict[str, list[WeightedRoad]]:
    """Open roads weighted by the comprehensive cost s_ij instead of the ten-criterion cost."""
    return cheng_snapshot(network, conditions, weather_by_city, traffic)[0]


def cheng_snapshot(network: RoadNetwork, conditions: Conditions, weather_by_city: dict[str, WeatherEffect] | None = None,
                   traffic: dict[str, tuple[float, float]] | None = None
                   ) -> tuple[dict[str, list[WeightedRoad]], dict[str, float]]:
    """The cost graph and each open road's travel time (s), which eq. 9 compares between snapshots."""
    validate_conditions(network, conditions)
    assumptions = network.provenance.get("assumptions", {})
    bpr_alpha, bpr_beta = assumptions.get("bpr_alpha", 0.15), assumptions.get("bpr_beta", 4.0)
    fixed = references(network)
    nodes = {node.id: node for node in network.nodes}
    roads, criteria = [], []
    for road in network.roads:
        if road.id in conditions.closed_road_ids:
            continue
        physical = segment_values(network, road, conditions, nodes, weather_by_city, traffic).physical
        roads.append(road)
        criteria.append((physical["distance_m"], physical["travel_time_s"],
                         estimated_flow_vph(physical["congestion"], physical["capacity_vph"], bpr_alpha, bpr_beta)))
    adjacency: dict[str, list[WeightedRoad]] = {node.id: [] for node in network.nodes}
    for road, cost in zip(roads, comprehensive_costs(criteria, fixed)):
        adjacency[road.source].append(WeightedRoad(road, cost))
    return adjacency, {road.id: row[1] for road, row in zip(roads, criteria)}


def rescale_pheromone(pheromone: dict[str, float], old_time: dict[str, float], new_time: dict[str, float],
                      initial: float = 1.0) -> list[str]:
    """Eq. 9 as written (reference mode only): tau(t+1) = t_(t+1) / t_t * tau(t). Returns the roads whose time changed.

    As printed, a road that becomes slower gains pheromone; this is not silently inverted. Closed roads lose their
    pheromone and reopened roads start again from the initial value.
    """
    changed = []
    for road_id in list(pheromone):
        if road_id not in new_time:
            del pheromone[road_id]
            changed.append(road_id)
        elif abs(new_time[road_id] - old_time[road_id]) > 1e-9 * old_time[road_id]:
            pheromone[road_id] *= new_time[road_id] / old_time[road_id]
            changed.append(road_id)
    for road_id in new_time.keys() - old_time.keys():
        pheromone[road_id] = initial
        changed.append(road_id)
    return changed


def cheng_route(adjacency, origin: str, destination: str, config: ChengConfig, positions, distance_reference_m: float,
                pheromone: dict[str, float] | None = None, started: float | None = None) -> SearchResult:
    """pheromone is used only in mode 'cheng_reference_printed'; the controlled mode always starts fresh."""
    warm = pheromone if config.mode == "cheng_reference_printed" else None
    return aco_search(adjacency, origin, destination, config, positions, distance_reference_m, warm, started)
