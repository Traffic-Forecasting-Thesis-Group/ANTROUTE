"""
ETA engine: travel time of a route from free-flow time and congestion risk.

    t(edge) = (free_flow(edge) + INTERSECTION_DELAY) * (1 + gamma * Risk(edge))

free_flow is OSM length / speed_kph -- the posted (or road-type default) limit, as if
driving every road at its limit without stopping. On real Metro Manila routes that
averages ~50 km/h, far above the city's measured free-flow speed (~30 km/h), because
signals and turns are missing. INTERSECTION_DELAY adds them back per junction passed,
so an expressway with few junctions stays fast while a route through many small
streets does not.

Both constants are calibrated against the TomTom Traffic Index 2025 for Metro Manila
(10 km in 31 min 45 s on average, congestion level 57.3% -> free-flow ~20.2 min per
10 km = ~29.7 km/h; evening rush hour 10 km in 43 min 29 s = ~2.15x free flow):

  INTERSECTION_DELAY  6.4 s: on 60 sampled cross-city fastest routes (mean 13.4 km,
                      103 junctions) this brings the posted-speed 49.8 km/h down to
                      29.7 km/h.
  DEFAULT_GAMMA       2.0: at the decoder's typical peak-window risk (~0.54 mean over
                      routed corridors) a route takes ~2.1x free flow, matching
                      evening rush hour; a fully congested edge takes 3x.

Both are first-pass, city-wide calibrations. Refit them once the team's Apple Maps /
Google Maps ETAs for the test trips are in (scripts/evaluate_routing.py collects them).
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import scipy.sparse as sp
from src.data.graph_data import GraphData
from src.routing.dynamic_weight import WeightedGraph

DEFAULT_GAMMA = 2.0
DEFAULT_INTERSECTION_DELAY_S = 6.4

# TomTom Traffic Index 2025, Metro Manila average congestion level: travel takes 57.3%
# longer than free flow over the whole day. The multiplier to use when no congestion
# risk is known at all, instead of pretending the roads are empty.
TYPICAL_CONGESTION_MULTIPLIER = 1.573


def congested_eta(
    free_flow_seconds: np.ndarray, risk: np.ndarray, gamma: float = DEFAULT_GAMMA
) -> np.ndarray:
    free_flow_seconds = np.asarray(free_flow_seconds, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)
    if free_flow_seconds.shape != risk.shape:
        raise ValueError(f"free_flow_seconds {free_flow_seconds.shape} and risk {risk.shape} must match")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError(f"gamma must be finite and >= 0, received {gamma}")
    if not np.all(np.isfinite(free_flow_seconds)) or np.any(free_flow_seconds <= 0):
        raise ValueError("every free-flow travel time must be finite and > 0")
    if not np.all(np.isfinite(risk)) or np.any((risk < 0) | (risk > 1)):
        raise ValueError("every risk must be finite and within [0, 1]")
    return free_flow_seconds * (1.0 + gamma * risk)


def load_free_flow_seconds(spatial_dir: Path, graph: GraphData) -> np.ndarray:
    spatial_dir = Path(spatial_dir)
    full_travel_time = sp.load_npz(spatial_dir / "metro_manila_travel_time.npz").tocsr()
    full_node_order = np.load(spatial_dir / "metro_manila_node_order.npy")
    position = {int(n): i for i, n in enumerate(full_node_order)}
    try:
        keep = np.array([position[int(n)] for n in graph.node_ids])
    except KeyError as missing:
        raise ValueError(
            f"node {missing} of this subgraph is not in {spatial_dir}'s full node order; the subgraph and the travel-time file were not built from the same run"
        ) from None
    sub_travel_time = full_travel_time[keep][:, keep]
    coo_adj = graph.adjacency.tocoo()
    coo_tt = sub_travel_time.tocoo()
    if not (np.array_equal(coo_adj.row, coo_tt.row) and np.array_equal(coo_adj.col, coo_tt.col)):
        raise ValueError(
            "the adjacency and free-flow travel-time matrices disagree on edge order for this subgraph; travel times would be attached to the wrong edges"
        )
    free_flow_seconds = coo_tt.data.astype(np.float64)
    if np.any(free_flow_seconds <= 0) or not np.all(np.isfinite(free_flow_seconds)):
        raise ValueError("the travel-time matrix holds a non-positive or non-finite value")
    return free_flow_seconds


def path_eta_seconds(
    wg: WeightedGraph,
    free_flow_seconds: np.ndarray,
    path: Sequence[int],
    gamma: float = DEFAULT_GAMMA,
    intersection_delay_s: float = DEFAULT_INTERSECTION_DELAY_S,
) -> float:
    """
    Congested travel time of `path`. Each edge is charged the delay of the junction it
    ends at, and that delay grows with congestion like the driving time does (queues at
    a signal lengthen in traffic).
    """
    if len(path) < 2:
        raise ValueError("a path needs at least two nodes")
    if not np.isfinite(intersection_delay_s) or intersection_delay_s < 0:
        raise ValueError(f"intersection_delay_s must be finite and >= 0, received {intersection_delay_s}")
    indices = [wg.index_of(n) for n in path]
    edges = [wg.edge_id(u, v) for u, v in zip(indices, indices[1:])]
    eta = congested_eta(free_flow_seconds[edges] + intersection_delay_s, wg.risk[edges], gamma)
    return float(eta.sum())