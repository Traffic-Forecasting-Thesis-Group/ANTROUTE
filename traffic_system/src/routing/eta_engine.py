"""
===============================================================================
ARCHITECTURE BLOCK 19 (part): ETA Estimation
===============================================================================
Diagram path:   part of [DYNAMIC ROUTING ENGINE]
                "ACO, alt. routes, multi-stop, ETA Estimation"

WHAT IT DOES
    Converts a chosen route into a predicted travel time:

        t_edge = t_freeflow * (1 + gamma * Risk)

    t_freeflow is the OSM free-flow seconds for that road (from block 13's
    travel-time matrix) and Risk is the Congestion Risk Score (block 17).
    gamma = DEFAULT_GAMMA = 1.0, so a fully congested edge is predicted to
    take twice as long as it would on an empty road.

    Note the shape is deliberately the same as the Dynamic Weight Engine's
    W = distance * (1 + lambda * Risk): one penalises distance for routing,
    this one penalises time for display.

HONEST RESULT (carry this into the paper)
    This is the weakest part of the system. In the 50-trip evaluation the
    Cheng IACO baseline, which estimates ETA from OBSERVED travel times,
    beat ANTROUTE on every error metric (MAE 826s vs 1095s, MAPE 37.8% vs
    51.8%, both significant). ANTROUTE wins clearly on route QUALITY and
    loses on ETA accuracy -- report both.

INPUT   <- free-flow seconds per edge, risk per edge, a chosen path
OUTPUT  -> predicted seconds, surfaced as duration_min in the app

KEY NAMES
    DEFAULT_GAMMA   1.0
    congested_eta() the formula, vectorised over edges
===============================================================================
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import scipy.sparse as sp
from src.data.graph_data import GraphData
from src.routing.dynamic_weight import WeightedGraph

DEFAULT_GAMMA = 1.0


# [BLOCK 19 - ETA] t = t_freeflow * (1 + gamma * Risk). Same shape as the
# Dynamic Weight Engine's formula, applied to time instead of distance.
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
    wg: WeightedGraph, free_flow_seconds: np.ndarray, path: Sequence[int], gamma: float = DEFAULT_GAMMA
) -> float:
    if len(path) < 2:
        raise ValueError("a path needs at least two nodes")
    indices = [wg.index_of(n) for n in path]
    edges = [wg.edge_id(u, v) for u, v in zip(indices, indices[1:])]
    eta = congested_eta(free_flow_seconds[edges], wg.risk[edges], gamma)
    return float(eta.sum())