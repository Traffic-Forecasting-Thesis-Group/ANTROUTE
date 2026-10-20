"""ANTROUTE search: the ten-criterion CRITIC edge cost on the shared ACO kernel (aco_kernel.py)."""
from __future__ import annotations

from .aco_kernel import (NoFeasibleRoute, SearchConfig, SearchExhausted, SearchResult, SearchTimeLimit, aco_search,
                         reachable_from as _reachable)

# Convergence rule of the study (Aadil et al., 2016), shared by ANTROUTE and the baseline: the best path cost has not
# improved over a window of 20 iterations; N_conv is the iteration that first meets it.
CONVERGENCE_PATIENCE = 20

AntColonyConfig = SearchConfig
ColonyResult = SearchResult

__all__ = ["AntColonyConfig", "ColonyResult", "CONVERGENCE_PATIENCE", "NoFeasibleRoute", "SearchExhausted", "SearchTimeLimit",
           "ant_colony_route", "distance_reference", "node_positions", "_reachable"]


def node_positions(network) -> dict[str, tuple[float, float]]:
    return {node.id: (node.longitude, node.latitude) for node in network.nodes}


def distance_reference(network, model: str) -> float:
    """Frozen destination-guidance distance reference (m) for 'antroute' or 'cheng', from the calibration."""
    try:
        return float(network.provenance["distance_reference_m"][model])
    except (KeyError, TypeError, ValueError):
        raise ValueError("The network has no frozen distance reference; rebuild the calibrated release") from None


def ant_colony_route(adjacency, origin: str, destination: str, config: SearchConfig, positions, distance_reference_m: float,
                     started: float | None = None) -> SearchResult:
    """ANTROUTE always starts from fresh pheromone, including every re-planning decision."""
    return aco_search(adjacency, origin, destination, config, positions, distance_reference_m, None, started)
