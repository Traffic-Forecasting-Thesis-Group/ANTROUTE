"""Evaluation of one static search (one snapshot)."""
from .aco_kernel import SearchConfig, SearchResult


def evaluate(result: SearchResult, config: SearchConfig, search_time_ms: float | None = None) -> dict:
    """Distance, common/native path cost, ACO search time and convergence behavior of one search."""
    return {
        "distance_travel_m": sum(edge.road.distance_m for edge in result.roads),
        "path_cost": sum(edge.cost for edge in result.roads),
        "search_time_ms": result.search_time_ms if search_time_ms is None else search_time_ms,
        "computational_time_ms": result.search_time_ms if search_time_ms is None else search_time_ms,
        "convergence_rule": {
            "patience": config.patience, "abs_tol": config.abs_tol, "rel_tol": config.rel_tol,
            "max_iterations": config.max_iterations, "wall_time_ms": config.wall_time_ms,
            "iterations": result.iterations_completed,
            "iterations_to_convergence": result.converged_at,  # N_conv; None when not converged.
            "first_best_found_iteration": result.first_best_found_iteration,
            "best_found_at_iteration": result.first_best_found_iteration,
            "converged": result.converged,
            "stop_reason": result.stop_reason,
            "best_cost_by_iteration": result.best_cost_by_iteration,
        },
    }
