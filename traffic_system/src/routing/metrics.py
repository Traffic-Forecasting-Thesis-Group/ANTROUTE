"""The four evaluation items specified for this study."""
from .aco_routing import AntColonyConfig, ColonyResult


def evaluate(result: ColonyResult, config: AntColonyConfig, elapsed_ms: float) -> dict:
    return {
        "distance_travel_m": sum(edge.road.distance_m for edge in result.roads),
        "path_cost": sum(edge.cost for edge in result.roads),
        "computational_time_ms": elapsed_ms,
        "convergence_rule": {
            "tolerance": config.tolerance,
            "patience": config.patience,
            "max_iterations": config.max_iterations,
            "iterations": len(result.best_cost_by_iteration),
            "iterations_to_convergence": result.best_found_at,
            "converged": result.converged,
            "stop_reason": "convergence_rule" if result.converged else "iteration_limit",
            "best_cost_by_iteration": result.best_cost_by_iteration,
        },
    }
