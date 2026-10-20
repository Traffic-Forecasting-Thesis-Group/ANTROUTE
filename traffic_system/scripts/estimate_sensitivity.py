"""Route sensitivity to the declared assumed scoring scenarios, with the calibration held fixed.

The flood class severity (class/3) is an assumed scoring scenario; configs/study_assumptions.json declares its
alternatives (equal, convex). Each alternative re-scores the same snapshot with the same fixed weights and
development maxima and the same ACO seed, so differences come from the mapping alone. Values above the
development maximum are flagged, not clipped. Illustrative, not empirical validation.
"""
import json
from pathlib import Path

from app.main import get_network
from app.places import index_for
from app.study_policy import require_historical_release
from shapely.geometry import shape
from src.routing.aco_routing import AntColonyConfig, ant_colony_route, distance_reference, node_positions
from src.routing.condition_update import Conditions
from src.routing.dynamic_weight import prepare_graph, segment_values
from src.routing.historical_traffic import traffic_for
from src.routing.historical_weather import historical_snapshot
from src.routing.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    require_historical_release()
    network = get_network()
    index = index_for(network)

    def representative(city):
        point = shape(network.boundaries[city]).representative_point()
        return min((node for node in network.nodes if node.city == city and node.id in index.connected),
                   key=lambda node: (node.longitude - point.x) ** 2 + (node.latitude - point.y) ** 2).id
    origin, destination = representative("Manila"), representative("Makati")
    conditions = Conditions(departure_date="2025-06-02", time_of_day=8, weather="historical")
    weather, _ = historical_snapshot(conditions.departure_date, conditions.time_of_day, network)
    traffic = traffic_for(network, conditions)
    declared = network.provenance["assumptions"]["flood_class_severity"]
    scenarios = {declared.get("scenario", "declared"): declared["values"], **declared.get("alternatives", {})}
    results = []
    for name, values in scenarios.items():
        variant = network.model_copy(update={"cost_config": network.cost_config.model_copy(
            update={"flood_class_severity": {hazard: float(value) for hazard, value in values.items()}})})
        config = AntColonyConfig()
        route = ant_colony_route(prepare_graph(variant, conditions, weather, traffic), origin, destination, config,
                                 node_positions(variant), distance_reference(variant, "antroute"))
        nodes = {node.id: node for node in variant.nodes}
        flags = sorted({flag for edge in route.roads for flag in segment_values(variant, edge.road, conditions, nodes, weather, traffic).flags
                        if flag.startswith("exceeds_development_max")})
        metrics = evaluate(route, config)
        results.append({"flood_class_severity_scenario": name, "values": values, "road_ids": [edge.road.id for edge in route.roads],
                        "evaluation": metrics, "exceedance_flags": flags})
        print(f"Flood severity '{name}': {metrics['distance_travel_m'] / 1000:.3f} km, cost {metrics['path_cost']:.4f}, "
              f"stop {metrics['convergence_rule']['stop_reason']}", flush=True)
    path = ROOT / "outputs/estimation_sensitivity.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"conditions": conditions.model_dump(mode="json"), "origin": origin, "destination": destination,
        "calibration_id": network.cost_config.calibration_id,
        "note": "Sensitivity of one route to the declared flood class severity alternatives with fixed weights, maxima and seed. "
                "Illustrative; not empirical validation.", "runs": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
