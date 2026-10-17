"""Report the four study metrics under explicit demand-assumption perturbations."""
import json
from pathlib import Path
from time import perf_counter

from app.main import get_network
from app.places import index_for
from app.study_policy import require_historical_release
from shapely.geometry import shape
from src.routing.aco_routing import AntColonyConfig, ant_colony_route
from src.routing.condition_update import Conditions
from src.routing.dynamic_weight import prepare_graph
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
    positions = {node.id: (node.longitude, node.latitude) for node in network.nodes}
    weather, _ = historical_snapshot(conditions.departure_date, conditions.time_of_day, network)
    results = []
    for scale in (.8, 1.0, 1.2):
        variant = network.model_copy(update={"provenance": {**network.provenance, "assumptions": {
            **network.provenance["assumptions"], "demand_scale": scale}}})
        config = AntColonyConfig()
        start = perf_counter()
        graph = prepare_graph(variant, conditions, weather)
        route = ant_colony_route(graph, origin, destination, config, positions)
        metrics = evaluate(route, config, (perf_counter() - start) * 1000)
        results.append({"assumed_demand_multiplier": scale, "evaluation": metrics})
        print(f"Demand x{scale}: {metrics['distance_travel_m'] / 1000:.3f} km, cost {metrics['path_cost']:.4f}", flush=True)
    path = ROOT / "outputs/estimation_sensitivity.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"conditions": conditions.model_dump(mode="json"), "origin": origin,
        "destination": destination, "note": "Illustrative sensitivity to unmeasured demand assumptions, not empirical validation. ACO seed held fixed.",
        "runs": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
