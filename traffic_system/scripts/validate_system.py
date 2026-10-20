"""Exercise the real five-city release end to end through the API (software verification, not empirical validation)."""
import json
from pathlib import Path

import networkx as nx
from fastapi.testclient import TestClient
from shapely.geometry import shape

from app.main import app, get_network
from app.study_policy import require_historical_release
from src.routing.network import CITIES

ROOT = Path(__file__).resolve().parents[1]


def main():
    require_historical_release()
    network = get_network()
    graph = nx.DiGraph((road.source, road.target) for road in network.roads)
    component = max(nx.strongly_connected_components(graph), key=len)
    representatives = {}
    for city in CITIES:
        center = shape(network.boundaries[city]).representative_point()
        representatives[city] = min((node for node in network.nodes if node.city == city and node.id in component),
                                    key=lambda node: (node.longitude - center.x) ** 2 + (node.latitude - center.y) ** 2).id
    cases = []
    for source, target in zip(CITIES, CITIES[1:] + CITIES[:1]):
        cases.append((f"{source} to {target}", representatives[source], representatives[target], 8, "clear", []))
    source, target = representatives["Manila"], representatives["Makati"]
    cases.extend([
        ("Manila to Makati off peak", source, target, 3, "clear", []),
        ("Manila to Makati rain", source, target, 8, "rain", []),
        ("Manila to Makati historical provider weather", source, target, 20, "historical", []),
    ])
    results = []
    with TestClient(app) as client:
        baseline = client.post("/routes/plan", json={"origin": source, "destination": target,
            "conditions": {"time_of_day": 8, "weather": "clear", "closed_road_ids": []}})
        if baseline.status_code != 200:
            raise RuntimeError(f"Baseline request failed: HTTP {baseline.status_code}")
        blocked = baseline.json()["route"]["road_ids"]
        cases.append(("Manila to Makati road closure", source, target, 8, "clear", [blocked[len(blocked) // 2]]))
        for label, origin, destination, hour, weather, closures in cases:
            response = client.post("/routes/plan", json={"origin": origin, "destination": destination,
                "conditions": {"time_of_day": hour, "weather": weather, "closed_road_ids": closures}})
            if response.status_code != 200:
                raise RuntimeError(f"{label}: HTTP {response.status_code}: {response.json().get('detail')}")
            body = response.json()
            assert body["route"]["node_ids"][0] == origin
            assert body["route"]["node_ids"][-1] == destination
            assert not set(closures).intersection(body["route"]["road_ids"])
            assert len(body["route"]["node_ids"]) == len(set(body["route"]["node_ids"]))
            assert body["decision"]["road_ids"] == body["route"]["road_ids"] and body["decision"]["contract_version"] == 2
            contributions = sum(item["contribution"] for item in body["contributions"])
            assert abs(contributions - body["decision"]["common_evaluation_cost"]) <= 1e-9 * max(1.0, contributions)
            result = {"case": label, "conditions": body["conditions"], "evaluation": body["evaluation"], "decision": body["decision"]}
            results.append(result)
            print(f"Passed: {label} ({body['evaluation']['distance_travel_m'] / 1000:.2f} km)", flush=True)
    output = ROOT / "outputs/study_validation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"network_built_at": network.provenance["built_at"],
                                  "calibration_id": network.cost_config.calibration_id,
                                  "representative_nodes": representatives, "runs": results}, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} verified real-network runs to outputs/study_validation.json")


if __name__ == "__main__":
    main()
