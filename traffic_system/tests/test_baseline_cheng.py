import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.routing.aco_routing import NoFeasibleRoute
from src.routing.baseline_cheng import (ChengConfig, ahp, cheng_graph, cheng_route, comprehensive_costs,
                                        estimated_flow_vph)
from src.routing.condition_update import Conditions
from src.routing.dynamic_weight import WeightedRoad
from src.routing.historical_traffic import bpr
from src.routing.network import Road

# Figure 3 and Tables 5-7 of Cheng (2023): dimensionless length, travel time and traffic flow.
PAPER_EDGES = {
    ("O", "1"): (.549, .659, .533), ("O", "3"): (.718, .462, .347), ("1", "2"): (.676, .704, .507),
    ("1", "4"): (.704, .634, .615), ("2", "6"): (.746, .987, 1.000), ("3", "4"): (.563, .423, .450),
    ("3", "10"): (1.000, 1.000, .770), ("4", "5"): (.282, .298, .254), ("4", "11"): (.972, .921, .758),
    ("5", "6"): (.282, .254, .256), ("5", "7"): (.423, .634, .464), ("6", "8"): (.451, .272, .303),
    ("7", "8"): (.268, .241, .304), ("7", "12"): (.437, .357, .687), ("8", "9"): (.282, .241, .258),
    ("8", "13"): (.394, .245, .488), ("9", "D"): (.423, .585, .411), ("10", "11"): (.563, .441, .488),
    ("11", "12"): (.282, .338, .450), ("12", "13"): (.310, .254, .251), ("13", "D"): (.282, .220, .256),
}
PAPER_GRID = {"O": (0, 0), "1": (1, 0), "2": (3, 0), "3": (0, 1), "4": (1, 1), "5": (2, 1), "6": (3, 1),
              "7": (2, 2), "8": (3, 2), "9": (4, 2), "10": (0, 3), "11": (1, 3), "12": (2, 3), "13": (3, 3), "D": (4, 3)}


def paper_network():
    costs = comprehensive_costs(list(PAPER_EDGES.values()))
    adjacency = {node: [] for node in PAPER_GRID}
    for ((source, target), (length, _, _)), cost in zip(PAPER_EDGES.items(), costs):
        road = Road.model_construct(id=f"{source}-{target}", source=source, target=target, distance_m=length)
        adjacency[source].append(WeightedRoad(road, cost))
    # Grid spacing of 0.5 length units, in degrees at the equator.
    positions = {node: (x * 0.5 / 111195, -y * 0.5 / 111195) for node, (x, y) in PAPER_GRID.items()}
    return adjacency, positions


def all_paths(adjacency, node, destination):
    if node == destination:
        yield []
    for edge in adjacency[node]:
        for rest in all_paths(adjacency, edge.road.target, destination):
            yield [edge, *rest]


def test_ahp_reproduces_paper_weights_and_consistency():
    result = ahp()
    assert result.weights == pytest.approx((0.637, 0.258, 0.105), abs=5e-4)
    assert result.lambda_max == pytest.approx(3.0385, abs=5e-4)
    assert result.consistency_ratio == pytest.approx(0.0333, abs=5e-4)
    assert result.consistency_ratio < 0.1


def test_comprehensive_cost_of_papers_reported_path():
    adjacency, _ = paper_network()
    edges = {edge.road.id: edge for roads in adjacency.values() for edge in roads}
    path = ["O-3", "3-4", "4-5", "5-7", "7-12", "12-13", "13-D"]
    assert sum(edges[road].road.distance_m for road in path) == pytest.approx(3.015)  # Table 8, actual length
    # Table 8 reports 2.885 with weights rounded to three decimals.
    assert sum(edges[road].cost for road in path) == pytest.approx(2.885, abs=5e-3)


@pytest.mark.parametrize("seed", range(3))
def test_finds_least_comprehensive_cost_path_on_paper_network(seed):
    adjacency, positions = paper_network()
    optimum = min(sum(edge.cost for edge in path) for path in all_paths(adjacency, "O", "D"))
    # Sec. 5.1.2: 23 ants for 15 nodes; Sec. 4.2.1: q0 = 0.2 for a small network.
    result = cheng_route(adjacency, "O", "D", ChengConfig(n_ants=23, max_iterations=100, q0=0.2, seed=seed), positions)
    assert sum(edge.cost for edge in result.roads) == pytest.approx(optimum)
    # The study's convergence rule: stop once the best cost has not improved for 20 consecutive iterations.
    assert result.converged_at == result.best_found_at + 20 == len(result.best_cost_by_iteration)


def test_reaching_the_iteration_limit_is_not_converged():
    adjacency, positions = paper_network()
    result = cheng_route(adjacency, "O", "D", ChengConfig(n_ants=23, max_iterations=10, q0=0.2), positions)
    assert not result.converged and result.converged_at is None
    assert len(result.best_cost_by_iteration) == 10


def test_flow_estimate_inverts_the_bpr_congestion():
    _, congestion = bpr(0.8, {"bpr_alpha": .15, "bpr_beta": 4})
    assert estimated_flow_vph(congestion, 1000, .15, 4) == pytest.approx(800)
    assert estimated_flow_vph(0, 1000, .15, 4) == 0


def test_cost_uses_the_three_criteria_of_the_snapshot(network):
    graph = cheng_graph(network, Conditions(time_of_day=8, weather="clear"))
    costs = {edge.road.id: edge.cost for roads in graph.values() for edge in roads}
    # Fixture: 100/100/1000 m, 10 s each, no congestion so no flow.
    w1, w2, _ = ahp().weights
    assert costs == pytest.approx({"ab": w1 * .1 + w2, "bd": w1 * .1 + w2, "ad": w1 + w2})
    rainy = cheng_graph(network, Conditions(time_of_day=8, weather="rain"))
    assert {edge.road.id: edge.cost for roads in rainy.values() for edge in roads} == pytest.approx(costs)  # time doubles everywhere


def test_closures_apply_as_in_antroute(network):
    positions = {node.id: (node.longitude, node.latitude) for node in network.nodes}

    def route(closed):
        graph = cheng_graph(network, Conditions(time_of_day=8, weather="clear", closed_road_ids=closed))
        return [edge.road.id for edge in cheng_route(graph, "a", "d", ChengConfig(), positions).roads]

    assert route({"ad"}) == ["ab", "bd"]
    assert route({"ab"}) == ["ad"]
    with pytest.raises(NoFeasibleRoute):
        route({"ab", "ad"})


def test_seed_makes_runs_repeatable(network):
    positions = {node.id: (node.longitude, node.latitude) for node in network.nodes}
    graph = cheng_graph(network, Conditions(time_of_day=8, weather="clear"))
    first = cheng_route(graph, "a", "d", ChengConfig(seed=7), positions)
    assert cheng_route(graph, "a", "d", ChengConfig(seed=7), positions).best_cost_by_iteration == first.best_cost_by_iteration


def test_baseline_api_scores_path_cost_like_antroute(network, monkeypatch):
    import app.main as service
    from app.study_policy import require_historical_release
    monkeypatch.setitem(app.dependency_overrides, require_historical_release, lambda: None)
    monkeypatch.setattr(service, "get_network", lambda: network)
    payload = {"origin": "a", "destination": "d",
               "conditions": {"time_of_day": 8, "weather": "clear", "closed_road_ids": ["ad"]}}
    with TestClient(app) as client:
        baseline = client.post("/routes/baseline", json=payload)
        antroute = client.post("/routes/plan", json=payload).json()
        assert baseline.status_code == 200
        body = baseline.json()
        assert body["route"]["road_ids"] == antroute["route"]["road_ids"] == ["ab", "bd"]
        assert set(body["evaluation"]) == {"distance_travel_m", "path_cost", "computational_time_ms", "convergence_rule"}
        assert body["evaluation"]["path_cost"] == pytest.approx(antroute["evaluation"]["path_cost"])
        assert body["baseline"]["doi"] == "10.1155/2023/7651100"
        payload["conditions"]["closed_road_ids"] = ["ab", "ad"]
        assert client.post("/routes/baseline", json=payload).status_code == 404
