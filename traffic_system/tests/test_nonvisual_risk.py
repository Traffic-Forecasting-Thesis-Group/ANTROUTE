"""
The non-visual Congestion Risk layer (src/models/nonvisual_risk.py): missing inputs are
explicit, the model learns the "unavailable" branch, and its prediction moves with each road's
own inputs. These are behaviour tests; accuracy is measured by scripts/train_nonvisual_risk.py.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.data.graph_data import GraphData
from src.models.congestion_risk_score import edge_index_from_adjacency
from src.models.nonvisual_risk import (
    COLUMN,
    FEATURES,
    MODEL_FEATURES,
    NonVisualRiskModel,
    PlacedIncidents,
    RoadFeatures,
    grid_cell,
    with_modality_dropout,
)

# Three roads in a row near EDSA-Ortigas, then one ~5 km away (far from any incident point).
LAT = [14.5934, 14.5944, 14.5954, 14.6400]
LON = [121.0583, 121.0583, 121.0583, 121.0583]
FLOOD = [0, 1, 3, 0]


@pytest.fixture
def roads(tmp_path):
    n = len(LAT)
    rows, cols, lengths = [0, 1, 2], [1, 2, 3], [110.0, 110.0, 5000.0]
    adjacency = sp.csr_matrix((lengths, (rows, cols)), shape=(n, n))
    graph = GraphData(node_ids=np.arange(100, 100 + n), adjacency=adjacency, a_hat=None,
                      edge_index=edge_index_from_adjacency(adjacency), camera_nodes={})
    pd.DataFrame({"node_id": graph.node_ids, "lat": LAT, "lon": LON, "is_cctv_node": False,
                  "cctv_label": "", "flood_hazard_level": FLOOD}).to_csv(
        tmp_path / "full_network_static_features.csv", index=False)
    cell = grid_cell(np.array(LAT[:1]), np.array(LON[:1]))[0]
    pd.DataFrame({"date": ["2026-05-25"], "ws_temp_c": [31.0], "ws_precip_mm": [12.0],
                  "ws_humidity_pct": [80.0], "grid_cell": [cell]}).to_csv(tmp_path / "weather.csv", index=False)
    incidents = PlacedIncidents(
        events=pd.DataFrame({"at": [pd.Timestamp("2026-05-25 17:30")], "lat": [14.5944], "lon": [121.0583], "impact": [1.0]}),
        points=np.array([[14.5944, 121.0583]]),
        feed_days=frozenset({date(2026, 5, 25), date(2026, 5, 26)}),
    )
    return RoadFeatures.build(graph, tmp_path, tmp_path / "weather.csv", incidents)


def edge(roads, u, v):
    return int(np.flatnonzero((roads.node_ids[roads.src] == u) & (roads.node_ids[roads.dst] == v))[0])


def test_incident_status_distinguishes_none_reported_from_unavailable(roads):
    near, far = edge(roads, 100, 101), edge(roads, 102, 103)
    live = roads.at_window(datetime(2026, 5, 25, 17, 30))
    assert live[near, COLUMN["incident_impact"]] > 0                 # an alert is live next to it
    quiet = roads.at_window(datetime(2026, 5, 26, 17, 30))
    assert quiet[near, COLUMN["incident_impact"]] == 0.0             # feed on, covered, nothing reported
    assert quiet[near, COLUMN["incident_covered"]] == 1.0
    assert quiet[far, COLUMN["incident_covered"]] == 1.0             # its tail node is within coverage
    no_feed = roads.at_window(datetime(2026, 5, 27, 17, 30))
    assert np.isnan(no_feed[near, COLUMN["incident_impact"]])        # no feed that day: unknown, not 0
    assert no_feed[near, COLUMN["incident_feed_available"]] == 0.0


def test_a_road_far_from_every_placeable_point_has_unknown_incident_status(roads):
    roads.covered = np.array([True, True, False])                    # pretend the last road is out of reach
    x = roads.at_window(datetime(2026, 5, 26, 17, 30))
    far = edge(roads, 102, 103)
    assert np.isnan(x[far, COLUMN["incident_impact"]]) and x[far, COLUMN["incident_covered"]] == 0.0


def test_weather_comes_from_the_roads_own_cell_and_is_marked_missing_otherwise(roads):
    near, far = edge(roads, 100, 101), edge(roads, 102, 103)
    x = roads.at_window(datetime(2026, 5, 25, 17, 30))
    assert x[near, COLUMN["precip"]] == pytest.approx(12.0) and x[near, COLUMN["weather_available"]] == 1.0
    assert x[near, COLUMN["flood_x_precip"]] == pytest.approx(12.0 * 1 / 3)   # flood level 1 of 3 at node 101
    missing = roads.at_window(datetime(2026, 5, 24, 17, 30))         # no weather row that day
    assert np.isnan(missing[near, COLUMN["precip"]]) and missing[near, COLUMN["weather_available"]] == 0.0


def test_modality_dropout_adds_copies_with_each_source_withheld():
    x = np.random.default_rng(0).random((5, len(FEATURES))).astype(np.float32)
    y = np.linspace(0, 1, 5)
    xs, ys = with_modality_dropout(x, y)
    assert xs.shape == (15, len(FEATURES)) and ys.shape == (15,)
    assert np.isnan(xs[5:10, COLUMN["incident_impact"]]).all() and (xs[5:10, COLUMN["incident_covered"]] == 0).all()
    assert np.isnan(xs[10:15, COLUMN["precip"]]).all() and (xs[10:15, COLUMN["weather_available"]] == 0).all()
    assert np.allclose(xs[:5], x)                                    # the originals are untouched


def synthetic_rows(n=4000, seed=0):
    """Rows where risk rises with rain and with a live incident, and road columns are noise."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, len(FEATURES))).astype(np.float32)
    x[:, COLUMN["precip"]] = rng.uniform(0, 30, n)
    x[:, COLUMN["incident_impact"]] = rng.choice([0.0, 1.0], n, p=[0.8, 0.2])
    x[:, COLUMN["weather_available"]] = 1.0
    x[:, COLUMN["incident_covered"]] = 1.0
    x[:, COLUMN["incident_feed_available"]] = 1.0
    y = np.clip(0.4 + 0.01 * x[:, COLUMN["precip"]] + 0.3 * x[:, COLUMN["incident_impact"]] + rng.normal(0, 0.03, n), 0, 1)
    return x, y


def test_prediction_responds_to_road_specific_inputs_and_ignores_unused_columns():
    x, y = synthetic_rows()
    model = NonVisualRiskModel.fit(x, y, {"split": "synthetic"})
    row = x[:200].copy()
    dry, wet = row.copy(), row.copy()
    dry[:, COLUMN["precip"]], wet[:, COLUMN["precip"]] = 0.0, 25.0
    assert model.predict(wet).mean() > model.predict(dry).mean() + 0.1
    calm, incident = row.copy(), row.copy()
    calm[:, COLUMN["incident_impact"]], incident[:, COLUMN["incident_impact"]] = 0.0, 1.0
    assert model.predict(incident).mean() > model.predict(calm).mean() + 0.15
    rewired = row.copy()
    rewired[:, COLUMN["edge_speed"]] = 5.0                            # a road column: not a model input
    assert np.allclose(model.predict(rewired), model.predict(row))
    assert set(model.features) == set(MODEL_FEATURES)


def test_unavailable_inputs_still_get_a_prediction_from_the_learned_branch():
    x, y = synthetic_rows()
    model = NonVisualRiskModel.fit(x, y, {"split": "synthetic"})
    unknown = x[:50].copy()
    unknown[:, COLUMN["incident_impact"]] = np.nan
    unknown[:, COLUMN["incident_count"]] = np.nan
    unknown[:, COLUMN["incident_covered"]] = 0.0
    p = model.predict(unknown)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()


def test_save_and_load_keep_the_feature_list(tmp_path):
    x, y = synthetic_rows(500)
    model = NonVisualRiskModel.fit(x, y, {"split": "synthetic"})
    model.save(tmp_path / "m.joblib")
    loaded = NonVisualRiskModel.load(tmp_path / "m.joblib")
    assert loaded.features == model.features
    assert np.allclose(loaded.predict(x[:20]), model.predict(x[:20]))
