"""train_stgnn_edge.py and predict_congestion_risk.py: CCTV flow features, the camera embedding and
the NOAH flood hazard node feature."""

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from test_stgnn_training import TRAIN_DAY, data, path_graph  # noqa: F401  (shared synthetic frames, labels, weather, graph)
from src.data.graph_data import flood_node_features, load_flood_hazard, subgraph_from_arrays
from src.data.graph_dataset import GraphWindowDataset
from src.data.training_data import (
    FLOW_FEATURES,
    FlowFeatures,
    build_training_records,
    fit_vehicle_scale,
    load_flow_lookup,
    load_label_lookup,
)
from src.models.cnn_lstm_fusion import CNNLSTMFusion
from src.models.mlp_decoder import MLPDecoder
from src.models.radr_stgnn import RADRSTGNN
from src.models.traffic_risk_model_edge import TrafficRiskModel

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_edge = load_script("train_stgnn_edge")
predict_edge = load_script("predict_congestion_risk")
NODES, CAMERAS = ["NODE_A", "NODE_B"], {"CAM_A": "NODE_A", "CAM_B": "NODE_B"}


def rewrite_vehicle_counts(frames: Path, count_of) -> None:
    """Give every frame its own vehicle count, so the flow columns are not constant."""
    path = frames / "auto_labels.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        row["n_vehicles"] = count_of(i, row)
        row["occupancy"] = round(i / len(rows), 4)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_flow_columns_follow_the_detections_and_are_zero_where_there_is_no_frame(data):
    rewrite_vehicle_counts(data["frames"], lambda i, row: i % 20)
    records, _, _ = build_training_records(data["frames"], data["weather"])
    lookup = load_flow_lookup(data["frames"])
    flow = FlowFeatures(lookup, fit_vehicle_scale(records, lookup))

    record = next(r for r in records
                  if r.camera_id == "CAM_A" and r.split == "train" and r.frame_paths[3] is not None)
    columns = flow(record, 0)
    assert columns.shape == (30, FLOW_FEATURES)
    n, occ = lookup[record.frame_paths[3]]
    assert torch.allclose(columns[3], torch.tensor([np.log1p(n) / flow.vehicle_scale, occ, 1.0], dtype=torch.float32))

    record.frame_paths[5] = None                       # a minute with no frame: no detection either
    assert flow(record, 0)[5].tolist() == [0.0, 0.0, 0.0]


def test_vehicle_scale_is_fitted_on_training_frames_only(data):
    # Validation-day frames carry far more vehicles; they must not set the scale.
    rewrite_vehicle_counts(data["frames"], lambda i, row: 500 if TRAIN_DAY not in row["frame_path"] else 9)
    records, _, _ = build_training_records(data["frames"], data["weather"])
    assert fit_vehicle_scale(records, load_flow_lookup(data["frames"])) == np.log1p(9)


def test_graph_windows_carry_weather_time_and_flow_columns_in_that_order(data):
    records, _, _ = build_training_records(data["frames"], data["weather"])
    lookup = load_flow_lookup(data["frames"])
    flow = FlowFeatures(lookup, fit_vehicle_scale(records, lookup))
    ds = GraphWindowDataset(records, "train", load_label_lookup(data["frames"]), NODES, CAMERAS, 32,
                            time_features=True, flow=flow)
    assert ds.weather_dim == 3 + 2 + FLOW_FEATURES
    temporal = ds[0]["temporal"]
    assert temporal.shape[-1] == ds.weather_dim
    assert (temporal[..., -1] == 1.0).all()            # every fixture minute has a frame, so a detection


def test_the_camera_embedding_starts_neutral_and_shifts_only_camera_nodes():
    fusion = CNNLSTMFusion(text_dim=8, temporal_dim=2, image_size=32, patch_size=16, patch_embed_dim=8)
    stgnn = RADRSTGNN(in_features=fusion.lstm.hidden_size)
    with_emb = TrafficRiskModel(fusion, stgnn, MLPDecoder(stgnn.output_dim), n_camera_nodes=2)
    assert torch.count_nonzero(with_emb.camera_embedding.weight) == 0
    assert TrafficRiskModel(fusion, stgnn, MLPDecoder(stgnn.output_dim)).camera_embedding is None


def test_edge_training_with_flow_and_camera_embedding_round_trips_into_scoring(data, tmp_path, monkeypatch):
    rewrite_vehicle_counts(data["frames"], lambda i, row: i % 20)
    out = tmp_path / "ckpt" / "edge.pt"
    result = train_edge.train(
        data["frames"], data["weather"], None, None, out, camera_csv=data["camera_csv"], epochs=2,
        batch_size=2, image_size=32, patch_size=16, patch_embed_dim=8, device="cpu", graph=data["graph"],
        flow_features=True, camera_embedding=True)
    assert all(math.isfinite(h["train_loss"]) for h in result["history"])

    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    assert cfg["flow_features"] and cfg["camera_embedding"] and cfg["flood_features"]
    # The spatial branch reads the fused features plus one flood column.
    assert ckpt["model"]["stgnn.gcn.layers.0.linear.weight"].shape[1] == 128 + 1
    assert cfg["temporal_dim"] == 3 + 2 + FLOW_FEATURES and cfg["vehicle_scale"] == np.log1p(19)

    # Scoring rebuilds the model and the flow columns from the checkpoint config alone.
    monkeypatch.setattr(predict_edge, "build_subgraph", lambda spatial_dir, k: data["graph"])
    summary = predict_edge.predict(out, data["frames"], data["weather"], tmp_path / "scores", device="cpu")
    assert summary["windows"] > 0
    with (tmp_path / "scores" / "risk_edges.csv").open(encoding="utf-8", newline="") as f:
        risks = [float(r["risk"]) for r in csv.DictReader(f)]
    assert risks and all(0.0 <= r <= 1.0 for r in risks)


def test_old_checkpoints_without_the_new_flags_still_score(data, tmp_path, monkeypatch):
    out = tmp_path / "ckpt" / "old.pt"
    train_edge.train(
        data["frames"], data["weather"], None, None, out, camera_csv=data["camera_csv"], epochs=1,
        batch_size=2, image_size=32, patch_size=16, patch_embed_dim=8, device="cpu", graph=data["graph"],
        flood_features=False)
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["model"]["stgnn.gcn.layers.0.linear.weight"].shape[1] == 128   # adjacency only, as before
    for key in ("flow_features", "vehicle_scale", "camera_embedding", "flood_features"):   # as v1-v4 were saved
        ckpt["config"].pop(key)
    torch.save(ckpt, out)

    monkeypatch.setattr(predict_edge, "build_subgraph", lambda spatial_dir, k: data["graph"])
    assert predict_edge.predict(out, data["frames"], data["weather"], tmp_path / "scores", device="cpu")["windows"] > 0


# ---------------------------------------------------------------- flood hazard node feature

def test_subgraph_keeps_each_kept_nodes_flood_level_and_scales_it_to_unit_range():
    flood = np.array([3, 0, 1, 2, 0, 0, 0, 0, 1, 2, 3, 0])
    graph = subgraph_from_arrays(path_graph(12), np.arange(100, 112), {"A": 2, "B": 9}, k=2, flood_hazard=flood)
    assert graph.flood_hazard.tolist() == flood[graph.node_ids - 100].tolist()
    features = flood_node_features(graph)
    assert features.shape == (graph.n_nodes, 1)
    assert torch.allclose(features[:, 0], torch.tensor(graph.flood_hazard / 3, dtype=torch.float32))


def test_a_graph_without_flood_levels_refuses_to_make_the_feature():
    graph = subgraph_from_arrays(path_graph(12), np.arange(12), {"A": 2, "B": 9}, k=2)
    with pytest.raises(ValueError, match="flood"):
        flood_node_features(graph)


def test_flood_levels_follow_the_adjacency_node_order_and_none_may_be_missing(tmp_path):
    (tmp_path / "full_network_static_features.csv").write_text(
        "node_id,lat,lon,is_cctv_node,cctv_label,flood_hazard_level\n"
        "30,0,0,False,,2\n10,0,0,True,A,0\n20,0,0,False,,3\n", encoding="utf-8")
    assert load_flood_hazard(tmp_path, np.array([10, 20, 30])).tolist() == [0, 3, 2]
    with pytest.raises(ValueError, match="no flood hazard level"):
        load_flood_hazard(tmp_path, np.array([10, 20, 30, 40]))


def test_flood_feature_reaches_the_edge_scores():
    torch.manual_seed(0)
    fusion = CNNLSTMFusion(text_dim=8, temporal_dim=2, image_size=32, patch_size=16, patch_embed_dim=8)
    stgnn = RADRSTGNN(in_features=fusion.lstm.hidden_size + 1)
    model = TrafficRiskModel(fusion, stgnn, MLPDecoder(stgnn.output_dim)).eval()
    graph = subgraph_from_arrays(path_graph(10), np.arange(10), {"A": 2, "B": 7}, k=1,
                                 flood_hazard=np.zeros(10, dtype=int))
    camera_index = torch.tensor([graph.camera_nodes["A"], graph.camera_nodes["B"]])
    batch = {"images": torch.rand(1, 2, 4, 3, 32, 32), "text": torch.zeros(1, 2, 4, 8),
             "temporal": torch.rand(1, 2, 4, 2), "visual_mask": torch.ones(1, 2, 4, dtype=torch.bool),
             "text_mask": torch.zeros(1, 2, 4, dtype=torch.bool)}
    dry = flood_node_features(graph)
    flooded = dry.clone()
    flooded[0] = 1.0                                    # one high-hazard node, away from both cameras
    with torch.no_grad():
        a = model(batch, graph.a_hat, camera_index, graph.edge_index, dry)
        b = model(batch, graph.a_hat, camera_index, graph.edge_index, flooded)
    assert a.shape == (1, graph.edge_index.shape[1])
    assert not torch.allclose(a, b)


def test_no_flood_ablation_trains_on_the_adjacency_alone_and_scores(data, tmp_path, monkeypatch):
    out = tmp_path / "ckpt" / "no_flood.pt"
    train_edge.train(
        data["frames"], data["weather"], None, None, out, camera_csv=data["camera_csv"], epochs=1,
        batch_size=2, image_size=32, patch_size=16, patch_embed_dim=8, device="cpu", graph=data["graph"],
        flood_features=False)
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["config"]["flood_features"] is False
    assert ckpt["model"]["stgnn.gcn.layers.0.linear.weight"].shape[1] == 128

    monkeypatch.setattr(predict_edge, "build_subgraph", lambda spatial_dir, k: data["graph"])
    assert predict_edge.predict(out, data["frames"], data["weather"], tmp_path / "scores", device="cpu")["windows"] > 0
