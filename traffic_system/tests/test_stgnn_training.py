import csv
import importlib.util
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src.data.graph_data import (
    derive_camera_map,
    khop_nodes,
    resolve_camera_map,
    subgraph_from_arrays,
)
from src.data.graph_dataset import GraphWindowDataset, last_labelled_step
from src.data.training_data import IGNORE_INDEX, build_training_records, load_label_lookup
from src.models.cnn_lstm_fusion import CNNLSTMFusion
from src.models.radr_stgnn import RADRSTGNN
from src.models.traffic_risk_model import TrafficRiskModel, camera_node_loss, scatter_camera_features

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_stgnn.py"
spec = importlib.util.spec_from_file_location("train_stgnn", SCRIPT)
train_stgnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_stgnn)

LABELS = ["Light", "Medium", "Heavy"]
TRAIN_DAY, VAL_DAY = "2026-05-04", "2026-05-20"      # both in alignment.VISUAL_SPLIT


def path_graph(n=12):
    """0-1-2-...-(n-1), undirected."""
    edges = [(i, i + 1) for i in range(n - 1)]
    rows = [a for a, b in edges] + [b for a, b in edges]
    cols = [b for a, b in edges] + [a for a, b in edges]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------- graph

def test_khop_nodes_grow_with_k_and_ignore_edge_direction():
    adj = sp.csr_matrix(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]))   # directed chain
    assert khop_nodes(adj, [2], 0).tolist() == [2]
    assert khop_nodes(adj, [2], 1).tolist() == [1, 2, 3]
    assert khop_nodes(adj, [2], 2).tolist() == [0, 1, 2, 3]


def test_subgraph_keeps_camera_positions_and_is_normalised():
    graph = subgraph_from_arrays(path_graph(12), np.arange(100, 112), {"A": 2, "B": 9}, k=2)
    assert graph.node_ids.tolist() == [0 + 100, 101, 102, 103, 104, 107, 108, 109, 110, 111]
    assert graph.node_ids[graph.camera_nodes["A"]] == 102 and graph.node_ids[graph.camera_nodes["B"]] == 109
    assert graph.a_hat.shape == (10, 10) and graph.edge_index.shape[0] == 2
    assert torch.allclose(graph.a_hat, graph.a_hat.T)


def test_camera_names_map_to_intersections_when_the_folder_says_so():
    labels = ["EDSA-Quezon Ave", "8337-Ayala NB 1-PTZ", "Roxas Blvd-Kalaw"]
    cams = ["F563_8337 - Ayala NB 1", "B283_3050", "ABCD_9999 - Roxas Blvd Kalaw"]
    assert derive_camera_map(cams, labels) == {cams[0]: labels[1], cams[2]: labels[2]}

    csv_path = Path(__file__).parent / "_camera_nodes_tmp.csv"
    csv_path.write_text("camera_id,intersection\nB283_3050,EDSA-Quezon Ave\nZZZ,Not A Node\n", encoding="utf-8")
    try:
        mapping, unmapped = resolve_camera_map(cams, labels, csv_path)
    finally:
        csv_path.unlink()
    assert mapping[cams[1]] == "EDSA-Quezon Ave" and unmapped == []


# ---------------------------------------------------------------- model

def tiny_model(n_nodes=10):
    fusion = CNNLSTMFusion(text_dim=768, temporal_dim=3, image_size=32, patch_size=16, patch_embed_dim=8)
    return TrafficRiskModel(fusion, RADRSTGNN(in_features=fusion.lstm.hidden_size)), n_nodes


def fake_batch(b=2, k=2, t=4, size=32):
    return {"images": torch.rand(b, k, t, 3, size, size), "text": torch.rand(b, k, t, 768),
            "temporal": torch.rand(b, k, t, 3), "visual_mask": torch.ones(b, k, t, dtype=torch.bool),
            "text_mask": torch.zeros(b, k, t, dtype=torch.bool)}


def test_scatter_places_camera_features_and_fills_the_rest():
    feats = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    fill = torch.full((4,), -1.0)
    out = scatter_camera_features(feats, torch.tensor([1, 5]), 7, fill)
    assert out.shape == (2, 3, 7, 4)
    assert torch.equal(out[:, :, 1], feats[:, :, 0]) and torch.equal(out[:, :, 5], feats[:, :, 1])
    assert (out[:, :, [0, 2, 3, 4, 6]] == -1).all()


def test_forward_shapes_and_gradients_reach_every_stage():
    model, n = tiny_model()
    graph = subgraph_from_arrays(path_graph(n), np.arange(n), {"A": 2, "B": 7}, k=1)
    camera_index = torch.tensor([graph.camera_nodes["A"], graph.camera_nodes["B"]])
    logits = model(fake_batch(), graph.a_hat, camera_index)
    assert logits.shape == (2, graph.n_nodes, 3)

    target = torch.tensor([[0, 2], [1, IGNORE_INDEX]])
    camera_node_loss(logits, target, camera_index).backward()
    for name, params in (("fusion", model.fusion.parameters()), ("gcn", model.stgnn.gcn.parameters()),
                         ("gru", model.stgnn.gru.parameters()), ("head", model.head.parameters())):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params), f"no gradient in {name}"
    assert model.placeholder.grad is not None and model.placeholder.grad.abs().sum() > 0


def test_loss_ignores_unlabelled_nodes():
    logits = torch.randn(2, 6, 3)
    camera_index = torch.tensor([1, 4])
    labelled = camera_node_loss(logits, torch.tensor([[0, 2], [1, 0]]), camera_index)
    only_one = camera_node_loss(logits, torch.tensor([[0, IGNORE_INDEX], [IGNORE_INDEX, IGNORE_INDEX]]), camera_index)
    expected = torch.nn.functional.cross_entropy(logits[0:1, [1]].reshape(1, 3), torch.tensor([0]))
    assert torch.allclose(only_one, expected) and not torch.allclose(labelled, only_one)


def test_last_labelled_step_picks_the_final_label():
    assert last_labelled_step(torch.tensor([0, 1, IGNORE_INDEX, 2, IGNORE_INDEX])) == 2
    assert last_labelled_step(torch.full((5,), IGNORE_INDEX)) == IGNORE_INDEX


# ---------------------------------------------------------------- data + end to end

@pytest.fixture
def data(tmp_path):
    frames = tmp_path / "frames"
    rows = []
    for cam in ("CAM_A", "CAM_B"):
        for day in (TRAIN_DAY, VAL_DAY):
            (frames / day / cam).mkdir(parents=True)
            for m in range(60):
                rel = f"{day}/{cam}/f_{m:03d}.jpg"
                cv2.imwrite(str(frames / rel), np.full((32, 32, 3), 4 * m, np.uint8))
                rows.append((cam, rel, (datetime.fromisoformat(f"{day}T17:00:00") + timedelta(minutes=m)).isoformat()))
    with (frames / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera_id", "timestamp", "frame_path", "start_estimated"])
        for cam, rel, ts in rows:
            w.writerow([cam, ts, rel, False])
    with (frames / "auto_labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_path", "camera_id", "timestamp", "n_vehicles", "occupancy", "boxes", "auto_label"])
        for i, (cam, rel, ts) in enumerate(rows):
            w.writerow([rel, cam, ts, 1, 0.1, "[]", LABELS[(i // 7) % 3]])
    weather = tmp_path / "weather.csv"
    with weather.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ws_temp_c", "ws_precip_mm", "ws_humidity_pct", "grid_cell"])
        for d in range(1, 32):
            w.writerow([f"2026-05-{d:02d}", 30 + d % 3, d % 5, 60 + d, "a"])
    camera_csv = tmp_path / "camera_nodes.csv"
    camera_csv.write_text("camera_id,intersection\nCAM_A,NODE_A\nCAM_B,NODE_B\n", encoding="utf-8")
    graph = subgraph_from_arrays(path_graph(12), np.arange(12), {"NODE_A": 2, "NODE_B": 9}, k=2)
    return {"frames": frames, "weather": weather, "camera_csv": camera_csv, "graph": graph}


def test_graph_windows_pair_each_node_with_its_camera(data):
    records, _, _ = build_training_records(data["frames"], data["weather"])
    lookup = load_label_lookup(data["frames"])
    ds = GraphWindowDataset(records, "train", lookup, ["NODE_A", "NODE_B"], {"CAM_A": "NODE_A", "CAM_B": "NODE_B"}, 32)
    assert len(ds) > 0
    item = ds[0]
    assert item["images"].shape == (2, 30, 3, 32, 32) and item["target"].shape == (2,)
    assert item["visual_mask"].any(dim=1).all() and set(item["target"].tolist()) <= {0, 1, 2}


def test_unmapped_node_slots_are_masked_and_unlabelled(data):
    records, _, _ = build_training_records(data["frames"], data["weather"])
    lookup = load_label_lookup(data["frames"])
    ds = GraphWindowDataset(records, "train", lookup, ["NODE_A", "NODE_B"], {"CAM_A": "NODE_A"}, 32)   # CAM_B unmapped
    item = ds[0]
    assert not item["visual_mask"][1].any() and item["target"][1] == IGNORE_INDEX
    assert item["visual_mask"][0].any() and item["target"][0] != IGNORE_INDEX


def test_training_runs_end_to_end_and_checkpoint_round_trips(data, tmp_path):
    out = tmp_path / "ckpt" / "stgnn.pt"
    result = train_stgnn.train(
        data["frames"], data["weather"], None, None, out, camera_csv=data["camera_csv"], epochs=2,
        batch_size=2, image_size=32, patch_size=16, patch_embed_dim=8, device="cpu", graph=data["graph"])
    assert out.exists() and len(result["history"]) == 2
    assert all(math.isfinite(h["train_loss"]) for h in result["history"])
    assert result["history"][0]["val"]["n_nodes"] > 0
    assert (out.with_suffix(".metrics.csv")).read_text(encoding="utf-8").count("\n") == 3   # header + 2 epochs

    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["config"]["node_labels"] == ["NODE_A", "NODE_B"] and ckpt["config"]["camera_map"]["CAM_A"] == "NODE_A"
    cfg = ckpt["config"]
    assert cfg["temporal_dim"] == 3 + 2 and cfg["time_features"] and cfg["node_embedding"]   # weather + time in session
    fusion = CNNLSTMFusion(text_dim=768, temporal_dim=cfg["temporal_dim"], image_size=32, patch_size=16, patch_embed_dim=8)
    model = TrafficRiskModel(fusion, RADRSTGNN(in_features=fusion.lstm.hidden_size),
                             n_camera_nodes=len(cfg["node_labels"]))
    model.load_state_dict(ckpt["model"])


def test_time_in_session_features_are_appended_and_can_be_switched_off(data):
    from src.data.training_data import LabeledWindowDataset

    records, _, _ = build_training_records(data["frames"], data["weather"])
    lookup = load_label_lookup(data["frames"])
    plain = LabeledWindowDataset(records, "train", lookup, image_size=32)
    timed = LabeledWindowDataset(records, "train", lookup, image_size=32, time_features=True)
    assert plain.temporal_dim == 3 and timed.temporal_dim == 5
    a, b = plain[0], timed[0]
    assert a["temporal"].shape == (30, 3) and b["temporal"].shape == (30, 5)
    assert torch.equal(b["temporal"][:, :3], a["temporal"])                  # weather columns untouched
    position, pm = b["temporal"][:, 3], b["temporal"][:, 4]
    assert (position[1:] > position[:-1]).all() and 0 <= position.min() and position.max() <= 1
    assert torch.allclose(position[1] - position[0], torch.tensor(1 / 119))  # one minute of a 120-minute session
    assert (pm == 1.0).all()                                                 # the fixture frames are the 17:00 session


def test_the_node_embedding_starts_neutral_and_learns_a_per_intersection_bias():
    torch.manual_seed(0)
    fusion = CNNLSTMFusion(text_dim=768, temporal_dim=3, image_size=32, patch_size=16, patch_embed_dim=8)
    plain = TrafficRiskModel(fusion, RADRSTGNN(in_features=fusion.lstm.hidden_size), n_camera_nodes=0)
    biased = TrafficRiskModel(fusion, RADRSTGNN(in_features=fusion.lstm.hidden_size), n_camera_nodes=2)
    biased.load_state_dict(plain.state_dict(), strict=False)
    graph = subgraph_from_arrays(path_graph(10), np.arange(10), {"A": 2, "B": 7}, k=1)
    camera_index = torch.tensor([graph.camera_nodes["A"], graph.camera_nodes["B"]])
    batch = fake_batch()
    plain.eval(), biased.eval()
    assert torch.allclose(plain(batch, graph.a_hat, camera_index), biased(batch, graph.a_hat, camera_index), atol=1e-6)

    with torch.no_grad():
        biased.node_embedding.weight[0] += 1.0                               # only intersection A gets a bias
    diff = (biased(batch, graph.a_hat, camera_index) - plain(batch, graph.a_hat, camera_index)).abs().sum(-1)
    assert diff[:, graph.camera_nodes["A"]].min() > 0 and diff[:, graph.camera_nodes["B"]].max() < 1e-6

    biased.train()
    camera_node_loss(biased(batch, graph.a_hat, camera_index), torch.tensor([[0, 2], [1, 2]]), camera_index).backward()
    assert biased.node_embedding.weight.grad.abs().sum() > 0


def test_resume_continues_from_the_saved_epoch(data, tmp_path):
    out = tmp_path / "stgnn.pt"
    kwargs = dict(camera_csv=data["camera_csv"], batch_size=2, image_size=32, patch_size=16,
                  patch_embed_dim=8, device="cpu", graph=data["graph"])
    train_stgnn.train(data["frames"], data["weather"], None, None, out, epochs=1, **kwargs)
    result = train_stgnn.train(data["frames"], data["weather"], None, None, out, epochs=2, resume=out, **kwargs)
    assert [h["epoch"] for h in result["history"]] == [2]


def test_fusion_can_be_initialised_from_the_stage_one_checkpoint(data, tmp_path):
    spec1 = importlib.util.spec_from_file_location(
        "train_cnn_lstm", Path(__file__).resolve().parents[1] / "scripts" / "train_cnn_lstm.py")
    stage1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(stage1)
    stage1_ckpt = tmp_path / "stage1.pt"
    stage1.train(data["frames"], data["weather"], None, None, stage1_ckpt, epochs=1, batch_size=4,
                 image_size=32, patch_size=16, patch_embed_dim=8, device="cpu")

    result = train_stgnn.train(
        data["frames"], data["weather"], None, None, tmp_path / "stgnn.pt", camera_csv=data["camera_csv"],
        epochs=1, batch_size=2, image_size=32, patch_size=16, patch_embed_dim=8, device="cpu",
        graph=data["graph"], init_fusion=stage1_ckpt)
    assert math.isfinite(result["history"][0]["train_loss"])

    with pytest.raises(ValueError, match="must match"):
        train_stgnn.train(data["frames"], data["weather"], None, None, tmp_path / "x.pt",
                          camera_csv=data["camera_csv"], epochs=1, image_size=32, patch_size=16,
                          patch_embed_dim=16, device="cpu", graph=data["graph"], init_fusion=stage1_ckpt)


def test_training_refuses_to_run_when_no_camera_maps_to_a_node(data, tmp_path):
    empty_csv = tmp_path / "none.csv"
    empty_csv.write_text("camera_id,intersection\n", encoding="utf-8")
    with pytest.raises(ValueError, match="camera_nodes.csv"):
        train_stgnn.train(data["frames"], data["weather"], None, None, tmp_path / "o.pt", camera_csv=empty_csv,
                          epochs=1, image_size=32, patch_size=16, patch_embed_dim=8, device="cpu",
                          graph=data["graph"])


def test_nodes_with_several_cameras_use_every_camera_over_training_but_the_fullest_for_evaluation(data, tmp_path):
    root = data["frames"]
    for name in ("manifest.csv", "auto_labels.csv"):                         # add a second, shorter camera at NODE_A
        lines = (root / name).read_text(encoding="utf-8").splitlines()
        extra = [l.replace("CAM_A", "CAM_A2") for l in lines[1:] if "CAM_A" in l and TRAIN_DAY in l][:40]
        (root / name).write_text("\n".join(lines + extra) + "\n", encoding="utf-8")
    for l in (root / "manifest.csv").read_text(encoding="utf-8").splitlines()[1:]:
        if "CAM_A2" in l:
            src = root / l.split(",")[2].replace("CAM_A2", "CAM_A")
            dst = root / l.split(",")[2]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

    records, _, _ = build_training_records(root, data["weather"])
    lookup = load_label_lookup(root)
    mapping = {"CAM_A": "NODE_A", "CAM_A2": "NODE_A", "CAM_B": "NODE_B"}
    fixed = GraphWindowDataset(records, "train", lookup, ["NODE_A", "NODE_B"], mapping, 32)
    varied = GraphWindowDataset(records, "train", lookup, ["NODE_A", "NODE_B"], mapping, 32, sample_cameras=True)

    assert any(len(cands) == 2 for w in fixed.index for label, cands in w.items() if label == "NODE_A")
    def frames_in(base_i):
        record_index, start = fixed.base.index[base_i]
        return sum(p is not None for p in records[record_index].frame_paths[start:start + 30])

    two_camera_windows = [w["NODE_A"] for w in fixed.index if len(w.get("NODE_A", [])) == 2]
    assert two_camera_windows and all(frames_in(c[0]) >= frames_in(c[1]) for c in two_camera_windows)
    assert any(frames_in(c[0]) > frames_in(c[1]) for c in two_camera_windows)   # evaluation: fullest camera first

    used = set()
    for _ in range(12):
        for j, w in enumerate(varied.index):
            if len(w.get("NODE_A", [])) == 2:
                random.seed(j * 31 + _)
                item = varied[j]
                used.add(int(item["visual_mask"][0].sum()))                # frames present differ between the two cameras
    assert len(used) >= 2                                                    # training saw both cameras of the node
