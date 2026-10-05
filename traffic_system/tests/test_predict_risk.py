import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_stgnn_training import TRAIN_DAY, VAL_DAY, data, train_stgnn  # noqa: F401  (shared synthetic data + trainer)
from src.vision.label_store import save_human_label

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict_risk.py"
spec = importlib.util.spec_from_file_location("predict_risk", SCRIPT)
predict_risk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_risk)

SMALL = dict(image_size=32, patch_size=16, patch_embed_dim=8, device="cpu")


def rows_of(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def trained(data, tmp_path):
    for m in range(0, 60, 2):                                    # human labels on both days
        for day in (TRAIN_DAY, VAL_DAY):
            save_human_label(data["frames"], f"{day}/CAM_A/f_{m:03d}.jpg", ["Light", "Medium", "Heavy"][(m // 6) % 3])
            save_human_label(data["frames"], f"{day}/CAM_B/f_{m:03d}.jpg", ["Heavy", "Heavy", "Medium"][(m // 6) % 3])
    ckpt = tmp_path / "model.pt"
    train_stgnn.train(data["frames"], data["weather"], None, None, ckpt, camera_csv=data["camera_csv"], epochs=1,
                      batch_size=2, graph=data["graph"], human_only=True, split="auto", min_session_labels=10, **SMALL)
    return ckpt


def test_risk_is_the_expected_severity_of_the_class_probabilities():
    probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.2, 0.4, 0.4]])
    assert predict_risk.node_risk(probs).tolist() == pytest.approx([0.0, 0.5, 1.0, 0.6])


def test_summary_compares_with_the_majority_baseline_per_split():
    rows = [{"split": "test", "human_label": "Heavy", "predicted": "Heavy"},
            {"split": "test", "human_label": "Heavy", "predicted": "Light"},
            {"split": "test", "human_label": "Light", "predicted": "Light"},
            {"split": "infer", "human_label": "", "predicted": "Medium"}]
    out = predict_risk.summarise(rows)
    assert out["test"]["accuracy"] == pytest.approx(2 / 3) and out["test"]["majority_baseline_accuracy"] == pytest.approx(2 / 3)
    assert out["infer"] == {"windows": 1, "labelled_windows": 0}


def test_scoring_writes_valid_probabilities_and_risks_for_every_window(data, trained, tmp_path):
    out = tmp_path / "scores"
    summary = predict_risk.predict(trained, data["frames"], data["weather"], out, graph=data["graph"], batch_size=4, **{"device": "cpu"})
    nodes = rows_of(out / "risk_nodes.csv")
    assert len(nodes) == summary["windows"] * 2 and {r["intersection"] for r in nodes} == {"NODE_A", "NODE_B"}
    for r in nodes:
        p = [float(r["p_light"]), float(r["p_medium"]), float(r["p_heavy"])]
        assert sum(p) == pytest.approx(1.0, abs=2e-4) and 0.0 <= float(r["risk"]) <= 1.0
        assert float(r["risk"]) == pytest.approx(0.5 * p[1] + p[2], abs=2e-4)
        assert r["predicted"] == ["Light", "Medium", "Heavy"][int(np.argmax(p))]
        assert r["window_end"] > r["window_start"] and r["camera_present"] in ("True", "False")
    assert {r["split"] for r in nodes} == {"train", "val"}                      # the two labelled sessions
    assert any(r["human_label"] for r in nodes)
    assert json.loads((out / "risk_summary.json").read_text())["splits"]["val"]["labelled_windows"] > 0


def test_sessions_the_model_never_saw_are_scored_as_infer(data, trained, tmp_path):
    ckpt = torch.load(trained, map_location="cpu", weights_only=False)
    ckpt["config"]["visual_split"] = {k: v for k, v in ckpt["config"]["visual_split"].items() if not k.startswith(VAL_DAY)}
    torch.save(ckpt, trained)
    out = tmp_path / "scores"
    predict_risk.predict(trained, data["frames"], data["weather"], out, graph=data["graph"], device="cpu")
    nodes = rows_of(out / "risk_nodes.csv")
    assert {(r["day"], r["split"]) for r in nodes} == {(TRAIN_DAY, "train"), (VAL_DAY, "infer")}


def test_edge_scores_cover_every_road_edge_of_every_window(data, trained, tmp_path):
    out = tmp_path / "scores"
    summary = predict_risk.predict(trained, data["frames"], data["weather"], out, graph=data["graph"], device="cpu", write_edges=True)
    edges = rows_of(out / "risk_edges.csv")
    assert len(edges) == summary["windows"] * data["graph"].edge_index.shape[1]
    assert all(0.0 <= float(r["risk"]) <= 1.0 for r in edges)
    node_ids = {int(n) for n in data["graph"].node_ids}
    assert all(int(r["source_node_id"]) in node_ids and int(r["target_node_id"]) in node_ids for r in edges)


def test_scoring_refuses_a_different_road_graph_than_the_one_trained_on(data, trained, tmp_path):
    from src.data.graph_data import subgraph_from_arrays
    from test_stgnn_training import path_graph

    other = subgraph_from_arrays(path_graph(12), np.arange(50, 62), {"NODE_A": 2, "NODE_B": 9}, k=2)
    with pytest.raises(ValueError, match="subgraph differs"):
        predict_risk.predict(trained, data["frames"], data["weather"], tmp_path / "x", graph=other, device="cpu")
