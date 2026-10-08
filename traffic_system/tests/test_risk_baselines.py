"""Edge-risk baselines (src/models/risk_baselines.py) and the loss-centring finding behind them."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.risk_baselines import baseline_table, breakdown, fit_predictors, labelled_rows, variance_split
from src.models.traffic_risk_model_edge import edge_risk_loss

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_stgnn_edge.py"
spec = importlib.util.spec_from_file_location("train_stgnn_edge", SCRIPT)
train_stgnn_edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_stgnn_edge)


def frame(rows):
    out = pd.DataFrame(rows, columns=["day", "session", "window_start", "source_node_id", "target_node_id",
                                      "risk", "camera_edge", "split", "weak_target"])
    return out


def toy():
    """Edge 1>2 is always Heavy at PM, edge 3>4 always Light; the model says 0.5 everywhere."""
    rows = []
    for split, day in (("train", "2026-05-04"), ("train", "2026-05-06"), ("test", "2026-05-25")):
        for k in range(4):
            w = f"{day}T17:{k * 5:02d}:00"
            rows.append([day, "PM", w, 1, 2, 0.5, True, split, 1.0])
            rows.append([day, "PM", w, 3, 4, 0.5, True, split, 0.0])
            rows.append([day, "PM", w, 5, 6, 0.4 + 0.01 * k, False, split, np.nan])
    return frame(rows)


def test_baselines_are_fitted_on_train_only():
    f = toy()
    # make the test split's labels extreme; the train-fitted constants must not move
    f.loc[f["split"].eq("test") & f["weak_target"].notna(), "weak_target"] = 1.0
    fitted = fit_predictors(labelled_rows(f)[lambda r: r["split"] == "train"])
    assert fitted["mean"] == pytest.approx(0.5) and fitted["history"][("1>2", "PM")] == 1.0


def test_the_table_scores_every_predictor_on_the_same_rows():
    table = baseline_table(toy())
    test = table[table["split"] == "test"].set_index("predictor")
    assert test["n"].nunique() == 1 and test["n"].iloc[0] == 8          # 4 windows x 2 camera edges
    assert test.loc["edge x AM/PM history", "mae"] == pytest.approx(0.0)  # it knows which road is which
    assert test.loc["model", "mae"] == pytest.approx(0.5)
    assert "train median" in test.index and "constant 0.5" in test.index


def test_breakdowns_and_variance_split():
    f = toy()
    by_target = breakdown(f, "target")
    assert set(by_target["target"]) == {0.0, 1.0}
    spread = variance_split(f)
    assert 0 <= spread["between_windows"] <= 1 and 0 <= spread["between_roads"] <= 1


@pytest.mark.parametrize("weighted, expected", [(False, 0.70), (True, 0.5)])
def test_balanced_class_weights_centre_the_score_at_one_half_whatever_the_label_mix(weighted, expected):
    """Root cause of stgnn_edge_v7's offset: its loss's best constant is 0.5, the labels' mean ~0.72."""
    targets = torch.tensor([[1.0] * 60 + [0.5] * 20 + [0.0] * 20])     # mean 0.70, mostly Heavy
    counts = np.array([20, 20, 60], dtype=float)                       # Light / Medium / Heavy
    class_weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32) if weighted else None
    logit = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([logit], lr=0.05)
    edge_ids = torch.arange(targets.shape[1])
    for _ in range(800):
        loss, _ = edge_risk_loss(logit.expand(1, targets.shape[1]), edge_ids, targets, class_weights)
        opt.zero_grad(); loss.backward(); opt.step()
    assert torch.sigmoid(logit).item() == pytest.approx(expected, abs=0.01)


def test_the_context_encoder_gets_its_own_learning_rate():
    from test_node_context import N_FEATURES  # noqa: F401  (ensures the module path is importable)
    from src.models.cnn_lstm_fusion import CNNLSTMFusion
    from src.models.mlp_decoder import MLPDecoder
    from src.models.radr_stgnn import RADRSTGNN
    from src.models.traffic_risk_model_edge import TrafficRiskModel

    fusion = CNNLSTMFusion(text_dim=8, temporal_dim=3, image_size=32, patch_size=16, patch_embed_dim=8,
                           lstm_hidden_dim=16, visual_feature_dim=8, text_projection_dim=8)
    model = TrafficRiskModel(fusion, RADRSTGNN(in_features=16), MLPDecoder(64), context_dim=N_FEATURES)
    default = train_stgnn_edge.param_groups(model, 1e-4, 1e-3)
    tuned = train_stgnn_edge.param_groups(model, 1e-4, 1e-3, lr_context=1e-4)
    assert [g["lr"] for g in default] == [1e-4, 1e-3, 1e-3]            # v7's behaviour is the default
    assert [g["lr"] for g in tuned] == [1e-4, 1e-3, 1e-4]
    n_params = sum(1 for _ in model.parameters())
    assert sum(len(g["params"]) for g in tuned) == n_params           # every parameter in exactly one group
