import math
import pytest
import torch
import torch.nn as nn
from src.models.mlp_decoder import MLPDecoder
from src.models.radr_stgnn import RADRSTGNN
from src.models.traffic_risk_model_edge import (
    IGNORE_INDEX,
    TrafficRiskModel,
    camera_edge_ids,
    camera_edge_targets,
    edge_risk_loss,
    sample_weights_from_targets,
)


class FakeFusion(nn.Module):

    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=output_dim, batch_first=True)
        self.proj = nn.Linear(1, output_dim)

    def forward(
        self, images, text_embeddings, temporal_features, lengths=None, visual_mask=None, text_mask=None
    ):
        b, t = images.shape[:2]
        signal = images.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        return self.proj(signal)


def small_graph():
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 1, 2], [1, 2, 3, 4, 5, 6, 7, 0, 0, 1]])
    camera_index = torch.tensor([2, 5])
    return (edge_index, camera_index, 8)


def make_batch(batch_size, n_cameras, t=4, image_size=8):
    return {
        "images": torch.randn(batch_size, n_cameras, t, 3, image_size, image_size),
        "text": torch.zeros(batch_size, n_cameras, t, 4),
        "temporal": torch.zeros(batch_size, n_cameras, t, 2),
        "visual_mask": torch.ones(batch_size, n_cameras, t, dtype=torch.bool),
        "text_mask": torch.zeros(batch_size, n_cameras, t, dtype=torch.bool),
    }


def build_model(embed_dim=16):
    fusion = FakeFusion(output_dim=embed_dim)
    stgnn = RADRSTGNN(in_features=embed_dim, gcn_hidden=16, gcn_out=8, gru_hidden=12)
    decoder = MLPDecoder(node_embedding_dim=stgnn.output_dim, hidden_dim=16)
    return TrafficRiskModel(fusion, stgnn, decoder)


def test_forward_returns_one_score_per_edge():
    edge_index, camera_index, n_nodes = small_graph()
    model = build_model()
    a_hat = torch.eye(n_nodes)
    batch = make_batch(batch_size=3, n_cameras=len(camera_index))
    logits = model(batch, a_hat, camera_index, edge_index)
    assert logits.shape == (3, edge_index.shape[1])


def test_camera_edge_ids_keeps_only_touching_edges_and_rejects_none():
    edge_index, camera_index, n = small_graph()
    ids = camera_edge_ids(edge_index, camera_index, n)
    src, dst = (edge_index[0, ids], edge_index[1, ids])
    assert all((int(s) in (2, 5) or int(d) in (2, 5) for s, d in zip(src, dst)))
    with pytest.raises(ValueError):
        camera_edge_ids(torch.tensor([[0], [1]]), torch.tensor([5]), 6)


def test_camera_edge_targets_matches_hand_computed_values():
    edge_index, camera_index, n = small_graph()
    ids = camera_edge_ids(edge_index, camera_index, n)
    target = torch.tensor([[0, 2], [1, IGNORE_INDEX]])
    targets = camera_edge_targets(target, camera_index, n, edge_index, ids)
    assert targets.shape == (2, len(ids))
    src, dst = (edge_index[0, ids], edge_index[1, ids])
    for col, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
        touches_2, touches_5 = (2 in (s, d), 5 in (s, d))
        if touches_2 and (not touches_5):
            assert targets[0, col] == 0.0
        if touches_5 and (not touches_2):
            assert targets[0, col] == 1.0
    for col, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
        if 5 in (s, d) and 2 not in (s, d):
            assert math.isnan(float(targets[1, col]))


def test_edge_risk_loss_ignores_nan_and_reports_count():
    logits = torch.zeros(2, 5)
    ids = torch.tensor([0, 1, 2])
    targets = torch.tensor([[1.0, float("nan"), 0.0], [float("nan")] * 3])
    loss, count = edge_risk_loss(logits, ids, targets)
    assert count == 2
    assert float(loss) == pytest.approx(math.log(2))


def test_edge_risk_loss_with_no_labels_returns_zero_count():
    logits = torch.zeros(2, 5)
    ids = torch.tensor([0, 1])
    loss, count = edge_risk_loss(logits, ids, torch.full((2, 2), float("nan")))
    assert count == 0
    assert float(loss) == 0.0


def test_sample_weights_from_targets_maps_each_bucket_and_ignores_nan():
    weights = torch.tensor([2.0, 1.0, 0.5])  # Light, Medium, Heavy
    targets = torch.tensor([0.0, 0.5, 1.0, float("nan")])
    out = sample_weights_from_targets(targets, weights)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.5)
    # the nan row is masked out by edge_risk_loss before this is read, so its weight is
    # irrelevant -- only check it didn't crash or silently pick an out-of-range class.
    assert torch.isfinite(out[3])


def test_sample_weights_from_targets_is_none_when_class_weights_is_none():
    targets = torch.tensor([0.0, 0.5, 1.0])
    assert sample_weights_from_targets(targets, None) is None


def test_edge_risk_loss_with_class_weights_upweights_the_minority_class():
    """A Light-class error should cost more than an equally-wrong Heavy-class error when
    class_weights says Light is rarer, since that's the whole point of weighting: stop the
    loss from being dominated by whichever class has the most edges."""
    logits = torch.zeros(1, 2)  # sigmoid(0) = 0.5 for both edges: both wrong by the same margin
    ids = torch.tensor([0, 1])
    light_wrong = torch.tensor([[0.0, float("nan")]])
    heavy_wrong = torch.tensor([[float("nan"), 1.0]])
    class_weights = torch.tensor([3.0, 1.0, 1.0])  # Light weighted 3x Medium/Heavy

    light_loss, _ = edge_risk_loss(logits, ids, light_wrong, class_weights)
    heavy_loss, _ = edge_risk_loss(logits, ids, heavy_wrong, class_weights)
    assert float(light_loss) == pytest.approx(3.0 * float(heavy_loss))


def test_joint_training_reduces_loss_on_a_learnable_signal():
    torch.manual_seed(0)
    edge_index, camera_index, n_nodes = small_graph()
    edge_ids = camera_edge_ids(edge_index, camera_index, n_nodes)
    model = build_model()
    a_hat = torch.eye(n_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    def batch_and_targets(batch_size):
        batch = make_batch(batch_size, n_cameras=len(camera_index))
        classes = torch.randint(0, 3, (batch_size, len(camera_index)))
        for k in range(len(camera_index)):
            batch["images"][:, k] += classes[:, k].view(-1, 1, 1, 1, 1).float()
        targets = camera_edge_targets(classes, camera_index, n_nodes, edge_index, edge_ids)
        return (batch, targets)

    losses = []
    for _ in range(60):
        batch, targets = batch_and_targets(batch_size=8)
        logits = model(batch, a_hat, camera_index, edge_index)
        loss, count = edge_risk_loss(logits, edge_ids, targets)
        assert count > 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5