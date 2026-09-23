import pytest
import torch

from src.models.cnn_lstm_fusion import VisualCNNEncoder, CNNLSTMFusion


TEXT_DIM = 768
TEMPORAL_DIM = 10
HIDDEN = 128


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def make_inputs(B=2, T=5):
    images = torch.rand(B, T, 3, 224, 224)
    text = torch.rand(B, T, TEXT_DIM)
    temporal = torch.rand(B, T, TEMPORAL_DIM)
    return images, text, temporal


def make_model(**kwargs):
    return CNNLSTMFusion(
        text_dim=TEXT_DIM,
        temporal_dim=TEMPORAL_DIM,
        lstm_hidden_dim=HIDDEN,
        **kwargs,
    )


# Shapes

def test_visual_encoder_shape():
    images, _, _ = make_inputs()
    output = VisualCNNEncoder(visual_feature_dim=128)(images)
    assert output.shape == (2, 5, 128)


def test_fusion_shape_and_finite():
    images, text, temporal = make_inputs()
    output = make_model()(images, text, temporal)
    assert output.shape == (2, 5, HIDDEN)
    assert torch.isfinite(output).all()


def test_single_timestep_single_sample():
    images, text, temporal = make_inputs(B=1, T=1)
    output = make_model()(images, text, temporal)
    assert output.shape == (1, 1, HIDDEN)


# Input validation

@pytest.mark.parametrize(
    "images_shape, text_shape, temporal_shape",
    [
        ((2, 5, 1, 224, 224), (2, 5, TEXT_DIM), (2, 5, TEMPORAL_DIM)),  # grayscale
        ((2, 5, 3, 192, 192), (2, 5, TEXT_DIM), (2, 5, TEMPORAL_DIM)),  # wrong size
        ((2, 3, 224, 224),    (2, 5, TEXT_DIM), (2, 5, TEMPORAL_DIM)),  # missing T
        ((2, 5, 3, 224, 224), (2, 4, TEXT_DIM), (2, 5, TEMPORAL_DIM)),  # text T mismatch
        ((2, 5, 3, 224, 224), (2, 5, 512),      (2, 5, TEMPORAL_DIM)),  # wrong text dim
        ((2, 5, 3, 224, 224), (2, 5, TEXT_DIM), (2, 5, 7)),             # wrong temporal dim
    ],
)
def test_rejects_bad_shapes(images_shape, text_shape, temporal_shape):
    model = make_model()
    with pytest.raises(ValueError):
        model(
            torch.rand(images_shape),
            torch.rand(text_shape),
            torch.rand(temporal_shape),
        )


def test_rejects_bad_lengths():
    images, text, temporal = make_inputs(B=2, T=5)
    with pytest.raises(ValueError):
        make_model()(images, text, temporal, lengths=torch.tensor([0, 5]))
    with pytest.raises(ValueError):
        make_model()(images, text, temporal, lengths=torch.tensor([6, 5]))


# Gradients

def test_every_branch_receives_gradient():
    model = make_model()
    images, text, temporal = make_inputs(B=2, T=3)

    model(images, text, temporal).square().mean().backward()

    for name, p in model.named_parameters():
        if name.startswith("missing_"):
            continue  # only used when a modality is absent (tested below)
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_missing_placeholders_receive_gradient_when_used():
    model = make_model()
    images, text, temporal = make_inputs(B=2, T=3)
    mask = torch.tensor([[True, False, True], [True, True, False]])

    model(images, text, temporal, visual_mask=mask, text_mask=mask).square().mean().backward()

    assert model.missing_visual.grad.abs().sum() > 0
    assert model.missing_text.grad.abs().sum() > 0


# Behaviour

def test_lstm_is_causal():
    """Changing the last timestep must not change earlier outputs."""
    model = make_model().eval()
    images, text, temporal = make_inputs(B=2, T=4)

    with torch.no_grad():
        out1 = model(images, text, temporal)
        images[:, -1] = torch.rand_like(images[:, -1])
        text[:, -1] = torch.rand_like(text[:, -1])
        temporal[:, -1] = torch.rand_like(temporal[:, -1])
        out2 = model(images, text, temporal)

    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6)
    assert not torch.allclose(out1[:, -1], out2[:, -1])


def test_samples_are_independent_in_eval():
    model = make_model().eval()
    images, text, temporal = make_inputs(B=2, T=3)

    with torch.no_grad():
        out1 = model(images, text, temporal)
        images[1] = torch.rand_like(images[1])
        out2 = model(images, text, temporal)

    assert torch.allclose(out1[0], out2[0], atol=1e-6)


def test_padding_is_ignored():
    model = make_model().eval()
    images, text, temporal = make_inputs(B=2, T=5)
    lengths = torch.tensor([3, 5])

    with torch.no_grad():
        padded = model(images, text, temporal, lengths=lengths)
        short = model(images[:1, :3], text[:1, :3], temporal[:1, :3])

        # Garbage in the padded region must not matter
        images[0, 3:] = torch.rand_like(images[0, 3:])
        padded_again = model(images, text, temporal, lengths=lengths)

    assert torch.allclose(padded[0, :3], short[0], atol=1e-5)
    assert torch.all(padded[0, 3:] == 0)
    assert torch.allclose(padded, padded_again, atol=1e-6)


def test_missing_frames_are_not_encoded():
    model = make_model().eval()
    images, text, temporal = make_inputs(B=1, T=3)
    visual_mask = torch.tensor([[True, False, True]])

    with torch.no_grad():
        out1 = model(images, text, temporal, visual_mask=visual_mask)
        images[0, 1] = torch.rand_like(images[0, 1])
        out2 = model(images, text, temporal, visual_mask=visual_mask)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_can_overfit_tiny_batch():
    """Sanity check: the module is actually trainable end to end."""
    model = make_model(dropout=0.0)
    images, text, temporal = make_inputs(B=2, T=2)
    target = torch.rand(2, 2, HIDDEN) * 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model(images, text, temporal), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < 0.5 * losses[0]