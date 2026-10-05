from datetime import date

import numpy as np
import torch
from PIL import Image

from src.data.alignment import SESSION_STEPS, SessionRecord
from src.data.dataset import ANTROUTEWindowDataset
from src.models.cnn_lstm_fusion import CNNLSTMFusion


def make_record(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (640, 360), color=(255, 0, 0)).save(frame_path)

    frame_paths = [None] * SESSION_STEPS
    frame_paths[0] = str(frame_path)
    frame_paths[2] = str(frame_path)

    return SessionRecord(
        camera_id="cam1",
        day=date(2026, 5, 4),
        session="AM",
        regime="visual",
        split="train",
        frame_paths=frame_paths,
        text_steps={1: np.ones(768, dtype=np.float32)},
        weather=np.linspace(0, 1, 10, dtype=np.float32),
    )


def test_item_shapes_and_masks(tmp_path):
    dataset = ANTROUTEWindowDataset([make_record(tmp_path)], split="train")
    assert len(dataset) == 19

    item = dataset[0]
    assert item["images"].shape == (30, 3, 224, 224)
    assert item["text"].shape == (30, 768)
    assert item["temporal"].shape == (30, 10)

    assert item["visual_mask"][[0, 2]].all() and item["visual_mask"].sum() == 2
    assert item["text_mask"][1] and item["text_mask"].sum() == 1
    assert 0.0 <= item["images"].min() and item["images"].max() <= 1.0
    assert torch.all(item["images"][1] == 0)


def test_batch_runs_through_model(tmp_path):
    dataset = ANTROUTEWindowDataset([make_record(tmp_path)], split="train")
    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])

    model = CNNLSTMFusion(text_dim=768, temporal_dim=10).eval()
    with torch.no_grad():
        output = model(
            batch["images"],
            batch["text"],
            batch["temporal"],
            visual_mask=batch["visual_mask"],
            text_mask=batch["text_mask"],
        )

    assert output.shape == (2, 30, 128)
    assert torch.isfinite(output).all()

def test_missing_frame_file_is_skipped_not_crashed(tmp_path):
    """A frame path present in the session record but not actually on disk (e.g. an interrupted
    Drive sync) must be treated as no frame for that step, not raise and kill the DataLoader."""
    import warnings
    import numpy as np
    from datetime import date
    from src.data.alignment import SESSION_STEPS

    real_frame = tmp_path / "real.jpg"
    from PIL import Image
    Image.new("RGB", (8, 8)).save(real_frame)

    paths = [None] * SESSION_STEPS
    paths[0] = str(real_frame)
    paths[1] = str(tmp_path / "missing.jpg")   # never written -- simulates an incomplete sync

    record = SessionRecord(camera_id="CAM", day=date(2026, 5, 13), session="AM", regime="visual",
                           split="train", frame_paths=paths, text_steps={}, weather=np.zeros(3, dtype="float32"))
    ds = ANTROUTEWindowDataset([record], "train", regime="visual", image_size=8)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        item = ds[0]

    assert bool(item["visual_mask"][0]) is True             # the real file still loads
    assert bool(item["visual_mask"][1]) is False             # the missing one is masked out, not crashed
    assert torch.count_nonzero(item["images"][1]) == 0
    assert any("unreadable" in str(w.message) for w in caught)
