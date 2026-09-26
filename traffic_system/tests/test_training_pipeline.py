import csv
import importlib.util
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.training_data import (
    IGNORE_INDEX,
    LabeledWindowDataset,
    build_training_records,
    load_label_lookup,
    load_tweets_table,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_cnn_lstm.py"
spec = importlib.util.spec_from_file_location("train_cnn_lstm", SCRIPT)
train_cnn_lstm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_cnn_lstm)

LABELS = ["Light", "Medium", "Heavy"]
TRAIN_DAY, VAL_DAY = "2026-05-04", "2026-05-20"   # both in alignment.VISUAL_SPLIT


@pytest.fixture
def data(tmp_path):
    frames = tmp_path / "frames"
    rows = []
    for day in (TRAIN_DAY, VAL_DAY):
        (frames / day / "CAM1").mkdir(parents=True)
        for m in range(60):  # PM session 17:00-17:59, 1 frame per minute
            rel = f"{day}/CAM1/f_{m:03d}.jpg"
            cv2.imwrite(str(frames / rel), np.full((32, 32, 3), 4 * m, np.uint8))
            ts = datetime.fromisoformat(f"{day}T17:00:00") + timedelta(minutes=m)
            rows.append((rel, ts.isoformat()))

    with (frames / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera_id", "timestamp", "frame_path", "start_estimated"])
        for rel, ts in rows:
            w.writerow(["CAM1", ts, rel, False])
    with (frames / "auto_labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_path", "camera_id", "timestamp", "n_vehicles", "occupancy", "boxes", "auto_label"])
        for i, (rel, ts) in enumerate(rows):
            w.writerow([rel, "CAM1", ts, 1, 0.1, "[]", LABELS[(i // 5) % 3]])

    weather = tmp_path / "weather.csv"
    with weather.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ws_temp_c", "ws_precip_mm", "ws_humidity_pct", "grid_cell"])
        for d in range(1, 32):
            for cell in ("a", "b"):
                w.writerow([f"2026-05-{d:02d}", 30 + d % 3, d % 5, 60 + d, cell])

    twitter = tmp_path / "twitter" / TRAIN_DAY
    twitter.mkdir(parents=True)
    tweets = [{"createdAt": f"Mon May 04 09:{m}:00 +0000 2026", "text": "x"} for m in (10, 20, 30)]
    (twitter / "tweets_1700_1900.json").write_text(json.dumps({"data": tweets}), encoding="utf-8")
    embeddings = tmp_path / "embeddings.pt"
    torch.save({"embeddings": torch.randn(3, 768)}, embeddings)
    return {"frames": frames, "weather": weather, "twitter": tmp_path / "twitter", "embeddings": embeddings}


def test_tweet_times_are_recovered_in_manila_time(data):
    df = load_tweets_table(data["twitter"], data["embeddings"], ["CAM1"], days=[datetime(2026, 5, 4).date()])
    assert len(df) == 3 and set(df["camera_id"]) == {"CAM1"}
    assert df["created_at"].dt.tz_convert("Asia/Manila").dt.hour.tolist() == [17, 17, 17]


def test_stale_embeddings_are_rejected(data):
    torch.save({"embeddings": torch.randn(2, 768)}, data["embeddings"])
    with pytest.raises(ValueError, match="stale"):
        load_tweets_table(data["twitter"], data["embeddings"], ["CAM1"])


def test_sessions_windows_and_targets(data):
    records, _, skipped = build_training_records(
        data["frames"], data["weather"], data["twitter"], data["embeddings"])
    assert skipped == 0
    lookup = load_label_lookup(data["frames"])
    train = LabeledWindowDataset(records, "train", lookup, image_size=32)
    assert len(train) > 0

    item = train[0]
    assert item["images"].shape == (30, 3, 32, 32) and item["temporal"].shape == (30, 3)
    assert item["target"].shape == (30,) and set(item["target"].tolist()) <= {0, 1, 2}
    assert bool(item["text_mask"].any())            # tweets fall in the 17:00 session
    assert IGNORE_INDEX not in item["target"].tolist()[:1]


def test_training_runs_and_saves_checkpoint(data, tmp_path):
    out = tmp_path / "ckpt" / "model.pt"
    result = train_cnn_lstm.train(
        data["frames"], data["weather"], data["twitter"], data["embeddings"], out,
        epochs=2, batch_size=4, image_size=32, patch_size=16, patch_embed_dim=8, device="cpu")

    assert out.exists() and len(result["history"]) == 2
    assert all(math.isfinite(h["train_loss"]) for h in result["history"])
    assert result["history"][0]["val"]["n_steps"] > 0
    checkpoint = torch.load(out, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["patch_embed_dim"] == 8


def test_training_without_text(data, tmp_path):
    out = tmp_path / "model.pt"
    result = train_cnn_lstm.train(
        data["frames"], data["weather"], None, None, out,
        epochs=1, batch_size=4, image_size=32, patch_size=16, patch_embed_dim=8, device="cpu")
    assert out.exists() and math.isfinite(result["history"][0]["train_loss"])


def _drop_validation_day(frames: Path):
    for name in ("manifest.csv", "auto_labels.csv"):
        path = frames / name
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(l for i, l in enumerate(lines) if i == 0 or VAL_DAY not in l), encoding="utf-8")


def test_a_checkpoint_is_saved_even_without_validation_data(data, tmp_path):
    _drop_validation_day(data["frames"])                    # train-only data: the score is -loss, which is below -1
    out = tmp_path / "no_val.pt"
    result = train_cnn_lstm.train(
        data["frames"], data["weather"], None, None, out, epochs=2, batch_size=4, image_size=32,
        patch_size=16, patch_embed_dim=8, device="cpu")
    assert result["history"][0]["val"] is None and out.exists()
