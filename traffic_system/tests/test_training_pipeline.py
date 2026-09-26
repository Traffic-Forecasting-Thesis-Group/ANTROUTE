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


def test_human_only_keeps_just_the_frames_a_person_reviewed(data):
    from src.vision.label_store import save_human_label

    frames = data["frames"]
    everything = load_label_lookup(frames)
    assert load_label_lookup(frames, human_only=True) == {}
    save_human_label(frames, f"{TRAIN_DAY}/CAM1/f_003.jpg", "Heavy")
    save_human_label(frames, f"{TRAIN_DAY}/CAM1/f_004.jpg", "Light")
    only = load_label_lookup(frames, human_only=True)
    assert sorted(only.values()) == [0, 2] and len(only) == 2 and len(everything) == 120
    assert all(k in everything for k in only)


def test_training_on_human_labels_only_still_runs_and_refuses_when_there_are_none(data, tmp_path):
    from src.vision.label_store import save_human_label

    common = dict(epochs=1, batch_size=4, image_size=32, patch_size=16, patch_embed_dim=8, device="cpu", human_only=True)
    with pytest.raises(ValueError, match="human_only"):
        train_cnn_lstm.train(data["frames"], data["weather"], None, None, tmp_path / "none.pt", **common)
    for m in range(0, 30, 3):
        save_human_label(data["frames"], f"{TRAIN_DAY}/CAM1/f_{m:03d}.jpg", LABELS[(m // 3) % 3])
    result = train_cnn_lstm.train(data["frames"], data["weather"], None, None, tmp_path / "human.pt", **common)
    assert math.isfinite(result["history"][0]["train_loss"]) and (tmp_path / "human.pt").exists()


def test_train_and_validation_days_can_live_in_different_folders(data, tmp_path):
    import shutil
    from src.vision.label_store import save_human_label

    train_root, val_root = tmp_path / "train_root", tmp_path / "val_root"
    for root, day in ((train_root, TRAIN_DAY), (val_root, VAL_DAY)):
        shutil.copytree(data["frames"] / day, root / day)
        for name in ("manifest.csv", "auto_labels.csv"):
            lines = (data["frames"] / name).read_text(encoding="utf-8").splitlines(keepends=True)
            (root / name).write_text("".join(l for i, l in enumerate(lines) if i == 0 or day in l), encoding="utf-8")
        for m in range(0, 60, 2):
            save_human_label(root, f"{day}/CAM1/f_{m:03d}.jpg", LABELS[(m // 4) % 3])

    roots = [train_root, val_root]
    records, _, _ = build_training_records(roots, data["weather"])
    assert {r.day.isoformat() for r in records} >= {TRAIN_DAY, VAL_DAY}
    lookup = load_label_lookup(roots, human_only=True)
    assert len(lookup) == 60 and all(str(train_root) in k or str(val_root) in k for k in lookup)

    result = train_cnn_lstm.train(roots, data["weather"], None, None, tmp_path / "multi.pt", epochs=1, batch_size=4,
                                  image_size=32, patch_size=16, patch_embed_dim=8, device="cpu", human_only=True)
    assert result["history"][0]["val"]["n_steps"] > 0          # validated on the frames of the second folder


def test_a_frame_present_in_two_folders_is_used_once_from_the_first(data, tmp_path):
    import shutil
    from src.data.training_data import load_frames_table

    copy = tmp_path / "copy"
    shutil.copytree(data["frames"], copy)
    table = load_frames_table([data["frames"], copy])
    assert len(table) == 120 and table["frame_path"].str.startswith(str(data["frames"])).all()


def test_datasets_can_be_pickled_for_dataloader_workers(data):
    import pickle
    from src.data.graph_dataset import GraphWindowDataset

    records, _, _ = build_training_records(data["frames"], data["weather"])
    lookup = load_label_lookup(data["frames"])
    plain = LabeledWindowDataset(records, "train", lookup, image_size=32)
    graph = GraphWindowDataset(records, "train", lookup, ["N"], {"CAM1": "N"}, 32)
    for ds in (plain, graph):
        clone = pickle.loads(pickle.dumps(ds))
        assert len(clone) == len(ds)
    assert torch.equal(pickle.loads(pickle.dumps(plain))[0]["target"], plain[0]["target"])


def test_training_with_worker_processes_runs(data, tmp_path):
    result = train_cnn_lstm.train(
        data["frames"], data["weather"], None, None, tmp_path / "workers.pt", epochs=1, batch_size=4,
        image_size=32, patch_size=16, patch_embed_dim=8, device="cpu", num_workers=2)
    assert math.isfinite(result["history"][0]["train_loss"])


def test_auto_split_is_70_15_15_by_whole_sessions_in_date_order():
    from datetime import date
    from src.data.alignment import VISUAL_SPLIT
    from src.data.training_data import assign_session_splits

    days = [date(2026, 5, d) for d in (4, 6, 8, 11, 13, 15, 18, 20, 22, 25)]
    everything = [(d, s) for d in days for s in ("AM", "PM")]
    split = assign_session_splits(everything)
    assert split == VISUAL_SPLIT                                   # 14 / 3 / 3: the project's official split, reproduced

    seven = assign_session_splits(everything[:7])
    assert [list(seven.values()).count(k) for k in ("train", "val", "test")] == [5, 1, 1]
    ordered = [seven[s] for s in sorted(seven)]
    assert ordered == ["train"] * 5 + ["val", "test"]              # chronological, later sessions held out


def test_auto_split_handles_small_numbers_of_sessions():
    from datetime import date
    from src.data.training_data import assign_session_splits

    s = [(date(2026, 5, d), "PM") for d in (4, 6, 8, 11, 13, 15, 18, 20, 22, 25)]
    assert list(assign_session_splits(s[:1]).values()) == ["train"]
    assert list(assign_session_splits(s[:2]).values()) == ["train", "val"]
    assert list(assign_session_splits(s[:3]).values()) == ["train", "val", "test"]
    ten = list(assign_session_splits(s).values())
    assert len(ten) == 10 and all(ten.count(k) >= 1 for k in ("train", "val", "test")) and ten.count("train") >= 6
    assert assign_session_splits([]) == {}


def test_session_of_uses_the_two_daily_sessions():
    from datetime import date
    from src.data.training_data import session_of

    assert session_of("2026-05-04T07:30:00") == (date(2026, 5, 4), "AM")
    assert session_of("2026-05-04T18:59:00") == (date(2026, 5, 4), "PM")


def test_custom_splits_reach_the_session_records_and_training(data, tmp_path):
    from datetime import date
    from src.data.training_data import labelled_sessions

    records, _, _ = build_training_records(
        data["frames"], data["weather"], visual_split={(date(2026, 5, 4), "PM"): "test", (date(2026, 5, 20), "PM"): "train"})
    assert {(r.day.isoformat(), r.split) for r in records} == {(TRAIN_DAY, "test"), (VAL_DAY, "train")}

    lookup = load_label_lookup(data["frames"])
    assert labelled_sessions(data["frames"], lookup) == [(date(2026, 5, 4), "PM"), (date(2026, 5, 20), "PM")]
    result = train_cnn_lstm.train(data["frames"], data["weather"], None, None, tmp_path / "auto.pt", epochs=1,
                                  batch_size=4, image_size=32, patch_size=16, patch_embed_dim=8, device="cpu", split="auto",
                                  min_session_labels=10)
    assert result["history"][0]["val"]["n_steps"] > 0              # two sessions: the later one became validation


def test_sessions_with_too_few_labels_are_not_used_for_the_auto_split(data):
    from datetime import date
    from src.data.training_data import labelled_sessions
    from src.vision.label_store import save_human_label

    frames = data["frames"]
    for m in range(40):
        save_human_label(frames, f"{TRAIN_DAY}/CAM1/f_{m:03d}.jpg", "Heavy")
    for m in range(3):
        save_human_label(frames, f"{VAL_DAY}/CAM1/f_{m:03d}.jpg", "Light")
    lookup = load_label_lookup(frames, human_only=True)
    assert labelled_sessions(frames, lookup, min_labels=1) == [(date(2026, 5, 4), "PM"), (date(2026, 5, 20), "PM")]
    assert labelled_sessions(frames, lookup, min_labels=10) == [(date(2026, 5, 4), "PM")]
