"""
train_stgnn_edge.py -> checkpoint -> predict_congestion_risk.py -> risk_edges.csv, end to end on a
small synthetic city, for the multimodal CRS (node context on) and the CCTV-only ablation.
"""

import csv
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.node_context import FEATURES, N_FEATURES
from test_node_context import SIDE, city  # noqa: F401  (the 6x6 road grid, flood levels, weather cells)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_stgnn_edge = load_script("train_stgnn_edge")
predict_congestion_risk = load_script("predict_congestion_risk")

LABELS = ["Light", "Medium", "Heavy"]
DAYS = ("2026-05-04", "2026-05-20")   # both in alignment.VISUAL_SPLIT


@pytest.fixture
def data(tmp_path, city):  # noqa: F811
    frames = tmp_path / "frames"
    rows = []
    for cam in ("CAM_A", "CAM_B"):
        for day in DAYS:
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

    # daily weather for every grid cell of the city, a different amount of rain each day
    weather = tmp_path / "weather.csv"
    with weather.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ws_temp_c", "ws_precip_mm", "ws_humidity_pct", "grid_cell"])
        for d in range(1, 32):
            for k, cell in enumerate(city["cells"]):
                w.writerow([f"2026-05-{d:02d}", 28 + d % 5, (d * 3 + k) % 40, 55 + d + k, cell])
    camera_csv = tmp_path / "camera_nodes.csv"
    camera_csv.write_text("camera_id,intersection\nCAM_A,C0\nCAM_B,C3\n", encoding="utf-8")
    # MMDA alerts at a non-camera landmark and at a camera intersection, inside the first training session
    twitter = tmp_path / "twitter" / DAYS[0]
    twitter.mkdir(parents=True)
    alerts = [{"createdAt": "Mon May 04 09:20:00 +0000 2026", "id": "t1",
               "text": "MMDA ALERT: Road crash incident at Cam Alpha involving a motorcycle as of 5:20 PM. Two lanes occupied."}]
    (twitter / "tweets_1700_1900.json").write_text(json.dumps({"data": alerts}), encoding="utf-8")
    landmarks = tmp_path / "landmarks.csv"
    landmarks.write_text("landmark,intersection\ncam alpha,C2\n", encoding="utf-8")
    return {"frames": frames, "weather": weather, "camera_csv": camera_csv, "spatial": city["dir"],
            "twitter": tmp_path / "twitter", "landmarks": landmarks}


def train(data, out, **kw):
    return train_stgnn_edge.train(
        data["frames"], data["weather"], None, None, out, spatial_dir=data["spatial"],
        camera_csv=data["camera_csv"], k=SIDE * 2, epochs=2, batch_size=2, image_size=32, patch_size=16,
        patch_embed_dim=8, device="cpu", events_root=data["twitter"], landmarks_csv=data["landmarks"],
        intersections_csv=None, **kw,
    )


def score(data, checkpoint, out_dir):
    return predict_congestion_risk.predict(
        checkpoint, data["frames"], data["weather"], out_dir, spatial_dir=data["spatial"], device="cpu",
        with_labels=False, events_root=data["twitter"], landmarks_csv=data["landmarks"], intersections_csv=None,
    )


def test_the_multimodal_checkpoint_records_everything_needed_to_rebuild_its_inputs(data, tmp_path):
    out = tmp_path / "ckpt" / "v7.pt"
    result = train(data, out)
    assert all(np.isfinite(h["train_loss"]) for h in result["history"])
    cfg = torch.load(out, map_location="cpu", weights_only=False)["config"]
    assert cfg["context"] and cfg["edge_features"] and not cfg["flood_hazard"]   # flood now rides in the context
    assert cfg["context_features"] == FEATURES and len(cfg["context_weather_min"]) == 3
    assert cfg["context_drop"] == []


def test_scoring_writes_the_file_the_router_reads_with_a_score_on_every_edge(data, tmp_path):
    out = tmp_path / "ckpt" / "v7.pt"
    train(data, out)
    summary = score(data, out, tmp_path / "risk")
    assert summary["context"] and summary["flood_hazard"] and summary["missing_weather_days"] == []

    edges = pd.read_csv(tmp_path / "risk" / "risk_edges.csv")
    # the columns app/risk_routing.py and the Dynamic Weight Engine read, unchanged
    for column in ("window_start", "window_end", "source_node_id", "target_node_id", "risk",
                   "camera_edge", "split", "weak_target"):
        assert column in edges.columns
    assert edges["risk"].between(0, 1).all() and edges["risk"].notna().all()
    n_edges = edges.groupby(["source_node_id", "target_node_id"]).ngroups
    assert len(edges) == n_edges * edges["window_start"].nunique()
    assert edges["camera_edge"].any() and (~edges["camera_edge"]).any()


def test_the_ablation_checkpoints_still_score(data, tmp_path):
    cctv_only = tmp_path / "ckpt" / "cctv.pt"
    train(data, cctv_only, context=False)
    cfg = torch.load(cctv_only, map_location="cpu", weights_only=False)["config"]
    assert not cfg["context"] and cfg["flood_hazard"]
    assert not score(data, cctv_only, tmp_path / "risk_cctv")["context"]

    dropped = tmp_path / "ckpt" / "no_weather.pt"
    train(data, dropped, drop_context=("weather", "flood"))
    assert torch.load(dropped, map_location="cpu", weights_only=False)["config"]["context_drop"] == ["weather", "flood"]
    assert score(data, dropped, tmp_path / "risk_dropped")["context_drop"] == ["weather", "flood"]


def test_training_uses_the_split_file_exactly_and_scoring_inherits_it(data, tmp_path):
    from datetime import date

    from src.data.training_data import save_split

    split = {(date(2026, 5, 4), "PM"): "train", (date(2026, 5, 20), "PM"): "val"}
    save_split(split, tmp_path / "split.json")
    out = tmp_path / "ckpt" / "v7.pt"
    train(data, out, split_file=tmp_path / "split.json")
    cfg = torch.load(out, map_location="cpu", weights_only=False)["config"]
    assert cfg["visual_split"] == {"2026-05-04|PM": "train", "2026-05-20|PM": "val"}

    score(data, out, tmp_path / "risk")
    edges = pd.read_csv(tmp_path / "risk" / "risk_edges.csv", usecols=["day", "split"])
    assert edges.groupby("day")["split"].unique().map(list).to_dict() == {"2026-05-04": ["train"], "2026-05-20": ["val"]}


def test_a_checkpoint_scores_only_on_the_graph_it_was_trained_on(data, tmp_path):
    out = tmp_path / "ckpt" / "v7.pt"
    train(data, out)
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    ckpt["config"]["context_features"] = FEATURES[:-1]
    torch.save(ckpt, tmp_path / "ckpt" / "stale.pt")
    with pytest.raises(ValueError, match="context features"):
        score(data, tmp_path / "ckpt" / "stale.pt", tmp_path / "risk_stale")
    assert N_FEATURES == len(FEATURES)
