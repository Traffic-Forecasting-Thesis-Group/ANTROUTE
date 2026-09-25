import csv

import pandas as pd
import pytest

from src.vision.congestion_autolabel import (
    assign_labels,
    label_from_score,
    occupancy,
    tercile_thresholds,
)
from src.vision.label_store import load_frames, save_human_label


def test_occupancy_sums_areas_and_caps_at_one():
    assert occupancy([]) == 0.0
    assert occupancy([[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]) == pytest.approx(0.5)
    assert occupancy([[0, 0, 1, 1], [0, 0, 1, 1]]) == 1.0


def test_terciles_split_evenly():
    values = list(range(1, 10))
    t1, t2 = tercile_thresholds(values)
    labels = [label_from_score(v, t1, t2) for v in values]
    assert [labels.count(x) for x in ("Light", "Medium", "Heavy")] == [3, 3, 3]


def test_labels_are_relative_to_each_camera():
    rows = [{"camera_id": "busy", "occupancy": v} for v in (0.5, 0.6, 0.7)]
    rows += [{"camera_id": "quiet", "occupancy": v} for v in (0.01, 0.02, 0.03)]
    out = {(r["camera_id"], r["occupancy"]): r["auto_label"] for r in assign_labels(rows)}
    assert out[("busy", 0.5)] == "Light" and out[("busy", 0.7)] == "Heavy"
    assert out[("quiet", 0.03)] == "Heavy"


def test_constant_scores_all_light():
    rows = [{"camera_id": "c", "occupancy": 0.0} for _ in range(5)]
    assert {r["auto_label"] for r in assign_labels(rows)} == {"Light"}


def make_root(tmp_path):
    fields = ["frame_path", "camera_id", "timestamp", "n_vehicles", "occupancy", "boxes", "auto_label"]
    with (tmp_path / "auto_labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, label in enumerate(("Light", "Heavy")):
            w.writerow({"frame_path": f"d/c/{i}.jpg", "camera_id": "c",
                        "timestamp": f"2026-05-04T17:0{i}:00", "n_vehicles": i,
                        "occupancy": 0.1 * i, "boxes": "[]", "auto_label": label})
    return tmp_path


def test_human_label_overrides_auto_and_can_be_cleared(tmp_path):
    root = make_root(tmp_path)
    df = load_frames(root)
    assert list(df["label"]) == ["Light", "Heavy"] and set(df["source"]) == {"auto"}

    save_human_label(root, "d/c/0.jpg", "Medium")
    df = load_frames(root)
    assert df.loc[0, "label"] == "Medium" and df.loc[0, "source"] == "human"
    assert df.loc[1, "label"] == "Heavy"

    save_human_label(root, "d/c/0.jpg", None)
    assert load_frames(root).loc[0, "label"] == "Light"
    with pytest.raises(ValueError):
        save_human_label(root, "d/c/0.jpg", "Gridlock")
