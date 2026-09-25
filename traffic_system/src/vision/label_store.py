"""
Read/write helpers for congestion labels, kept free of any UI dependency so the
Streamlit viewer (app/label_viewer.py) and the tests can both use them.

Files under the frames root:
    auto_labels.csv   machine pseudo-labels (congestion_autolabel.py)
    labels.csv        human labels; a human label always overrides the auto label
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

LABELS = ("Light", "Medium", "Heavy")
HUMAN_FIELDS = ["frame_path", "label", "labeled_at"]


def load_frames(root: Path) -> pd.DataFrame:
    """auto_labels.csv joined with human labels, with the effective 'label' and its 'source'."""
    auto = pd.read_csv(root / "auto_labels.csv", parse_dates=["timestamp"])
    human = load_human_labels(root)
    df = auto.merge(human[["frame_path", "label"]], on="frame_path", how="left")
    df["source"] = df["label"].notna().map({True: "human", False: "auto"})
    df["label"] = df["label"].fillna(df["auto_label"])
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return df.sort_values(["camera_id", "timestamp"]).reset_index(drop=True)


def load_human_labels(root: Path) -> pd.DataFrame:
    path = root / "labels.csv"
    if not path.exists():
        return pd.DataFrame(columns=HUMAN_FIELDS)
    return pd.read_csv(path)


def save_human_label(root: Path, frame_path: str, label: str) -> None:
    """Set (or, with label=None, clear) the human label for one frame."""
    if label is not None and label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    human = load_human_labels(root)
    human = human[human["frame_path"] != frame_path]
    if label is not None:
        human.loc[len(human)] = [frame_path, label, datetime.now().isoformat(timespec="seconds")]
    human.to_csv(root / "labels.csv", index=False)


def parse_boxes(boxes_json: str) -> List[list]:
    """[[x1,y1,x2,y2,cls,conf], ...] with normalised coordinates."""
    try:
        return json.loads(boxes_json) if isinstance(boxes_json, str) else []
    except json.JSONDecodeError:
        return []
