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
    if (root / "auto_labels.csv").exists():
        auto = pd.read_csv(root / "auto_labels.csv")
        auto["has_auto"] = True
    else:  # frames only: no detections yet, so nothing but the frames and your own labels
        from src.vision.timeline import corrected_manifest
        auto = corrected_manifest(root)[["frame_path", "camera_id", "timestamp"]].assign(
            n_vehicles=0, occupancy=0.0, boxes="[]", auto_label="Medium", has_auto=False)
    auto["timestamp"] = pd.to_datetime(auto["timestamp"], format="ISO8601")
    human = load_human_labels(root)
    df = auto.merge(human[["frame_path", "label"]], on="frame_path", how="left")
    df["source"] = df["label"].notna().map({True: "human", False: "auto"})
    df["label"] = df["label"].fillna(df["auto_label"])
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return df.sort_values(["camera_id", "timestamp"]).reset_index(drop=True)


def label_files(root: Path) -> List[Path]:
    """Every labels*.csv in the folder: the shared legacy labels.csv, plus one per labeler
    (labels_<name>.csv). Splitting by labeler means two people labeling at once each write only
    their own file, so one person's save can never overwrite another's (a single shared labels.csv,
    read-modify-written in full on every save, would race when two people save around the same time)."""
    return sorted(root.glob("labels*.csv"))


def load_human_labels(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p).assign(source_file=p.name) for p in label_files(root) if p.stat().st_size > 0]
    if not frames:
        return pd.DataFrame(columns=HUMAN_FIELDS)
    all_labels = pd.concat(frames, ignore_index=True)
    # If the same frame was labeled in more than one file (e.g. re-labeled by a different person),
    # the most recent labeled_at wins.
    return (all_labels.sort_values("labeled_at").drop_duplicates("frame_path", keep="last")
            [HUMAN_FIELDS].reset_index(drop=True))


def _label_path(root: Path, labeler: str) -> Path:
    import re
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", labeler.strip()) if labeler and labeler.strip() else ""
    return root / (f"labels_{safe}.csv" if safe else "labels.csv")


def save_human_label(root: Path, frame_path: str, label: str, labeler: str = "") -> None:
    """Set (or, with label=None, clear) the human label for one frame, in this labeler's own file
    (labels_<labeler>.csv), so concurrent labelers never write the same file."""
    if label is not None and label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    path = _label_path(root, labeler)
    own = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame(columns=HUMAN_FIELDS)
    own = own[own["frame_path"] != frame_path]
    if label is not None:
        own.loc[len(own)] = [frame_path, label, datetime.now().isoformat(timespec="seconds")]
    own.to_csv(path, index=False)


def parse_boxes(boxes_json: str) -> List[list]:
    """[[x1,y1,x2,y2,cls,conf], ...] with normalised coordinates."""
    try:
        return json.loads(boxes_json) if isinstance(boxes_json, str) else []
    except json.JSONDecodeError:
        return []
