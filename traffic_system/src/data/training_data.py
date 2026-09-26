"""
Builds the CNN + LSTM training data from the three real sources:

  frames   <frames_root>/manifest.csv        (extract_frames.py)
  labels   <frames_root>/auto_labels.csv + labels.csv   (autolabel + label_viewer)
  text     data/processed/embeddings.pt + data/raw/twitter (createdAt per tweet)
  weather  data/processed/temporal/weatherstack_historical.csv

and hands them to alignment.build_sessions. Only the visual regime is built
(sessions that have CCTV frames), since the congestion labels come from frames.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd
import torch

from src.data.alignment import (
    BUFFER_START,
    WINDOW_STEPS,
    LOCAL_TZ,
    SessionRecord,
    build_sessions,
)
from src.data.dataset import ANTROUTEWindowDataset
from src.vision.label_store import load_frames
from src.vision.timeline import corrected_manifest

WEATHER_COLUMNS = ["ws_temp_c", "ws_precip_mm", "ws_humidity_pct"]
CLASS_TO_IDX = {"Light": 0, "Medium": 1, "Heavy": 2}
IGNORE_INDEX = -100


def frame_key(frames_root: Path, relative_path: str) -> str:
    return str(Path(frames_root) / relative_path)


def load_frames_table(frames_root: Path, allow_estimated: bool = False) -> pd.DataFrame:
    """manifest.csv -> camera_id, timestamp, frame_path (absolute).

    Frames whose segment start is only a lower bound (an earlier segment failed) are
    dropped by default because their timestamps can be wrong.
    """
    df = corrected_manifest(Path(frames_root))
    if not allow_estimated and "start_estimated" in df:
        df = df[~df["start_estimated"].astype(str).eq("True")]
    df = df[["camera_id", "timestamp", "frame_path"]].copy()
    df["frame_path"] = [frame_key(frames_root, p) for p in df["frame_path"]]
    return df.reset_index(drop=True)


def load_weather_daily(csv_path: Path) -> pd.DataFrame:
    """One row per date: mean over that date's grid cells (no per-camera geocoding exists)."""
    df = pd.read_csv(csv_path, usecols=["date"] + WEATHER_COLUMNS)
    return df.groupby("date", as_index=False)[WEATHER_COLUMNS].mean()


def load_tweets_table(
    raw_twitter_root: Path,
    embeddings_path: Path,
    camera_ids: Sequence[str],
    days: Optional[Sequence] = None,
) -> pd.DataFrame:
    """created_at + embedding per tweet, shared by every camera (tweets are not geocoded).

    Row i of embeddings.pt is the i-th tweet in sorted(rglob('tweets_*.json')) order, exactly
    as TwitterTrafficDataset enumerated them, so createdAt is recovered by replaying that walk.
    """
    created = []
    for path in sorted(Path(raw_twitter_root).rglob("tweets_*.json")):
        with path.open(encoding="utf-8") as f:
            content = json.load(f)
        tweets = content.get("data", content) if isinstance(content, dict) else content
        created.extend(t.get("createdAt") if isinstance(t, dict) else None for t in tweets)

    payload = torch.load(embeddings_path, map_location="cpu")
    embeddings = (payload["embeddings"] if isinstance(payload, dict) else payload).float().numpy()
    if len(created) != len(embeddings):
        raise ValueError(
            f"{embeddings_path} has {len(embeddings)} rows but {raw_twitter_root} now contains "
            f"{len(created)} tweets. embeddings.pt is stale; regenerate it with main.py."
        )

    stamps = pd.to_datetime(pd.Series(created), format="%a %b %d %H:%M:%S %z %Y",
                            errors="coerce", utc=True)
    keep = stamps.notna().to_numpy().copy()
    if days is not None:
        local_dates = stamps.dt.tz_convert(LOCAL_TZ).dt.date
        keep &= local_dates.isin(set(days)).to_numpy()

    base = pd.DataFrame({
        "created_at": stamps[keep].reset_index(drop=True),
        "embedding": list(embeddings[keep]),
    })
    if base.empty or not len(camera_ids):
        return empty_tweets()
    return pd.concat([base.assign(camera_id=c) for c in camera_ids], ignore_index=True)


def empty_tweets() -> pd.DataFrame:
    return pd.DataFrame({"camera_id": pd.Series(dtype=str),
                         "created_at": pd.to_datetime(pd.Series(dtype="datetime64[ns]")),
                         "embedding": pd.Series(dtype=object)})


def build_training_records(
    frames_root: Path,
    weather_csv: Path,
    raw_twitter_root: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
):
    """Sessions for every camera on the visual-regime days. Text is skipped if no paths are given."""
    frames = load_frames_table(frames_root)
    if frames.empty:
        raise ValueError("No usable frames in manifest.csv (all missing or start_estimated).")
    camera_ids = sorted(frames["camera_id"].unique())
    days = sorted(pd.to_datetime(frames["timestamp"], format="ISO8601").dt.date.unique())

    if raw_twitter_root is not None and embeddings_path is not None:
        tweets = load_tweets_table(raw_twitter_root, embeddings_path, camera_ids, days)
    else:
        tweets = empty_tweets()

    # nonvisual_start == buffer_start makes the non-visual regime empty
    return build_sessions(
        frames, tweets, load_weather_daily(weather_csv), camera_ids, WEATHER_COLUMNS,
        nonvisual_start=BUFFER_START, buffer_start=BUFFER_START,
    )


def load_label_lookup(frames_root: Path) -> Dict[str, int]:
    """absolute frame path -> class index (human label wins over auto label)."""
    df = load_frames(Path(frames_root))
    return {frame_key(frames_root, p): CLASS_TO_IDX[l] for p, l in zip(df["frame_path"], df["label"])}


def make_target_fn(lookup: Dict[str, int]) -> Callable[[SessionRecord, int], torch.Tensor]:
    """Per-step class index for a window; IGNORE_INDEX where the step has no labelled frame."""
    def target_fn(record: SessionRecord, start: int) -> torch.Tensor:
        target = torch.full((WINDOW_STEPS,), IGNORE_INDEX, dtype=torch.long)
        for j in range(WINDOW_STEPS):
            path = record.frame_paths[start + j]
            if path is not None and path in lookup:
                target[j] = lookup[path]
        return target
    return target_fn


class LabeledWindowDataset(ANTROUTEWindowDataset):
    """ANTROUTEWindowDataset restricted to visual windows containing at least one labelled frame."""

    def __init__(self, records: Sequence[SessionRecord], split: str, lookup: Dict[str, int],
                 image_size: int = 224):
        super().__init__(records, split, regime="visual", image_size=image_size,
                         target_fn=make_target_fn(lookup))
        self.index = [
            (i, s) for i, s in self.index
            if any(records[i].frame_paths[s + j] in lookup for j in range(WINDOW_STEPS)
                   if records[i].frame_paths[s + j] is not None)
        ]
