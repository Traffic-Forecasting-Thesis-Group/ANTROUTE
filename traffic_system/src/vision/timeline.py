"""
Real-time correction for extracted frames.

Some DVR streams carry no timing information, so ffmpeg / MP4Box read them at 25 fps even
though they were recorded at 30 fps. The DVR splits recordings into 30-minute chunks, so
such chunks measure 36 container-minutes. Every duration and frame offset in a folder with
that signature is scaled by 25/30 to recover real time.

The raw facts stay untouched in segments.csv / manifest.csv; this module recomputes the
timestamps from them so the correction can be tuned without re-extracting anything.
"""

from datetime import timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.vision.frame_extractor import SAMPLE_INTERVAL_SEC, parse_session_window

CHUNK_SEC = 30 * 60           # the DVR splits recordings every 30 minutes
DEFAULT_FPS, TRUE_FPS = 25.0, 30.0
SCALE = DEFAULT_FPS / TRUE_FPS
TOLERANCE = 0.05


def folder_time_scale(durations_sec: Sequence[float]) -> float:
    """SCALE if any chunk in the folder measures ~36 min (a 30 min chunk read at 25 fps), else 1."""
    misread = CHUNK_SEC * TRUE_FPS / DEFAULT_FPS
    return SCALE if any(abs(d - misread) <= misread * TOLERANCE for d in durations_sec) else 1.0


def corrected_manifest(frames_root: Path) -> pd.DataFrame:
    """manifest.csv with real-time timestamps and start_estimated recomputed from segments.csv."""
    frames_root = Path(frames_root)
    manifest = pd.read_csv(frames_root / "manifest.csv")
    segments_path = frames_root / "segments.csv"
    if not segments_path.exists():
        return manifest

    segments = pd.read_csv(segments_path).drop_duplicates("source_video", keep="last")
    segments["folder"] = segments["source_video"].str.rsplit("/", n=1).str[0]

    info = {}
    for _, group in segments.groupby("folder"):
        group = group.sort_values("segment")
        ok = group[group["status"] == "ok"]
        scale = folder_time_scale(ok["duration_sec"])
        window = parse_session_window(Path(group["source_video"].iloc[0]))
        if window is None:
            continue
        offset, uncertain = 0.0, False
        for _, seg in group.iterrows():
            if seg["status"] == "empty":
                continue                  # a fragment with no frames: nothing is missing
            if seg["status"] != "ok":
                uncertain = True          # unknown length: later start times are lower bounds
                continue
            info[seg["source_video"]] = (window[0] + timedelta(seconds=offset), uncertain, scale)
            offset += float(seg["duration_sec"]) * scale

    keep = manifest["source_video"].isin(info)
    manifest = manifest[keep].copy()
    start = manifest["source_video"].map(lambda v: info[v][0])
    scale = manifest["source_video"].map(lambda v: info[v][2])
    manifest["timestamp"] = [
        (s + timedelta(seconds=float(k) * SAMPLE_INTERVAL_SEC * c)).isoformat()
        for s, k, c in zip(start, manifest["frame_index"], scale)
    ]
    manifest["start_estimated"] = manifest["source_video"].map(lambda v: info[v][1])
    manifest["time_scale"] = scale
    return manifest.reset_index(drop=True)
