import csv
from pathlib import Path

import pandas as pd
import pytest

from src.vision.frame_extractor import (
    MANIFEST_FIELDS, SEGMENT_FIELDS, extract_all, plausible_duration,
)
from src.vision.timeline import corrected_manifest, folder_time_scale
from test_frame_extractor import WINDOW, write_video

VIDEO = "/content/drive/x/" + WINDOW + "/Media 1/CAM/Dados/20260504_{n}.dar"


def write_root(root: Path, segments):
    """segments: list of (number, status, duration_sec); one manifest row per frame index 0 and 1."""
    root.mkdir(exist_ok=True)
    with (root / "segments.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEGMENT_FIELDS)
        w.writeheader()
        for n, status, dur in segments:
            w.writerow({"camera_id": "CAM", "source_video": VIDEO.format(n=n), "segment": n,
                        "status": status, "method": "mp4box", "n_frames": 2, "duration_sec": dur,
                        "segment_start": "", "start_estimated": False, "error": ""})
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for n, status, _ in segments:
            if status == "ok":
                for k in (0, 1):
                    w.writerow({"camera_id": "CAM", "timestamp": "wrong", "frame_path": f"CAM/{n}_{k}.jpg",
                                "source_video": VIDEO.format(n=n), "segment": n, "frame_index": k,
                                "segment_start": "", "segment_duration_sec": 0, "start_estimated": False})
    return root


def stamp(df, n, k):
    row = df[(df["segment"] == n) & (df["frame_index"] == k)].iloc[0]
    return pd.Timestamp(row["timestamp"])


def test_scale_detects_thirty_minute_chunks_read_at_25_fps():
    assert folder_time_scale([1962, 2160, 2160]) == pytest.approx(25 / 30)
    assert folder_time_scale([1634, 1800, 1799]) == 1.0
    assert folder_time_scale([]) == 1.0


def test_timestamps_are_rescaled_for_misread_cameras(tmp_path):
    root = write_root(tmp_path / "out", [(35, "ok", 1962), (36, "ok", 2160), (37, "ok", 2160)])
    df = corrected_manifest(root)
    start = pd.Timestamp("2026-05-04 17:00:00")
    assert stamp(df, 35, 0) == start
    assert stamp(df, 36, 0) == start + pd.Timedelta(seconds=1962 * 25 / 30)     # 17:27:15
    assert stamp(df, 36, 1) == stamp(df, 36, 0) + pd.Timedelta(seconds=50)       # 60 container s = 50 real s
    assert stamp(df, 37, 0) == start + pd.Timedelta(seconds=(1962 + 2160) * 25 / 30)
    assert set(df["time_scale"].round(4)) == {round(25 / 30, 4)}


def test_normal_cameras_keep_container_time(tmp_path):
    root = write_root(tmp_path / "out", [(35, "ok", 1800), (36, "ok", 1800)])
    df = corrected_manifest(root)
    assert stamp(df, 36, 0) == pd.Timestamp("2026-05-04 17:30:00")
    assert stamp(df, 36, 1) == pd.Timestamp("2026-05-04 17:31:00")


def test_start_is_flagged_when_an_earlier_chunk_is_missing(tmp_path):
    root = write_root(tmp_path / "out", [(35, "ok", 1800), (36, "failed", 0), (37, "ok", 1800)])
    df = corrected_manifest(root)
    assert not df[df["segment"] == 35]["start_estimated"].any()
    assert df[df["segment"] == 37]["start_estimated"].all()


def test_truncated_conversions_are_rejected():
    assert not plausible_duration(0.02, 1_000_000_000)      # 1 GB chunk turned into one frame
    assert plausible_duration(1800, 1_000_000_000)
    assert plausible_duration(2, 4_000)                     # a tiny tail file is fine


def test_stale_single_frame_segment_is_redone_without_duplicate_rows(tmp_path):
    dados = tmp_path / "src" / "MAY 4" / WINDOW / "Media 1" / "CAM" / "Dados"
    video = dados / "20260504_1.mp4"
    write_video(video, 130)
    out = tmp_path / "out"
    out.mkdir()
    key = str(video)
    with (out / "segments.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEGMENT_FIELDS)
        w.writeheader()
        w.writerow({"camera_id": "CAM", "source_video": key, "segment": 1, "status": "ok",
                    "method": "mp4box", "n_frames": 1, "duration_sec": 0.0, "segment_start": "",
                    "start_estimated": False, "error": ""})
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerow({"camera_id": "CAM", "timestamp": "old", "frame_path": "old.jpg", "source_video": key,
                    "segment": 1, "frame_index": 0, "segment_start": "", "segment_duration_sec": 0,
                    "start_estimated": False})

    stats = extract_all(tmp_path / "src", out)
    assert stats["ok"] == 1 and stats["skipped"] == 0
    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    assert len(rows) == 3 and "old.jpg" not in {r["frame_path"] for r in rows}


def test_timestamps_with_and_without_fractions_are_all_parseable(tmp_path):
    from src.vision.label_store import load_frames

    root = write_root(tmp_path / "out", [(35, "ok", 1634.13), (36, "ok", 1800)])     # fractional start of chunk 36
    df = corrected_manifest(root)
    assert pd.to_datetime(df["timestamp"]).notna().all()                            # no format clash
    assert not df["timestamp"].str.contains(r"\.").any()                            # whole seconds

    fields = ["frame_path", "camera_id", "timestamp", "n_vehicles", "occupancy", "boxes", "auto_label"]
    with (root / "auto_labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, ts in enumerate(["2026-05-04T17:00:00", "2026-05-04T17:27:14.910000"]):    # files written before the fix
            w.writerow({"frame_path": f"a{i}.jpg", "camera_id": "CAM", "timestamp": ts, "n_vehicles": 1,
                        "occupancy": 0.1, "boxes": "[]", "auto_label": "Light"})
    assert len(load_frames(root)) == 2
