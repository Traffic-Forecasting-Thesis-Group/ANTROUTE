import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.frame_extractor import (
    discover_segments,
    extract_all,
    parse_session_window,
    segment_number,
)

FPS = 10
WINDOW = "2026.05.04.17.00.00 - 2026.05.04.19.00.00"


def write_video(path: Path, seconds: int, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    for i in range(seconds * FPS):
        writer.write(np.full((size[1], size[0], 3), i % 255, dtype=np.uint8))
    writer.release()


@pytest.fixture
def footage(tmp_path):
    dados = tmp_path / "src" / "MAY 4" / "5-7 PM" / WINDOW / "Media 1" / "CAM_A_3055" / "Dados"
    write_video(dados / "20260504_1.mp4", 130)   # frames at 0, 60, 120 s
    write_video(dados / "20260504_2.mp4", 70)    # frames at 0, 60 s
    return tmp_path / "src", tmp_path / "out"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_parse_session_window_and_segment_number():
    p = Path("x") / WINDOW / "Media 1" / "CAM" / "Dados" / "20260504_35.dar"
    assert parse_session_window(p) == (datetime(2026, 5, 4, 17), datetime(2026, 5, 4, 19))
    assert segment_number(p) == 35
    assert parse_session_window(Path("no/window/here.dar")) is None


def test_segments_are_grouped_and_ordered(footage):
    source, _ = footage
    (segments,) = discover_segments(source).values()
    assert [s.number for s in segments] == [1, 2]
    assert segments[0].camera_id == "CAM_A_3055"


def test_timestamps_accumulate_across_segments(footage):
    source, out = footage
    stats = extract_all(source, out)
    assert stats["ok"] == 2 and stats["failed"] == 0 and stats["frames"] == 5

    rows = read_csv(out / "manifest.csv")
    stamps = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    expected = [datetime(2026, 5, 4, 17, 0, 0), datetime(2026, 5, 4, 17, 1, 0), datetime(2026, 5, 4, 17, 2, 0),
                datetime(2026, 5, 4, 17, 2, 10), datetime(2026, 5, 4, 17, 3, 10)]
    for got, want in zip(stamps, expected):
        assert abs((got - want).total_seconds()) < 1.5
    assert {r["start_estimated"] for r in rows} == {"False"}
    assert list(rows[0])[:3] == ["camera_id", "timestamp", "frame_path"]
    assert all((out / r["frame_path"]).exists() for r in rows)


def test_frames_keep_aspect_and_cap_size(tmp_path):
    dados = tmp_path / "src" / WINDOW / "CAM" / "Dados"
    write_video(dados / "20260504_1.mp4", 5, size=(1280, 720))
    out = tmp_path / "out"
    extract_all(tmp_path / "src", out)
    frame = cv2.imread(str(out / read_csv(out / "manifest.csv")[0]["frame_path"]))
    assert max(frame.shape[:2]) == 640 and frame.shape[:2] == (360, 640)


def test_rerun_skips_finished_segments(footage):
    source, out = footage
    extract_all(source, out)
    stats = extract_all(source, out)
    assert stats["skipped"] == 2 and stats["ok"] == 0
    assert len(read_csv(out / "manifest.csv")) == 5


def test_failed_segment_is_recorded_and_marks_later_starts_estimated(footage):
    source, out = footage
    dados = next(source.rglob("Dados"))
    (dados / "20260504_2.mp4").unlink()
    (dados / "20260504_2.dar").write_bytes(b"not a video")
    write_video(dados / "20260504_3.mp4", 70)

    stats = extract_all(source, out)
    assert stats["failed"] == 1 and stats["ok"] == 2

    segments = {r["segment"]: r for r in read_csv(out / "segments.csv")}
    assert segments["2"]["status"] == "failed" and segments["2"]["error"]
    later = [r for r in read_csv(out / "manifest.csv") if r["segment"] == "3"]
    assert later and all(r["start_estimated"] == "True" for r in later)


def test_seek_and_sequential_sampling_agree(tmp_path):
    from src.vision.frame_extractor import _sample_by_seeking, _sample_sequentially, _valid_fps

    video = tmp_path / "v.mp4"
    write_video(video, 130)

    def run(sampler):
        cap = cv2.VideoCapture(str(video))
        out = sampler(cap, _valid_fps(cap), 60)
        cap.release()
        return out

    (seek_frames, seek_dur), (seq_frames, seq_dur) = run(_sample_by_seeking), run(_sample_sequentially)
    assert [k for k, _ in seek_frames] == [k for k, _ in seq_frames] == [0, 1, 2]
    assert abs(seek_dur - seq_dur) < 0.5
    for (_, a), (_, b) in zip(seek_frames, seq_frames):   # same moment of video, not just same count
        assert abs(float(a.mean()) - float(b.mean())) < 8
