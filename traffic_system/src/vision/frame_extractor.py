"""
Frame Extraction & Normalization stage of the ANTROUTE visual branch.

Turns raw CCTV segments (.dar / .mp4 / ...) into JPEG frames plus a manifest
that matches what src/data/alignment.py expects (camera_id, timestamp,
frame_path). Patch embedding is NOT done here: it is the first learned layer
of CNNLSTMFusion, so frames stay as ordinary images.

Layout expected on disk (as delivered by MMDA):
    <root>/.../<session start - session end>/.../<camera_id>/Dados/<name>_<n>.dar

Outputs (under out_root):
    <session date>/<camera_id>/<segment stem>_<k:03d>.jpg
    manifest.csv   one row per saved frame
    segments.csv   one row per source video (ok / failed, method, duration)
"""

import csv
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2

SAMPLE_INTERVAL_SEC = 60
FRAME_MAX_SIDE = 640      # aspect ratio kept; the dataset resizes to 224 at load time
JPEG_QUALITY = 95
SUPPORTED_EXTS = (".dar", ".mp4", ".avi", ".mkv")
CONVERT_TIMEOUT_SEC = 300   # a converter stuck on one file must not stall the whole run
ALLOW_REENCODE = False      # libx264 re-encode is very slow; enable only to rescue stubborn files

MANIFEST_FIELDS = [
    "camera_id", "timestamp", "frame_path", "source_video", "segment",
    "frame_index", "segment_start", "segment_duration_sec", "start_estimated",
]
SEGMENT_FIELDS = [
    "camera_id", "source_video", "segment", "status", "method", "n_frames",
    "duration_sec", "segment_start", "start_estimated", "error",
]

_STAMP = r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})"
_SESSION_RE = re.compile(_STAMP + r"\s*-\s*" + _STAMP)


@dataclass
class Segment:
    path: Path
    camera_id: str
    number: int
    window_start: datetime
    window_end: datetime


# ---------------------------------------------------------------------------
# Path parsing / discovery
# ---------------------------------------------------------------------------
def parse_session_window(path: Path) -> Optional[Tuple[datetime, datetime]]:
    """Session start/end from any ancestor folder named 'YYYY.MM.DD.HH.MM.SS - YYYY.MM.DD.HH.MM.SS'."""
    for part in reversed(path.parts):
        m = _SESSION_RE.search(part)
        if m:
            g = [int(x) for x in m.groups()]
            return datetime(*g[:6]), datetime(*g[6:])
    return None


def segment_number(path: Path) -> int:
    m = re.search(r"_(\d+)$", path.stem)
    return int(m.group(1)) if m else -1


def discover_segments(root: Path, exts=SUPPORTED_EXTS) -> Dict[Path, List[Segment]]:
    """Group videos by their containing folder, ordered by segment number."""
    groups: Dict[Path, List[Segment]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in exts:
            continue
        window = parse_session_window(path)
        if window is None:
            continue
        folder = path.parent
        camera_id = folder.parent.name if folder.name.lower() == "dados" else folder.name
        groups.setdefault(folder, []).append(
            Segment(path, camera_id, segment_number(path), window[0], window[1])
        )
    for segments in groups.values():
        segments.sort(key=lambda s: (s.number, s.path.name))
    return groups


# ---------------------------------------------------------------------------
# Reading video: several fallbacks, first one that yields frames wins
# ---------------------------------------------------------------------------
def _run(cmd: List[str]) -> bool:
    if shutil.which(cmd[0]) is None:
        return False
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
                       timeout=CONVERT_TIMEOUT_SEC)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def _attempts(src: Path, tmp_dir: Path) -> Iterator[Tuple[str, Path]]:
    """Lazily yield (method, readable path); conversions only run if earlier ones failed."""
    if src.suffix.lower() == ".mp4":
        yield "direct", src
    out = tmp_dir / "converted.mp4"
    commands = [
        ("mp4box", ["MP4Box", "-add", str(src), str(out)]),
        ("ffmpeg_copy", ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(out)]),
    ]
    if ALLOW_REENCODE:
        commands.append(("ffmpeg_reencode", ["ffmpeg", "-y", "-i", str(src), "-an", "-c:v", "libx264",
                                             "-preset", "ultrafast", "-crf", "28", str(out)]))
    for method, cmd in commands:
        if out.exists():
            out.unlink()
        if _run(cmd) and out.exists():
            yield method, out
    if src.suffix.lower() != ".mp4":
        yield "direct", src


def resize_max_side(frame, max_side: int = FRAME_MAX_SIDE):
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _valid_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps == fps and fps > 0 else 30.0


def _sample_by_seeking(cap, fps: float, interval_sec: int):
    """Jump straight to each sampling time. Returns None if the container's frame count or
    seeking cannot be trusted, so the caller can fall back to sequential decoding."""
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not n_frames or n_frames != n_frames or n_frames <= 0:
        return None
    duration = n_frames / fps
    frames = []
    for k in range(math.ceil(duration / interval_sec)):
        cap.set(cv2.CAP_PROP_POS_MSEC, k * interval_sec * 1000)
        ok, frame = cap.read()
        if not ok:
            return None
        frames.append((k, resize_max_side(frame)))
    return frames, duration


def _sample_sequentially(cap, fps: float, interval_sec: int):
    """Decode every frame and keep one per interval. Slow but works on any readable stream;
    duration comes from the frames actually decoded, not container metadata."""
    step = max(1, int(round(fps * interval_sec)))
    frames, idx = [], 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                frames.append((idx // step, resize_max_side(frame)))
        idx += 1
    return frames, idx / fps


def sample_frames(video_path: Path, interval_sec: int = SAMPLE_INTERVAL_SEC):
    """Keep one frame per `interval_sec` of video time -> ([(k, bgr_frame)], duration_sec).

    Seeks to each sample point (seconds per video); falls back to decoding the whole stream
    when seeking is unreliable for that file.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0
    fps = _valid_fps(cap)
    result = _sample_by_seeking(cap, fps, interval_sec)
    if result is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        result = _sample_sequentially(cap, fps, interval_sec)
    cap.release()
    return result


# ---------------------------------------------------------------------------
# Extraction driver
# ---------------------------------------------------------------------------
def _read_ok_segments(segments_csv: Path) -> Dict[str, dict]:
    if not segments_csv.exists():
        return {}
    with segments_csv.open(encoding="utf-8", newline="") as f:
        return {r["source_video"]: r for r in csv.DictReader(f) if r["status"] == "ok"}


def _append(path: Path, fields: List[str], rows: List[dict]) -> None:
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def extract_all(root: Path, out_root: Path, interval_sec: int = SAMPLE_INTERVAL_SEC,
                progress=lambda it, **kw: it) -> Dict[str, int]:
    """Extract frames for every segment under root. Resumable: segments already 'ok' are skipped."""
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_csv, segments_csv = out_root / "manifest.csv", out_root / "segments.csv"
    done = _read_ok_segments(segments_csv)

    groups = discover_segments(root)
    stats = {"folders": len(groups), "ok": 0, "skipped": 0, "failed": 0, "frames": 0}

    for folder, segments in progress(list(groups.items()), desc="camera folders"):
        offset = 0.0
        uncertain = False  # an earlier segment failed, so later start times are lower bounds
        for seg in segments:
            key = str(seg.path)
            if key in done:
                offset += float(done[key]["duration_sec"])
                uncertain = uncertain or done[key]["start_estimated"] == "True"
                stats["skipped"] += 1
                continue

            start = seg.window_start + timedelta(seconds=offset)
            frames, duration, method, error = [], 0.0, "", ""
            with tempfile.TemporaryDirectory() as tmp:
                for method, readable in _attempts(seg.path, Path(tmp)):
                    frames, duration = sample_frames(readable, interval_sec)
                    if frames:
                        break
                    error = f"{method}: no frames decoded"
                else:
                    method = ""
                    error = error or "no converter or reader could open this file"

            if not frames:
                _append(segments_csv, SEGMENT_FIELDS, [{
                    "camera_id": seg.camera_id, "source_video": key, "segment": seg.number,
                    "status": "failed", "method": method, "n_frames": 0, "duration_sec": 0,
                    "segment_start": start.isoformat(), "start_estimated": True, "error": error,
                }])
                uncertain = True
                stats["failed"] += 1
                continue

            day_dir = out_root / seg.window_start.strftime("%Y-%m-%d") / seg.camera_id
            day_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for k, frame in frames:
                file = day_dir / f"{seg.path.stem}_{k:03d}.jpg"
                cv2.imwrite(str(file), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                rows.append({
                    "camera_id": seg.camera_id,
                    "timestamp": (start + timedelta(seconds=k * interval_sec)).isoformat(),
                    "frame_path": file.relative_to(out_root).as_posix(),
                    "source_video": key, "segment": seg.number, "frame_index": k,
                    "segment_start": start.isoformat(),
                    "segment_duration_sec": round(duration, 2),
                    "start_estimated": uncertain,
                })
            _append(manifest_csv, MANIFEST_FIELDS, rows)
            _append(segments_csv, SEGMENT_FIELDS, [{
                "camera_id": seg.camera_id, "source_video": key, "segment": seg.number,
                "status": "ok", "method": method, "n_frames": len(rows),
                "duration_sec": round(duration, 2), "segment_start": start.isoformat(),
                "start_estimated": uncertain, "error": "",
            }])
            offset += duration
            stats["ok"] += 1
            stats["frames"] += len(rows)
    return stats


def session_coverage(segments_csv: Path, tolerance_min: float = 10.0) -> List[dict]:
    """Camera folders whose decoded duration is far from the nominal session length."""
    import collections
    totals, windows = collections.defaultdict(float), {}
    with segments_csv.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok":
                folder = str(Path(r["source_video"]).parent)
                totals[folder] += float(r["duration_sec"])
                windows[folder] = parse_session_window(Path(r["source_video"]))
    flagged = []
    for folder, total in totals.items():
        w = windows[folder]
        nominal = (w[1] - w[0]).total_seconds() if w else math.nan
        if w and abs(total - nominal) > tolerance_min * 60:
            flagged.append({"folder": folder, "total_min": round(total / 60, 1),
                            "nominal_min": round(nominal / 60, 1)})
    return flagged
