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
MAX_BYTES_PER_SEC = 4_000_000  # ~32 Mbps: a chunk whose output is shorter than size/this was truncated
SPS_SCAN_BYTES = 64 * 1024 * 1024  # how far into a chunk to look for the first H.264 parameter set

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
def first_sps_offset(path: Path, scan_bytes: int = SPS_SCAN_BYTES) -> Optional[int]:
    """Byte offset of the first H.264 SPS start code, or None if not found.

    MMDA .dar files are ~1 GB chunks of one continuous H.264 stream. A chunk cut mid-GOP
    begins with P-frames, so MP4Box/ffmpeg cannot identify the codec; the stream is fully
    decodable from its first SPS/keyframe onward.
    """
    with path.open("rb") as f:
        data = f.read(scan_bytes)
    i = data.find(b"\x00\x00\x01\x67")
    if i < 0:
        return None
    return i - 1 if i > 0 and data[i - 1] == 0 else i  # 4-byte start code


def _run(cmd: List[str], stdin_path: Optional[Path] = None, stdin_offset: int = 0) -> bool:
    if shutil.which(cmd[0]) is None:
        return False
    stdin = stdin_path.open("rb") if stdin_path else None
    try:
        if stdin:
            stdin.seek(stdin_offset)
        subprocess.run(cmd, stdin=stdin or subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True, timeout=CONVERT_TIMEOUT_SEC)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    finally:
        if stdin:
            stdin.close()


def _attempts(src: Path, tmp_dir: Path) -> Iterator[Tuple[str, Path]]:
    """Lazily yield (method, readable path); conversions only run if earlier ones failed."""
    is_mp4 = src.suffix.lower() == ".mp4"
    if is_mp4:
        yield "direct", src
    out = tmp_dir / "converted.mp4"

    # (method, command, byte offset to stream from, or None to let the tool open the file itself)
    commands = [
        ("mp4box", ["MP4Box", "-add", str(src), str(out)], None),
        ("ffmpeg_copy", ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(out)], None),
    ]
    offset = None if is_mp4 else first_sps_offset(src)
    if offset:  # chunk starts mid-stream: skip to the first keyframe and remux without re-encoding
        commands.insert(0, ("trim_remux",
                            ["ffmpeg", "-y", "-f", "h264", "-i", "pipe:0", "-c", "copy", str(out)], offset))
    if ALLOW_REENCODE:
        commands.append(("ffmpeg_reencode", ["ffmpeg", "-y", "-i", str(src), "-an", "-c:v", "libx264",
                                             "-preset", "ultrafast", "-crf", "28", str(out)], None))

    for method, cmd, stream_from in commands:
        if out.exists():
            out.unlink()
        ran = _run(cmd, src, stream_from) if stream_from is not None else _run(cmd)
        if ran and out.exists():
            yield method, out
    if not is_mp4:
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


def plausible_duration(duration_sec: float, size_bytes: int) -> bool:
    """False when the decoded duration is far too short for the source size. MP4Box can exit
    successfully after importing a single frame of a 1 GB chunk; that must not count as success."""
    return duration_sec * MAX_BYTES_PER_SEC >= size_bytes


def sample_frames(video_path: Path, interval_sec: int = SAMPLE_INTERVAL_SEC,
                  force_sequential: bool = False):
    """Keep one frame per `interval_sec` of video time -> ([(k, bgr_frame)], duration_sec).

    Seeks to each sample point (seconds per video); falls back to decoding the whole stream
    when seeking is unreliable for that file or when force_sequential is set.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0
    fps = _valid_fps(cap)
    result = None if force_sequential else _sample_by_seeking(cap, fps, interval_sec)
    if result is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        result = _sample_sequentially(cap, fps, interval_sec)
    cap.release()
    return result


# ---------------------------------------------------------------------------
# Extraction driver
# ---------------------------------------------------------------------------
def _sample_attempt(readable: Path, interval_sec: int, source_size: int):
    frames, duration = sample_frames(readable, interval_sec)
    if frames and not plausible_duration(duration, source_size):
        # container timing may just be wrong: count the frames that actually decode
        frames, duration = sample_frames(readable, interval_sec, force_sequential=True)
    if frames and not plausible_duration(duration, source_size):
        return [], 0.0
    return frames, duration


def _read_ok_segments(segments_csv: Path) -> Dict[str, dict]:
    """Latest attempt per video, keeping only those that succeeded."""
    if not segments_csv.exists():
        return {}
    with segments_csv.open(encoding="utf-8", newline="") as f:
        latest = {r["source_video"]: r for r in csv.DictReader(f)}
    return {k: r for k, r in latest.items() if r["status"] == "ok"}


def _drop_manifest_rows(manifest_csv: Path, source_videos: set) -> None:
    if not source_videos or not manifest_csv.exists():
        return
    with manifest_csv.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source_video"] not in source_videos]
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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
    # earlier runs (and the old notebook) accepted truncated conversions; redo those
    stale = {k for k, r in done.items()
             if Path(k).exists() and not plausible_duration(float(r["duration_sec"]), Path(k).stat().st_size)}
    for k in stale:
        del done[k]
    _drop_manifest_rows(manifest_csv, stale)
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
            size = seg.path.stat().st_size
            with tempfile.TemporaryDirectory() as tmp:
                for method, readable in _attempts(seg.path, Path(tmp)):
                    frames, duration = _sample_attempt(readable, interval_sec, size)
                    if frames:
                        break
                    error = f"{method}: no usable frames (none decoded, or far shorter than the file)"
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
    """Camera folders whose real-time duration is far from the nominal session length."""
    import collections
    from src.vision.timeline import folder_time_scale

    with segments_csv.open(encoding="utf-8", newline="") as f:
        latest = {r["source_video"]: r for r in csv.DictReader(f)}
    folders = collections.defaultdict(list)
    for video, r in latest.items():
        if r["status"] == "ok":
            folders[str(Path(video).parent)].append(float(r["duration_sec"]))

    flagged = []
    for folder, durations in folders.items():
        window = parse_session_window(Path(folder))
        total = sum(durations) * folder_time_scale(durations)
        nominal = (window[1] - window[0]).total_seconds() if window else math.nan
        if window and abs(total - nominal) > tolerance_min * 60:
            flagged.append({"folder": folder, "total_min": round(total / 60, 1),
                            "nominal_min": round(nominal / 60, 1)})
    return flagged
