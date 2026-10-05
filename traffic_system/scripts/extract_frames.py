"""
Extract 1 frame / 60 s from every MMDA CCTV segment and write frames + manifest.

Colab:
    from google.colab import drive; drive.mount("/content/drive")
    !apt-get install -y gpac ffmpeg > /dev/null
    !git clone -b dev https://github.com/Traffic-Forecasting-Thesis-Group/ANTROUTE.git
    !pip install -q opencv-python
    !python ANTROUTE/traffic_system/scripts/extract_frames.py SRC OUT 2

    SRC   footage folder            OUT   frames folder (created, resumable)
    2     camera folders processed in parallel (default 1)

Extract only part of the data, e.g. to start training early or to split work between notebooks:
    --only "MAY 6" "MAY 8"      folders whose path contains any of these texts
    --cameras 3050 3055 8337    camera ids containing any of these texts

Videos that appear again under another path (e.g. MAY 11/MAY 4/...) are extracted once.
Safe to re-run: segments already extracted are skipped, failed ones retried.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.vision.frame_extractor import extract_all, session_coverage  # noqa: E402

DEFAULT_SOURCE = "/content/drive/MyDrive/MMDA CCTV FOOTAGE/REQ. PUP STUDENT"
DEFAULT_OUTPUT = "/content/drive/MyDrive/MMDA_FRAMES"


def _progress(iterable, desc="", total=None):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, total=total)
    except ImportError:
        return iterable


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?", default=DEFAULT_SOURCE)
    p.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    p.add_argument("workers", nargs="?", type=int, default=1)
    p.add_argument("--only", nargs="+", default=[], metavar="TEXT")
    p.add_argument("--cameras", nargs="+", default=[], metavar="TEXT")
    a = p.parse_args()

    source, output = Path(a.source), Path(a.output)
    if not source.exists():
        raise SystemExit(f"Source folder not found: {source}")

    print(f"Source: {source}\nOutput: {output}\nWorkers: {a.workers}")
    if a.only or a.cameras:
        print(f"Only folders matching {a.only or 'anything'}, cameras matching {a.cameras or 'anything'}")
    stats = extract_all(source, output, progress=_progress, workers=a.workers, only=a.only, cameras=a.cameras)

    print("\n=== Extraction summary ===")
    print(f"Camera folders:       {stats['folders']}")
    print(f"Duplicate videos:     {stats['duplicates']} (same video under another path, not repeated)")
    print(f"Segments extracted:   {stats['ok']}")
    print(f"Segments skipped:     {stats['skipped']} (already done)")
    print(f"Segments failed:      {stats['failed']} (see segments.csv, status=failed)")
    print(f"Segments empty:       {stats['empty']} (tail fragments under ~30 s, nothing to extract)")
    print(f"Frames written:       {stats['frames']}")
    if stats["errors"]:
        print(f"Folders that crashed: {stats['errors']} (see messages above; re-run to retry)")

    flagged = session_coverage(output / "segments.csv")
    if flagged:
        print(f"\n{len(flagged)} camera folders differ from the nominal session length by >10 min:")
        for row in flagged[:20]:
            print(f"  {row['total_min']} min of {row['nominal_min']} min  {row['folder']}")


if __name__ == "__main__":
    main()
