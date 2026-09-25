"""
Extract 1 frame / 60 s from every MMDA CCTV segment and write frames + manifest.

Colab:
    from google.colab import drive; drive.mount("/content/drive")
    !apt-get install -y gpac ffmpeg > /dev/null
    !git clone https://github.com/Traffic-Forecasting-Thesis-Group/ANTROUTE.git
    !pip install -q opencv-python
    !python ANTROUTE/traffic_system/scripts/extract_frames.py

Local (Drive for Desktop):
    python scripts/extract_frames.py "G:/My Drive/MMDA CCTV FOOTAGE" "G:/My Drive/MMDA_FRAMES"

Pass a single camera folder as the first argument to pilot on a small subset.
Safe to re-run: segments already extracted are skipped, failed ones are retried.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.vision.frame_extractor import extract_all, session_coverage  # noqa: E402

DEFAULT_SOURCE = "/content/drive/MyDrive/MMDA CCTV FOOTAGE"
DEFAULT_OUTPUT = "/content/drive/MyDrive/MMDA_FRAMES"


def _progress(iterable, desc=""):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc)
    except ImportError:
        return iterable


def main():
    source = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE)
    output = Path(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT)
    if not source.exists():
        raise SystemExit(f"Source folder not found: {source}")

    print(f"Source: {source}\nOutput: {output}")
    stats = extract_all(source, output, progress=_progress)

    print("\n=== Extraction summary ===")
    print(f"Camera folders:       {stats['folders']}")
    print(f"Segments extracted:   {stats['ok']}")
    print(f"Segments skipped:     {stats['skipped']} (already done)")
    print(f"Segments failed:      {stats['failed']} (see segments.csv, status=failed)")
    print(f"Segments empty:       {stats['empty']} (tail fragments under ~30 s, nothing to extract)")
    print(f"Frames written:       {stats['frames']}")

    flagged = session_coverage(output / "segments.csv")
    if flagged:
        print(f"\n{len(flagged)} camera folders differ from the nominal session length by >10 min:")
        for row in flagged[:20]:
            print(f"  {row['total_min']} min of {row['nominal_min']} min  {row['folder']}")


if __name__ == "__main__":
    main()
