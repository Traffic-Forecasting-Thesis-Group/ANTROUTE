"""
Detect vehicles on extracted frames and write auto_labels.csv (Light / Medium / Heavy).

    pip install ultralytics
    python scripts/autolabel_frames.py "G:/My Drive/MMDA_FRAMES"
    python scripts/autolabel_frames.py "G:/My Drive/MMDA_FRAMES_B" 2026-05-20      # only these dates

Reads <frames_root>/manifest.csv (from extract_frames.py). Resumable: detections
already stored in detections.csv are reused.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.vision.congestion_autolabel import run_autolabel  # noqa: E402

DEFAULT_FRAMES_ROOT = "/content/drive/MyDrive/MMDA_FRAMES"


def _progress(iterable, desc=""):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc)
    except ImportError:
        return iterable


def main():
    args = sys.argv[1:]
    dates = [a for a in args[1:] if a[:2] == "20"]          # e.g. 2026-05-20 2026-05-22
    frames_root = Path(args[0] if args else DEFAULT_FRAMES_ROOT)
    if not (frames_root / "manifest.csv").exists():
        raise SystemExit(f"manifest.csv not found in {frames_root}; run extract_frames.py first")
    n = run_autolabel(frames_root, progress=_progress, dates=dates)
    print(f"Wrote {n} labelled frames to {frames_root / 'auto_labels.csv'}")


if __name__ == "__main__":
    main()
