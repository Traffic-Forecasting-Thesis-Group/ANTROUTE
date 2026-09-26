"""
Merge frame folders produced by separate Colab notebooks into one.

    python scripts/merge_frames.py DEST SRC1 [SRC2 ...]

Frames are MOVED from each SRC into DEST (a move inside Drive is quick), manifest.csv and
segments.csv rows are added for videos DEST does not already have, and detections.csv /
auto_labels.csv / labels.csv rows are added likewise. Nothing in DEST is overwritten, so it is
safe to run again. SRC folders are left in place, minus the frames that were moved.
"""

import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List


def _read(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _merge_csv(dest: Path, src: Path, key: str, name: str) -> int:
    """Append src/name rows whose `key` is not in dest/name; returns how many were added."""
    incoming = _read(src / name)
    if not incoming:
        return 0
    existing = {r[key] for r in _read(dest / name)}
    new = [r for r in incoming if r[key] not in existing]
    if not new:
        return 0
    target = dest / name
    write_header = not target.exists()
    with target.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(incoming[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(new)
    return len(new)


def merge(dest: Path, sources: List[Path]) -> Dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    totals = {"frames_moved": 0, "frames_already_there": 0, "manifest_rows": 0, "segment_rows": 0, "label_rows": 0}
    for src in sources:
        for row in _read(src / "manifest.csv"):
            source_file, target_file = src / row["frame_path"], dest / row["frame_path"]
            if target_file.exists():
                totals["frames_already_there"] += 1
            elif source_file.exists():
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_file), str(target_file))
                totals["frames_moved"] += 1
        totals["manifest_rows"] += _merge_csv(dest, src, "frame_path", "manifest.csv")
        totals["segment_rows"] += _merge_csv(dest, src, "source_video", "segments.csv")
        for name in ("detections.csv", "auto_labels.csv", "labels.csv"):
            totals["label_rows"] += _merge_csv(dest, src, "frame_path", name)
    return totals


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    totals = merge(Path(sys.argv[1]), [Path(s) for s in sys.argv[2:]])
    for name, value in totals.items():
        print(f"{name:22} {value}")


if __name__ == "__main__":
    main()
