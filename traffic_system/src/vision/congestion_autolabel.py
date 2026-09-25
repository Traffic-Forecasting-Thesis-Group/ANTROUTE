"""
Automatic congestion pseudo-labels from vehicle detections.

No ground-truth congestion labels exist, so each frame is labelled relative to its
own camera: occupancy (share of the frame covered by vehicle boxes) is split into
per-camera terciles -> Light / Medium / Heavy. These are pseudo-labels meant to be
reviewed and corrected in app/label_viewer.py.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

LABELS = ("Light", "Medium", "Heavy")
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO ids
DETECTION_FIELDS = ["frame_path", "n_vehicles", "occupancy", "boxes"]
AUTO_LABEL_FIELDS = ["frame_path", "camera_id", "timestamp", "n_vehicles",
                     "occupancy", "boxes", "auto_label"]


def occupancy(boxes_norm: Sequence[Sequence[float]]) -> float:
    """Summed area of normalised (x1,y1,x2,y2) boxes, capped at 1 because boxes overlap."""
    area = sum(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in boxes_norm)
    return float(min(1.0, area))


def tercile_thresholds(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    t1, t2 = np.quantile(arr, [1 / 3, 2 / 3])
    return float(t1), float(t2)


def label_from_score(score: float, t1: float, t2: float) -> str:
    if score <= t1:
        return LABELS[0]
    if score <= t2:
        return LABELS[1]
    return LABELS[2]


def assign_labels(rows: List[Dict]) -> List[Dict]:
    """Add 'auto_label' to each row using thresholds computed per camera_id."""
    by_camera: Dict[str, List[float]] = {}
    for r in rows:
        by_camera.setdefault(r["camera_id"], []).append(float(r["occupancy"]))
    thresholds = {cam: tercile_thresholds(vals) for cam, vals in by_camera.items()}
    for r in rows:
        t1, t2 = thresholds[r["camera_id"]]
        r["auto_label"] = label_from_score(float(r["occupancy"]), t1, t2)
    return rows


def _load_model(weights: str):
    from ultralytics import YOLO  # imported lazily: heavy, optional dependency
    return YOLO(weights)


def detect_batch(model, image_paths: Sequence[Path], conf: float = 0.25) -> List[Dict]:
    results = model([str(p) for p in image_paths], conf=conf, verbose=False,
                    classes=list(VEHICLE_CLASSES))
    out = []
    for res in results:
        xyxyn = res.boxes.xyxyn.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        score = res.boxes.conf.cpu().numpy()
        boxes = [[round(float(v), 4) for v in xyxyn[i]] + [int(cls[i]), round(float(score[i]), 3)]
                 for i in range(len(cls))]
        out.append({"n_vehicles": len(boxes),
                    "occupancy": round(occupancy([b[:4] for b in boxes]), 4),
                    "boxes": json.dumps(boxes)})
    return out


def run_autolabel(frames_root: Path, weights: str = "yolov8n.pt", conf: float = 0.25,
                  batch_size: int = 16, progress=lambda it, **kw: it) -> int:
    """Detect vehicles on every frame in manifest.csv (resumable) and write auto_labels.csv."""
    manifest_csv = frames_root / "manifest.csv"
    detections_csv = frames_root / "detections.csv"
    auto_csv = frames_root / "auto_labels.csv"

    from src.vision.timeline import corrected_manifest  # real-time timestamps for the viewer
    manifest = corrected_manifest(frames_root).astype(str).to_dict("records")

    done = {}
    if detections_csv.exists():
        with detections_csv.open(encoding="utf-8", newline="") as f:
            done = {r["frame_path"]: r for r in csv.DictReader(f)}

    todo = [r for r in manifest if r["frame_path"] not in done]
    if todo:
        model = _load_model(weights)
        new = not detections_csv.exists()
        with detections_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DETECTION_FIELDS)
            if new:
                writer.writeheader()
            for i in progress(range(0, len(todo), batch_size), desc="detecting"):
                chunk = todo[i:i + batch_size]
                dets = detect_batch(model, [frames_root / r["frame_path"] for r in chunk], conf)
                for r, d in zip(chunk, dets):
                    row = {"frame_path": r["frame_path"], **d}
                    writer.writerow(row)
                    done[r["frame_path"]] = row

    rows = [{"frame_path": r["frame_path"], "camera_id": r["camera_id"], "timestamp": r["timestamp"],
             **{k: done[r["frame_path"]][k] for k in ("n_vehicles", "occupancy", "boxes")}}
            for r in manifest if r["frame_path"] in done]
    assign_labels(rows)

    with auto_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUTO_LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
