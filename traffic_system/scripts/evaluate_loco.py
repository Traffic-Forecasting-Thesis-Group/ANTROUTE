"""
Leave-one-camera-out (LOCO): how well does the model score a road it has never seen a camera on?

Each LOCO run retrains with one intersection's cameras removed (train_stgnn_edge.py
--hold-out-camera <label>) and scores the subgraph with that checkpoint
(predict_congestion_risk.py), so the held-out intersection's roads are scored like any road
without a camera -- from the neighbouring cameras, weather, flood, incidents, clock and road
attributes. Its human labels are not in the LOCO file (it is not a camera there), so they come
from the full model's risk_edges.csv (--reference), matched by window and edge.

    python scripts/evaluate_loco.py --reference data/processed/risk_scores/risk_edges.csv \
        --runs <loco_EDSA-Shaw/> <loco_EDSA-Ortigas/> ... --out outputs/risk_eval/loco.json

On the held-out camera's labelled edges, per split and pooled over cameras:

    LOCO model           risk from the checkpoint that never saw this camera
    full model (seen)    risk from the reference checkpoint, which trained on this camera
    train mean           the training labels' mean, without this camera
    nearest camera       the label at the nearest other camera in the same window

MAE, RMSE and Spearman. The gap between "LOCO model" and "full model" is what the camera itself
contributes; LOCO vs nearest camera and train mean says whether the model generalises to roads
without a camera better than borrowing or a constant.

Each run directory holds the risk_edges.csv and risk_summary.json predict_congestion_risk.py
writes; the summary's held_out_cameras names the camera.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.models.risk_baselines import Cameras, nearest_camera, scores, with_cameras  # noqa: E402

KEY = ["window_start", "source_node_id", "target_node_id"]
SPLITS = ("train", "val", "test")


def loco_rows(loco: pd.DataFrame, reference: pd.DataFrame, cameras: Cameras, held_out: str) -> pd.DataFrame:
    """The held-out camera's labelled edges: LOCO risk, the reference's risk and its human label."""
    ref = reference[reference["weak_target"].notna()]
    ref = with_cameras(ref, cameras)
    ref = ref[ref["camera"] == held_out]
    merged = ref[KEY + ["split", "session", "weak_target", "risk", "camera"]].rename(columns={"risk": "full_risk"}).merge(
        loco[KEY + ["risk"]].rename(columns={"risk": "loco_risk"}), on=KEY, how="inner")
    return merged


def loco_table(loco: pd.DataFrame, reference: pd.DataFrame, cameras: Cameras, held_out: str) -> List[dict]:
    """One row per (split, predictor) for one held-out camera."""
    rows = loco_rows(loco, reference, cameras, held_out)
    if rows.empty:
        return []
    labelled = with_cameras(reference[reference["weak_target"].notna()], cameras)
    others = labelled[labelled["camera"] != held_out]
    train_mean = float(others.loc[others["split"] == "train", "weak_target"].mean())
    out = []
    for split in SPLITS + ("all",):
        part = rows if split == "all" else rows[rows["split"] == split]
        if part.empty:
            continue
        y = part["weak_target"].to_numpy(float)
        predictors = {
            "LOCO model": part["loco_risk"].to_numpy(float),
            "full model (camera seen)": part["full_risk"].to_numpy(float),
            "train mean (other cameras)": np.full(len(part), train_mean),
            "nearest camera (same window)": nearest_camera(part, others, cameras, train_mean),
        }
        for name, p in predictors.items():
            out.append({"camera": held_out, "split": split, "predictor": name, **scores(y, p)})
    return out


def load_cameras(spatial_dir: Path) -> Cameras:
    import evaluate_risk_baselines

    return evaluate_risk_baselines.load_cameras(spatial_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, default=REPO_ROOT / "data/processed/risk_scores/risk_edges.csv")
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="one directory per LOCO run")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

    reference = pd.read_csv(a.reference)
    cameras = load_cameras(a.spatial_dir)
    table: List[dict] = []
    for run in a.runs:
        summary = json.loads((run / "risk_summary.json").read_text(encoding="utf-8"))
        held = summary.get("held_out_cameras") or []
        if len(held) != 1:
            raise SystemExit(f"{run}: risk_summary.json names {held!r}; expected exactly one held-out camera")
        loco = pd.read_csv(run / "risk_edges.csv", usecols=KEY + ["risk"])
        rows = loco_table(loco, reference, cameras, held[0])
        if not rows:
            print(f"{run}: no labelled rows for {held[0]} in the reference")
        table.extend(rows)
    frame = pd.DataFrame(table)
    pooled: Dict[str, dict] = {}
    for (split, predictor), part in frame.groupby(["split", "predictor"]):
        weights = part["n"].to_numpy(float)
        pooled.setdefault(split, {})[predictor] = {
            "cameras": int(len(part)), "n": int(weights.sum()),
            "mae": float(np.average(part["mae"], weights=weights)),
            "rmse": float(np.sqrt(np.average(part["rmse"] ** 2, weights=weights))),
            "mean_spearman": float(part["spearman"].mean()),
        }
    pd.set_option("display.width", 200)
    print(frame[frame["split"].isin(["val", "test"])].round(3).to_string(index=False))
    print("\npooled over cameras (n-weighted MAE / RMSE; mean of per-camera Spearman):")
    for split in ("val", "test", "all"):
        for predictor, s in pooled.get(split, {}).items():
            print(f"  {split:5s} {predictor:30s} MAE {s['mae']:.3f}  RMSE {s['rmse']:.3f}  "
                  f"Spearman {s['mean_spearman']:.3f}  ({s['cameras']} cameras, {s['n']} rows)")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"per_camera": table, "pooled": pooled}, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
