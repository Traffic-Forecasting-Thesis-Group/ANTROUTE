"""
Risk validation: score a risk_edges.csv against simple predictors on the same labelled rows
(src/models/risk_baselines.py), every reference fitted on the train split only.

    python scripts/evaluate_risk_baselines.py --risk-edges data/processed/risk_scores/risk_edges.csv \
        --out outputs/risk_eval/v8_validation.json

The evidence that the score driving the routes means something, on the held-out windows (test =
May 25; val = May 20 and 22):

    overall         MAE, RMSE, R2 and Spearman for the model and every reference: constants,
                    the edge x AM/PM history, the calibrated model, and nearest-camera
                    propagation (the label at the nearest other camera in the same window)
    class means     the model's mean score per Light / Medium / Heavy label, and how often a
                    Heavy row scores above a Light one
    per camera      MAE of every predictor and the model's Spearman, camera by camera
    per session / per target class, and how much of the risk varies between windows vs roads

This is edge-risk regression only; route quality is scripts/evaluate_routing.py's job.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.models.risk_baselines import (  # noqa: E402
    Cameras,
    baseline_table,
    breakdown,
    class_means,
    variance_split,
)


def load_cameras(spatial_dir: Path) -> Cameras:
    """Camera node ids and positions from the city graph the risk file was scored on."""
    from src.data.graph_data import build_full_graph

    graph = build_full_graph(spatial_dir)
    node_ids = np.asarray(graph.node_ids)
    node_camera = {int(node_ids[i]): label for label, i in graph.camera_nodes.items()}
    nodes = pd.read_csv(spatial_dir / "full_network_static_features.csv", usecols=["node_id", "lat", "lon"])
    nodes = nodes[nodes["node_id"].isin(node_camera)].set_index("node_id")
    coords = {node_camera[int(n)]: (float(r.lat), float(r.lon)) for n, r in nodes.iterrows()}
    return Cameras(node_camera, coords)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--risk-edges", type=Path, default=REPO_ROOT / "data/processed/risk_scores/risk_edges.csv")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("--no-cameras", action="store_true",
                   help="skip the camera-based parts (nearest-camera propagation, per camera)")
    p.add_argument("--out", type=Path, default=None, help="also write every table as JSON here")
    a = p.parse_args()

    frame = pd.read_csv(a.risk_edges)
    cameras = None if a.no_cameras else load_cameras(a.spatial_dir)
    pd.set_option("display.width", 220)
    overall = baseline_table(frame, cameras)
    classes = class_means(frame)
    sessions, targets = breakdown(frame, "session", cameras), breakdown(frame, "target", cameras)
    per_camera = breakdown(frame, "camera", cameras) if cameras else pd.DataFrame()
    spread = variance_split(frame)
    print("Edge-risk regression, every reference fitted on the TRAIN split only:\n")
    print(overall.round(4).to_string(index=False))
    print("\nMean model score per human label:\n" + classes.round(3).to_string(index=False))
    print("\nMAE per target class:\n" + targets.round(3).to_string(index=False))
    if not per_camera.empty:
        print("\nPer camera (MAE; model_spearman):\n" + per_camera.round(3).to_string(index=False))
    print("\nMAE per session:\n" + sessions.round(3).to_string(index=False))
    print(f"\nshare of predicted-risk variance between windows {spread['between_windows']:.0%}, "
          f"between roads {spread['between_roads']:.0%}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "risk_edges": str(a.risk_edges),
            "overall": overall.to_dict(orient="records"),
            "class_means": classes.to_dict(orient="records"),
            "per_camera": per_camera.to_dict(orient="records"),
            "per_session": sessions.to_dict(orient="records"),
            "per_target": targets.to_dict(orient="records"),
            "variance": spread,
        }, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
