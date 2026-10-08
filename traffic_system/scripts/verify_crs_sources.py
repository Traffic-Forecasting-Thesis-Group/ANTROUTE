"""
Check that every data source moves a trained checkpoint's Congestion Risk Score, on camera
edges and on the roads without a camera.

    python scripts/verify_crs_sources.py --checkpoint checkpoints/stgnn_edge_risk.pt \
        --frames-root /content/drive/MyDrive/MMDA_FRAMES

For a sample of real windows it reports, per source (src/models/source_sensitivity.py), how
far the risk moves when only that source changes, and the share of non-camera edges whose
risk is identical in every sampled window (89% for the CCTV-only stgnn_edge_v6). A source
whose non-camera change is ~0 is not reaching the network; report that, don't hide it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.predict_congestion_risk import build_context, build_model, parse_session_key  # noqa: E402
from src.data.graph_data import build_subgraph  # noqa: E402
from src.data.graph_dataset import GraphWindowDataset  # noqa: E402
from src.data.training_data import InferenceWindowDataset, build_training_records, load_frames_table, session_of  # noqa: E402
from src.models.source_sensitivity import SOURCES, constant_edges, source_sensitivity  # noqa: E402
from src.models.traffic_risk_model_edge import camera_edge_ids  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--frames-root", required=True, type=Path, nargs="+")
    p.add_argument("--weather-csv", type=Path, default=REPO_ROOT / "data/processed/temporal/weatherstack_historical.csv")
    p.add_argument("--raw-twitter", type=Path, default=REPO_ROOT / "data/raw/twitter")
    p.add_argument("--embeddings", type=Path, default=REPO_ROOT / "data/processed/embeddings.pt")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("--landmarks-csv", type=Path, default=REPO_ROOT / "configs/event_landmarks.csv")
    p.add_argument("--event-intersections-csv", type=Path, default=REPO_ROOT / "configs/event_intersections.csv")
    p.add_argument("--windows", type=int, default=24, help="windows sampled, spread over the sessions")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None, help="also write the result as JSON here")
    a = p.parse_args()

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    graph = build_subgraph(a.spatial_dir, cfg["k"])
    if not np.array_equal(graph.node_ids, cfg["graph_node_ids"]):
        raise SystemExit("the road subgraph differs from the one this checkpoint was trained on")
    camera_index = torch.tensor([graph.camera_nodes[label] for label in cfg["node_labels"]], dtype=torch.long)
    cam_edges = camera_edge_ids(graph.edge_index, camera_index, graph.n_nodes)

    frames = load_frames_table(a.frames_root)
    saved = {parse_session_key(k): v for k, v in cfg["visual_split"].items()}
    visual_split = {s: saved.get(s, "infer") for s in sorted({session_of(t) for t in frames["timestamp"]})}
    use_text = bool(cfg.get("use_text"))
    records, _, _ = build_training_records(
        a.frames_root, a.weather_csv, a.raw_twitter if use_text else None, a.embeddings if use_text else None,
        visual_split=visual_split,
    )
    context = build_context(cfg, graph, a.spatial_dir, a.weather_csv, camera_index,
                            a.raw_twitter, a.landmarks_csv, a.event_intersections_csv)
    base = InferenceWindowDataset(records, cfg["image_size"], cfg.get("time_features", False), None)
    dataset = GraphWindowDataset(records, "infer", {}, cfg["node_labels"], cfg["camera_map"], cfg["image_size"],
                                 time_features=cfg.get("time_features", False), base=base, context=context)
    picks = np.linspace(0, len(dataset) - 1, min(a.windows, len(dataset))).round().astype(int)
    loader = DataLoader(Subset(dataset, sorted(set(picks.tolist()))), batch_size=a.batch_size)

    model = build_model(cfg, device, graph, a.spatial_dir)
    model.load_state_dict(ckpt["model"])
    model.eval()
    a_hat, edge_index, cams = graph.a_hat.to(device), graph.edge_index.to(device), camera_index.to(device)

    totals, risks, batches = {}, [], 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "target"}
        with torch.no_grad():
            risks.append(torch.sigmoid(model(batch, a_hat, cams, edge_index).float()).cpu())
        for source, stats in source_sensitivity(model, batch, a_hat, cams, edge_index, cam_edges.to(device)).items():
            acc = totals.setdefault(source, {k: 0.0 for k in stats})
            for k, v in stats.items():
                acc[k] = max(acc[k], v) if k.endswith("_max") else acc[k] + v
        batches += 1
    for stats in totals.values():
        for k in stats:
            if not k.endswith("_max"):
                stats[k] /= batches

    result = {
        "checkpoint": str(a.checkpoint),
        "context": bool(cfg.get("context")),
        "windows": int(sum(len(r) for r in risks)),
        "constant_non_camera_edges": constant_edges(torch.cat(risks), cam_edges),
        "sources": totals,
    }
    print(f"{result['windows']} windows | context {'on' if result['context'] else 'OFF (CCTV-only checkpoint)'}")
    print(f"non-camera edges with the same risk in every window: {result['constant_non_camera_edges']:.1%}\n")
    print(f"{'source':10s} {'camera mean':>12s} {'other mean':>11s} {'other max':>10s} {'other moved':>12s}")
    for source in SOURCES:
        if source not in totals:
            print(f"{source:10s} {'-- not an input of this checkpoint --':>48s}")
            continue
        s = totals[source]
        print(f"{source:10s} {s['camera_mean']:12.4f} {s['other_mean']:11.4f} {s['other_max']:10.4f} {s['other_moved']:12.1%}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
