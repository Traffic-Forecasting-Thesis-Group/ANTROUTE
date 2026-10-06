import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.alignment import WINDOW_STEPS, session_start  # noqa: E402
from src.data.graph_data import build_subgraph, flood_node_features  # noqa: E402
from src.data.graph_dataset import GraphWindowDataset  # noqa: E402
from src.data.training_data import (
    FlowFeatures,
    InferenceWindowDataset,
    build_training_records,
    load_flow_lookup,
    load_frames_table,
    load_label_lookup,
    session_of,
)  # noqa: E402
from src.models.cnn_lstm_fusion import CNNLSTMFusion  # noqa: E402
from src.models.mlp_decoder import MLPDecoder  # noqa: E402
from src.models.radr_stgnn import RADRSTGNN  # noqa: E402
from src.models.traffic_risk_model_edge import IGNORE_INDEX, TrafficRiskModel, camera_edge_ids, camera_edge_targets  # noqa: E402

EDGE_FIELDS = [
    "day",
    "session",
    "window_start",
    "window_end",
    "source_node_id",
    "target_node_id",
    "risk",
    "camera_edge",
]


def parse_session_key(key: str):
    from datetime import date

    day, session = key.split("|")
    return (date.fromisoformat(day), session)


def build_model(cfg: dict, device) -> TrafficRiskModel:
    fusion = CNNLSTMFusion(
        text_dim=cfg["text_dim"],
        temporal_dim=cfg["temporal_dim"],
        image_size=cfg["image_size"],
        patch_size=cfg["patch_size"],
        patch_embed_dim=cfg["patch_embed_dim"],
    )
    # Checkpoints from before the flood feature (v1-v2) have no "flood_features" key: road adjacency only.
    n_static = 1 if cfg.get("flood_features") else 0
    stgnn = RADRSTGNN(in_features=fusion.lstm.hidden_size + n_static)
    decoder = MLPDecoder(node_embedding_dim=stgnn.output_dim)
    n_camera_nodes = len(cfg["node_labels"]) if cfg.get("camera_embedding") else 0
    return TrafficRiskModel(fusion, stgnn, decoder, n_camera_nodes=n_camera_nodes).to(device)


def summarise(rows: List[dict]) -> dict:
    out = {}
    for split in sorted({r["split"] for r in rows}):
        labelled = [r for r in rows if r["split"] == split and r["camera_edge"] and (r["weak_target"] != "")]
        entry = {"edge_rows": sum((r["split"] == split for r in rows)), "labelled_edges": len(labelled)}
        if labelled:
            err = np.array([abs(r["risk"] - r["weak_target"]) for r in labelled])
            entry["mae"] = float(err.mean())
        out[split] = entry
    return out


def predict(
    checkpoint: Path,
    frames_roots,
    weather_csv: Path,
    out_dir: Path,
    raw_twitter_root: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
    spatial_dir: Path = REPO_ROOT / "data/processed/spatial",
    batch_size: int = 8,
    num_workers: int = 0,
    device: Optional[str] = None,
    with_labels: bool = True,
) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    node_labels, camera_map = (cfg["node_labels"], cfg["camera_map"])
    graph = build_subgraph(spatial_dir, cfg["k"])
    if not np.array_equal(graph.node_ids, cfg["graph_node_ids"]):
        raise ValueError(
            "The road subgraph differs from the one this checkpoint was trained on (different spatial files or k); scores would be misaligned."
        )
    camera_index = torch.tensor([graph.camera_nodes[label] for label in node_labels], dtype=torch.long)
    edge_ids = camera_edge_ids(graph.edge_index, camera_index, graph.n_nodes)
    is_camera_edge = torch.zeros(graph.edge_index.shape[1], dtype=torch.bool)
    is_camera_edge[edge_ids] = True
    saved = {parse_session_key(k): v for k, v in cfg["visual_split"].items()}
    frames = load_frames_table(frames_roots)
    sessions = sorted({session_of(t) for t in frames["timestamp"]})
    visual_split = {s: saved.get(s, "infer") for s in sessions}
    use_text = bool(cfg.get("use_text")) and raw_twitter_root is not None and (embeddings_path is not None)
    records, _, skipped = build_training_records(
        frames_roots,
        weather_csv,
        raw_twitter_root if use_text else None,
        embeddings_path if use_text else None,
        visual_split=visual_split,
    )
    lookup = load_label_lookup(frames_roots, human_only=True) if with_labels else None
    # The vehicle scale is the one fitted on the training split, never refitted on what is scored.
    flow = FlowFeatures(load_flow_lookup(frames_roots), cfg["vehicle_scale"]) if cfg.get("flow_features") else None
    base = InferenceWindowDataset(
        records, cfg["image_size"], cfg.get("time_features", False), lookup or None, flow=flow
    )
    dataset = GraphWindowDataset(
        records,
        "infer",
        {},
        node_labels,
        camera_map,
        cfg["image_size"],
        time_features=cfg.get("time_features", False),
        base=base,
    )
    print(
        f"{len(sessions)} sessions ({skipped} skipped for missing weather), {len(dataset)} windows, {len(node_labels)} intersections, {len(edge_ids)} camera-adjacent edges, text {('on' if use_text else 'off')}"
    )
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    a_hat = graph.a_hat.to(device)
    camera_index_d = camera_index.to(device)
    edge_index_d = graph.edge_index.to(device)
    node_features = flood_node_features(graph).to(device) if cfg.get("flood_features") else None
    node_ids = graph.node_ids
    src_np, dst_np = (graph.edge_index[0].numpy(), graph.edge_index[1].numpy())
    is_camera_edge_np = is_camera_edge.numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(batch, a_hat, camera_index_d, edge_index_d, node_features)
            risk = torch.sigmoid(logits.float()).cpu().numpy()
            weak_targets = camera_edge_targets(
                batch["target"].cpu(), camera_index, graph.n_nodes, graph.edge_index, edge_ids
            ).numpy()
            for b in range(risk.shape[0]):
                day, session, start = dataset.keys[seen + b]
                begin = session_start(day, session) + pd.Timedelta(minutes=int(start))
                end = begin + pd.Timedelta(minutes=WINDOW_STEPS)
                weak_by_edge = {int(e): float(v) for e, v in zip(edge_ids.tolist(), weak_targets[b])}
                for e in range(risk.shape[1]):
                    wt = weak_by_edge.get(e, float("nan"))
                    rows.append(
                        {
                            "day": day.isoformat(),
                            "session": session,
                            "window_start": begin.isoformat(),
                            "window_end": end.isoformat(),
                            "source_node_id": int(node_ids[src_np[e]]),
                            "target_node_id": int(node_ids[dst_np[e]]),
                            "risk": round(float(risk[b, e]), 4),
                            "camera_edge": bool(is_camera_edge_np[e]),
                            "split": visual_split.get((day, session), "infer"),
                            "weak_target": "" if np.isnan(wt) else round(wt, 4),
                        }
                    )
            seen += risk.shape[0]
    with (out_dir / "risk_edges.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EDGE_FIELDS + ["split", "weak_target"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(checkpoint),
        "windows": len(dataset),
        "sessions": len(sessions),
        "edges": graph.edge_index.shape[1],
        "camera_edges": len(edge_ids),
        "splits": summarise(rows),
    }
    (out_dir / "risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--frames-root", required=True, type=Path, nargs="+")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/risk_scores")
    p.add_argument(
        "--weather-csv", type=Path, default=REPO_ROOT / "data/processed/temporal/weatherstack_historical.csv"
    )
    p.add_argument("--raw-twitter", type=Path, default=REPO_ROOT / "data/raw/twitter")
    p.add_argument("--embeddings", type=Path, default=REPO_ROOT / "data/processed/embeddings.pt")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    summary = predict(
        a.checkpoint,
        a.frames_root,
        a.weather_csv,
        a.out_dir,
        a.raw_twitter,
        a.embeddings,
        a.spatial_dir,
        a.batch_size,
        a.num_workers,
        a.device,
        not a.no_labels,
    )
    print(json.dumps(summary["splits"], indent=2))
    print(f"wrote {a.out_dir / 'risk_edges.csv'}")


if __name__ == "__main__":
    main()