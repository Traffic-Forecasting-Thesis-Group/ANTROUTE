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
from src.data.graph_data import build_subgraph  # noqa: E402
from src.data.graph_dataset import GraphWindowDataset  # noqa: E402
from src.data.training_data import (
    InferenceWindowDataset,
    build_training_records,
    load_frames_table,
    load_label_lookup,
    session_of,
)  # noqa: E402
from src.data.node_context import (  # noqa: E402
    FEATURES as CONTEXT_FEATURES,
    NodeContext,
    edge_features as road_edge_features,
    event_node_index,
    load_events,
)
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


def build_model(cfg: dict, device, graph=None, spatial_dir: Path = REPO_ROOT / "data/processed/spatial") -> TrafficRiskModel:
    fusion = CNNLSTMFusion(
        text_dim=cfg["text_dim"],
        temporal_dim=cfg["temporal_dim"],
        image_size=cfg["image_size"],
        patch_size=cfg["patch_size"],
        patch_embed_dim=cfg["patch_embed_dim"],
    )
    stgnn = RADRSTGNN(in_features=fusion.lstm.hidden_size)
    if cfg.get("context"):
        # Multimodal checkpoint: per-node context + per-edge road attributes.
        if graph is None:
            raise ValueError("this checkpoint uses the node context; pass the graph")
        if list(cfg["context_features"]) != list(CONTEXT_FEATURES):
            raise ValueError("the checkpoint's context features differ from src/data/node_context.py")
        edge_attr = road_edge_features(graph, spatial_dir)
        decoder = MLPDecoder(node_embedding_dim=stgnn.output_dim, edge_feature_dim=edge_attr.shape[1])
        return TrafficRiskModel(
            fusion, stgnn, decoder, context_dim=len(CONTEXT_FEATURES), edge_features=edge_attr
        ).to(device)
    decoder = MLPDecoder(node_embedding_dim=stgnn.output_dim)
    # Checkpoints trained before the flood input existed have no flood_hazard key.
    flood = graph.flood_level if cfg.get("flood_hazard") and graph is not None else None
    if cfg.get("flood_hazard") and flood is None:
        raise ValueError("this checkpoint was trained with the flood hazard input; pass the graph")
    return TrafficRiskModel(fusion, stgnn, decoder, flood_level=flood).to(device)


def build_context(cfg: dict, graph, spatial_dir: Path, weather_csv: Path, camera_index,
                  events_root: Optional[Path], landmarks_csv: Optional[Path],
                  intersections_csv: Optional[Path]) -> Optional[NodeContext]:
    """The node context a checkpoint was trained with, scaled exactly as in training."""
    if not cfg.get("context"):
        return None
    return NodeContext.build(
        graph, spatial_dir, weather_csv,
        load_events(events_root, landmarks_csv), event_node_index(graph, spatial_dir, intersections_csv),
        camera_index.tolist(),
        weather_min=np.asarray(cfg["context_weather_min"]), weather_max=np.asarray(cfg["context_weather_max"]),
        drop=tuple(cfg.get("context_drop", ())),
    )


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
    events_root: Optional[Path] = REPO_ROOT / "data/raw/twitter",
    landmarks_csv: Optional[Path] = REPO_ROOT / "configs/event_landmarks.csv",
    intersections_csv: Optional[Path] = REPO_ROOT / "configs/event_intersections.csv",
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
    node_context = build_context(
        cfg, graph, spatial_dir, weather_csv, camera_index, events_root, landmarks_csv, intersections_csv
    )
    base = InferenceWindowDataset(records, cfg["image_size"], cfg.get("time_features", False), lookup or None)
    dataset = GraphWindowDataset(
        records,
        "infer",
        {},
        node_labels,
        camera_map,
        cfg["image_size"],
        time_features=cfg.get("time_features", False),
        base=base,
        context=node_context,
    )
    print(
        f"{len(sessions)} sessions ({skipped} skipped for missing weather), {len(dataset)} windows, {len(node_labels)} intersections, {len(edge_ids)} camera-adjacent edges, text {('on' if use_text else 'off')}"
    )
    if node_context is None:
        print(
            "WARNING: checkpoint has no node context -- weather, flood, events, clock and road attributes "
            "reach only edges within 2 hops of a camera; retrain with train_stgnn_edge.py"
        )
        if not cfg.get("flood_hazard"):
            print("WARNING: checkpoint predates the flood hazard input (thesis 3.6); retrain with train_stgnn_edge.py")
    else:
        placed = sum(len(v) for v in node_context.events.values())
        print(f"node context on all {graph.n_nodes} nodes; {placed} incident reports placed")
    if not use_text:
        print("WARNING: text branch off -- event text is not reaching the risk scores (thesis 3.4)")
    model = build_model(cfg, device, graph, spatial_dir)
    model.load_state_dict(ckpt["model"])
    model.eval()
    a_hat = graph.a_hat.to(device)
    camera_index_d = camera_index.to(device)
    edge_index_d = graph.edge_index.to(device)
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
                logits = model(batch, a_hat, camera_index_d, edge_index_d)
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
        "use_text": use_text,
        "flood_hazard": bool(cfg.get("flood_hazard")) or bool(cfg.get("context")),
        "context": bool(cfg.get("context")),
        "context_drop": list(cfg.get("context_drop", [])),
        "missing_weather_days": sorted(d.isoformat() for d in node_context.missing_weather) if node_context else [],
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
    p.add_argument("--events-root", type=Path, default=REPO_ROOT / "data/raw/twitter")
    p.add_argument("--landmarks-csv", type=Path, default=REPO_ROOT / "configs/event_landmarks.csv")
    p.add_argument("--event-intersections-csv", type=Path, default=REPO_ROOT / "configs/event_intersections.csv")
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
        a.events_root,
        a.landmarks_csv,
        a.event_intersections_csv,
    )
    print(json.dumps(summary["splits"], indent=2))
    print(f"wrote {a.out_dir / 'risk_edges.csv'}")


if __name__ == "__main__":
    main()