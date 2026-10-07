import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.alignment import VISUAL_SPLIT  # noqa: E402
from src.data.graph_data import DEFAULT_K, GraphData, build_subgraph, graph_sizes, resolve_camera_map  # noqa: E402
from src.data.graph_dataset import GraphWindowDataset  # noqa: E402
from src.data.training_data import (
    WEATHER_COLUMNS,
    assign_session_splits,
    build_training_records,
    describe_split,
    labelled_sessions,
    load_label_lookup,
)  # noqa: E402
from src.models.cnn_lstm_fusion import CNNLSTMFusion  # noqa: E402
from src.models.mlp_decoder import MLPDecoder  # noqa: E402
from src.models.radr_stgnn import RADRSTGNN  # noqa: E402
from src.models.traffic_risk_model_edge import (
    N_CLASSES,
    TrafficRiskModel,
    camera_edge_ids,
    camera_edge_targets,
    edge_risk_loss,
)  # noqa: E402


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def class_weights_from_lookup(lookup: Dict[str, int]) -> torch.Tensor:
    """Balanced weight per Light/Medium/Heavy class (sklearn 'balanced' convention):
    n_samples / (n_classes * n_class). Counters a label mix skewed toward Heavy, which
    otherwise pulls BCE's optimum toward predicting close to the dataset's mean label
    everywhere instead of learning real per-edge differences."""
    counts = np.bincount(list(lookup.values()), minlength=N_CLASSES).astype(float)
    return torch.tensor(
        np.where(counts > 0, counts.sum() / (N_CLASSES * np.maximum(counts, 1)), 1.0),
        dtype=torch.float32,
    )


@torch.no_grad()
def evaluate(model, loader, a_hat, camera_index, edge_index, edge_ids, n_nodes, device, class_weights=None) -> Optional[dict]:
    if len(loader.dataset) == 0:
        return None
    model.eval()
    total_loss, total_count, abs_err = (0.0, 0, 0.0)
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch, a_hat, camera_index, edge_index)
        targets = camera_edge_targets(batch["target"], camera_index, n_nodes, edge_index, edge_ids)
        loss, count = edge_risk_loss(logits, edge_ids, targets, class_weights)
        if count == 0:
            continue
        mask = ~torch.isnan(targets)
        total_loss += float(loss) * count
        total_count += count
        abs_err += float((torch.sigmoid(logits[:, edge_ids])[mask] - targets[mask]).abs().sum())
    if total_count == 0:
        return None
    return {"loss": total_loss / total_count, "mae": abs_err / total_count, "n_edges": total_count}


def train(
    frames_root,
    weather_csv: Path,
    raw_twitter_root: Optional[Path],
    embeddings_path: Optional[Path],
    out_path: Path,
    spatial_dir: Path = REPO_ROOT / "data/processed/spatial",
    camera_csv: Optional[Path] = REPO_ROOT / "configs/camera_nodes.csv",
    k: int = DEFAULT_K,
    epochs: int = 10,
    batch_size: int = 4,
    lr_fusion: float = 0.0001,
    lr_graph: float = 0.001,
    weight_decay: float = 0.0001,
    image_size: int = 224,
    patch_size: int = 16,
    patch_embed_dim: int = 768,
    num_workers: int = 0,
    max_train_windows: Optional[int] = None,
    device: Optional[str] = None,
    seed: int = 0,
    graph: Optional[GraphData] = None,
    human_only: bool = False,
    split: str = "official",
    min_session_labels: int = 100,
    time_features: bool = False,
    use_class_weights: bool = False,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph = graph or build_subgraph(spatial_dir, k)
    lookup = load_label_lookup(frames_root, human_only)
    class_weights = class_weights_from_lookup(lookup).to(device) if use_class_weights else None
    if class_weights is not None:
        print(f"class weights (Light/Medium/Heavy): {class_weights.tolist()}")
    visual_split = None
    if split == "auto":
        visual_split = assign_session_splits(labelled_sessions(frames_root, lookup, min_session_labels))
        print("70/15/15 split over the labelled sessions:\n" + describe_split(visual_split))
    records, scaler, skipped = build_training_records(
        frames_root, weather_csv, raw_twitter_root, embeddings_path, visual_split
    )
    if not lookup:
        raise ValueError("No labelled frames (with human_only, review frames in the viewer first).")
    camera_ids = sorted({r.camera_id for r in records})
    camera_map, unmapped = resolve_camera_map(camera_ids, list(graph.camera_nodes), camera_csv)
    print(
        f"graph: {graph.n_nodes} nodes, {graph.edge_index.shape[1]} edges | {len(camera_map)}/{len(camera_ids)} cameras mapped to intersections"
    )
    if unmapped:
        print(
            "  unmapped cameras (add them to configs/camera_nodes.csv):",
            ", ".join((c[-12:] for c in unmapped)),
        )
    if not camera_map:
        raise ValueError("No camera could be mapped to a graph intersection; fill configs/camera_nodes.csv.")
    node_labels = sorted(set(camera_map.values()))
    camera_index = torch.tensor(
        [graph.camera_nodes[label] for label in node_labels], dtype=torch.long, device=device
    )
    a_hat = graph.a_hat.to(device)
    edge_index = graph.edge_index.to(device)
    edge_ids = camera_edge_ids(graph.edge_index, camera_index.cpu(), graph.n_nodes).to(device)
    print(f"{len(edge_ids)} of {graph.edge_index.shape[1]} edges touch a camera node")
    datasets = {
        s: GraphWindowDataset(
            records,
            s,
            lookup,
            node_labels,
            camera_map,
            image_size,
            sample_cameras=s == "train",
            time_features=time_features,
        )
        for s in ("train", "val", "test")
    }
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} windows over {len(node_labels)} intersections")
    if len(datasets["train"]) == 0:
        raise ValueError("No labelled training windows. Extract, auto-label, review and map cameras first.")
    if max_train_windows and len(datasets["train"]) > max_train_windows:
        keep = sorted(np.random.permutation(len(datasets["train"]))[:max_train_windows])
        datasets["train"].index = [datasets["train"].index[i] for i in keep]
    loaders = {
        s: DataLoader(ds, batch_size=batch_size, shuffle=s == "train", num_workers=num_workers)
        for s, ds in datasets.items()
    }
    temporal_dim = datasets["train"].weather_dim
    fusion = CNNLSTMFusion(
        text_dim=768,
        temporal_dim=temporal_dim,
        image_size=image_size,
        patch_size=patch_size,
        patch_embed_dim=patch_embed_dim,
    )
    stgnn = RADRSTGNN(in_features=fusion.lstm.hidden_size)
    decoder = MLPDecoder(node_embedding_dim=stgnn.output_dim)
    # Project NOAH flood hazard level per node: the static spatial node attribute (thesis 3.6).
    model = TrafficRiskModel(fusion, stgnn, decoder, flood_level=graph.flood_level).to(device)
    fusion_params = set((id(p) for p in model.fusion.parameters()))
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if id(p) in fusion_params], "lr": lr_fusion},
            {"params": [p for p in model.parameters() if id(p) not in fusion_params], "lr": lr_graph},
        ],
        weight_decay=weight_decay,
    )
    amp = device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    config = {
        "k": k,
        "image_size": image_size,
        "patch_size": patch_size,
        "patch_embed_dim": patch_embed_dim,
        "text_dim": 768,
        "temporal_dim": temporal_dim,
        "time_features": time_features,
        "use_class_weights": use_class_weights,
        "use_text": raw_twitter_root is not None,
        "flood_hazard": graph.flood_level is not None,
        "visual_split": {f"{d.isoformat()}|{s}": v for (d, s), v in (visual_split or VISUAL_SPLIT).items()},
        "weather_columns": WEATHER_COLUMNS,
        "node_labels": node_labels,
        "camera_map": camera_map,
        "graph_node_ids": graph.node_ids,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_path.with_suffix(".metrics.csv")
    history, best = ([], float("inf"))
    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        for batch in loaders["train"]:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(batch, a_hat, camera_index, edge_index)
                targets = camera_edge_targets(
                    batch["target"], camera_index, graph.n_nodes, edge_index, edge_ids
                )
                loss, count = edge_risk_loss(logits.float(), edge_ids, targets, class_weights)
            if count == 0:
                continue
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            running.append(float(loss.detach()))
        val = evaluate(
            model, loaders["val"], a_hat, camera_index, edge_index, edge_ids, graph.n_nodes, device, class_weights
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(running)) if running else float("nan"), "val": val}
        history.append(row)
        print(f"epoch {epoch}: train_loss={row['train_loss']:.4f} val={val}")
        with metrics_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if metrics_csv.stat().st_size == 0:
                writer.writerow(["epoch", "train_loss", "val_loss", "val_mae", "val_n_edges"])
            writer.writerow(
                [epoch, row["train_loss"]]
                + ([val["loss"], val["mae"], val["n_edges"]] if val else ["", "", ""])
            )
        score = val["loss"] if val else row["train_loss"]
        if score < best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "config": config,
                    "weather_min": scaler.min_,
                    "weather_max": scaler.max_,
                },
                out_path,
            )
    if not out_path.exists():
        raise RuntimeError("No checkpoint was written (no epochs were run).")
    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, loaders["test"], a_hat, camera_index, edge_index, edge_ids, graph.n_nodes, device, class_weights)
    print(f"best epoch {checkpoint['epoch']}, test={test}")
    return {"history": history, "test": test, "best_epoch": checkpoint["epoch"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--frames-root",
        type=Path,
        nargs="+",
        help="one or more frames folders; the first wins when a frame is in several",
    )
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("--camera-csv", type=Path, default=REPO_ROOT / "configs/camera_nodes.csv")
    p.add_argument(
        "--weather-csv", type=Path, default=REPO_ROOT / "data/processed/temporal/weatherstack_historical.csv"
    )
    p.add_argument("--raw-twitter", type=Path, default=REPO_ROOT / "data/raw/twitter")
    p.add_argument("--embeddings", type=Path, default=REPO_ROOT / "data/processed/embeddings.pt")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints/stgnn_edge_risk.pt")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr-fusion", type=float, default=0.0001)
    p.add_argument("--lr-graph", type=float, default=0.001)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-windows", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["official", "auto"], default="official")
    p.add_argument("--min-session-labels", type=int, default=100)
    p.add_argument("--human-only", action="store_true")
    p.add_argument(
        "--time-features", action="store_true",
        help="ablation only, not in the thesis: append time-in-session and a PM flag to the weather input",
    )
    p.add_argument("--no-time-features", action="store_true", help=argparse.SUPPRESS)  # old default; now a no-op
    p.add_argument(
        "--class-weights", action="store_true",
        help="balance the BCE loss against the Light/Medium/Heavy label skew (57%% Heavy by default), "
             "instead of letting the model learn to predict close to the mean label everywhere",
    )
    p.add_argument("--graph-sizes", action="store_true")
    a = p.parse_args()
    if a.graph_sizes:
        for k, nodes, edges in graph_sizes(a.spatial_dir, range(1, 11)):
            print(
                f"k={k:>2}: {nodes:>6} nodes, {edges:>6} edges, dense adjacency {nodes * nodes * 4 / 1000000.0:8.1f} MB"
            )
        return
    if not a.frames_root:
        p.error("--frames-root is required")
    train(
        a.frames_root,
        a.weather_csv,
        None if a.no_text else a.raw_twitter,
        None if a.no_text else a.embeddings,
        a.out,
        a.spatial_dir,
        a.camera_csv,
        a.k,
        a.epochs,
        a.batch_size,
        a.lr_fusion,
        a.lr_graph,
        image_size=a.image_size,
        num_workers=a.num_workers,
        max_train_windows=a.max_train_windows,
        device=a.device,
        seed=a.seed,
        human_only=a.human_only,
        split=a.split,
        min_session_labels=a.min_session_labels,
        time_features=a.time_features,
        use_class_weights=a.class_weights,
    )


if __name__ == "__main__":
    main()