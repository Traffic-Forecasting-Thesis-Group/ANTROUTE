"""
Train CNNLSTMFusion + RADR STGNN end to end on the visual congestion labels.

    python scripts/train_stgnn.py --graph-sizes                     # pick a k for the road subgraph
    python scripts/train_stgnn.py --frames-root "G:/My Drive/MMDA_FRAMES" \
        --init-fusion checkpoints/cnn_lstm_congestion.pt --epochs 10

Pipeline per 30-step window: fusion on each camera node -> scatter into the road subgraph
[B, T, N, 128] -> RADRSTGNN([B, N, 64]) -> Linear head -> per-node Light / Medium / Heavy.
Loss is cross-entropy on the camera nodes that have a label; the best checkpoint by
validation macro-F1 is saved to --out. The edge decoder / risk score is not trained here.

Cameras are matched to graph intersections from footage folder names plus configs/camera_nodes.csv
(camera_id,intersection); unmatched cameras are listed and skipped.
"""

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
from src.data.metrics import macro_f1, ordinal_mae  # noqa: E402
from src.data.training_data import (  # noqa: E402
    WEATHER_COLUMNS, assign_session_splits, build_training_records, describe_split, labelled_sessions,
    load_label_lookup,
)
from src.models.cnn_lstm_fusion import CNNLSTMFusion  # noqa: E402
from src.models.radr_stgnn import RADRSTGNN  # noqa: E402
from src.models.traffic_risk_model import IGNORE_INDEX, TrafficRiskModel, camera_node_loss  # noqa: E402

N_CLASSES = 3


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def class_weights_from_lookup(lookup: Dict[str, int]) -> torch.Tensor:
    counts = np.bincount(list(lookup.values()), minlength=N_CLASSES).astype(float)
    return torch.tensor(np.where(counts > 0, counts.sum() / (N_CLASSES * np.maximum(counts, 1)), 1.0),
                        dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, a_hat, camera_index, device, class_weights=None) -> Optional[dict]:
    if len(loader.dataset) == 0:
        return None
    model.eval()
    losses, true, pred = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch, a_hat, camera_index)
        target = batch["target"]
        mask = target != IGNORE_INDEX
        if not mask.any():
            continue
        losses.append(camera_node_loss(logits, target, camera_index, class_weights).item())
        true.append(target[mask].cpu().numpy())
        pred.append(logits[:, camera_index].argmax(-1)[mask].cpu().numpy())
    true, pred = np.concatenate(true), np.concatenate(pred)
    return {"loss": float(np.mean(losses)), "accuracy": float((true == pred).mean()),
            "macro_f1": macro_f1(true, pred), "ordinal_mae": ordinal_mae(true, pred),
            "n_nodes": int(len(true))}


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
    lr_fusion: float = 1e-4,
    lr_graph: float = 1e-3,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    patch_size: int = 16,
    patch_embed_dim: int = 768,
    init_fusion: Optional[Path] = None,
    resume: Optional[Path] = None,
    use_class_weights: bool = False,
    num_workers: int = 0,
    max_train_windows: Optional[int] = None,
    device: Optional[str] = None,
    seed: int = 0,
    graph: Optional[GraphData] = None,
    human_only: bool = False,
    split: str = "official",
    min_session_labels: int = 100,
    time_features: bool = True,
    node_embedding: bool = True,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    graph = graph or build_subgraph(spatial_dir, k)
    lookup = load_label_lookup(frames_root, human_only)
    visual_split = None
    if split == "auto":
        visual_split = assign_session_splits(labelled_sessions(frames_root, lookup, min_session_labels))
        print("70/15/15 split over the labelled sessions:\n" + describe_split(visual_split))
    records, scaler, skipped = build_training_records(
        frames_root, weather_csv, raw_twitter_root, embeddings_path, visual_split)
    if not lookup:
        raise ValueError("No labelled frames (with human_only, review frames in the viewer first).")

    camera_ids = sorted({r.camera_id for r in records})
    camera_map, unmapped = resolve_camera_map(camera_ids, list(graph.camera_nodes), camera_csv)
    print(f"graph: {graph.n_nodes} nodes, {graph.edge_index.shape[1]} edges | "
          f"{len(camera_map)}/{len(camera_ids)} cameras mapped to intersections")
    if unmapped:
        print("  unmapped cameras (add them to configs/camera_nodes.csv):", ", ".join(c[-12:] for c in unmapped))
    if not camera_map:
        raise ValueError("No camera could be mapped to a graph intersection; fill configs/camera_nodes.csv.")

    node_labels = sorted(set(camera_map.values()))
    camera_index = torch.tensor([graph.camera_nodes[label] for label in node_labels], dtype=torch.long, device=device)
    a_hat = graph.a_hat.to(device)

    datasets = {s: GraphWindowDataset(records, s, lookup, node_labels, camera_map, image_size,
                                     sample_cameras=(s == "train"), time_features=time_features)
                for s in ("train", "val", "test")}
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} windows over {len(node_labels)} intersections")
    if len(datasets["train"]) == 0:
        raise ValueError("No labelled training windows. Extract, auto-label, review and map cameras first.")
    if max_train_windows and len(datasets["train"]) > max_train_windows:
        keep = sorted(np.random.permutation(len(datasets["train"]))[:max_train_windows])
        datasets["train"].index = [datasets["train"].index[i] for i in keep]
    loaders = {s: DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"), num_workers=num_workers)
               for s, ds in datasets.items()}

    temporal_dim = datasets["train"].weather_dim
    fusion = CNNLSTMFusion(text_dim=768, temporal_dim=temporal_dim, image_size=image_size,
                           patch_size=patch_size, patch_embed_dim=patch_embed_dim)
    if init_fusion and Path(init_fusion).exists():
        stage1 = torch.load(init_fusion, map_location="cpu", weights_only=False)
        cfg = stage1["config"]
        if (cfg["image_size"], cfg["patch_size"], cfg["patch_embed_dim"]) != (image_size, patch_size, patch_embed_dim):
            raise ValueError(f"{init_fusion} was trained with {cfg}; image/patch settings must match.")
        own = fusion.state_dict()
        state = {k_[len("fusion."):]: v for k_, v in stage1["model"].items()
                 if k_.startswith("fusion.") and k_[len("fusion."):] in own and own[k_[len("fusion."):]].shape == v.shape}
        fusion.load_state_dict(state, strict=False)      # layers whose size changed (temporal input) start fresh
        print(f"fusion initialised from {init_fusion} ({len(state)} of {len(own)} tensors)")

    model = TrafficRiskModel(fusion, RADRSTGNN(in_features=fusion.lstm.hidden_size), N_CLASSES,
                             n_camera_nodes=len(node_labels) if node_embedding else 0).to(device)
    fusion_params = set(id(p) for p in model.fusion.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in fusion_params], "lr": lr_fusion},
        {"params": [p for p in model.parameters() if id(p) not in fusion_params], "lr": lr_graph},
    ], weight_decay=weight_decay)
    weights = class_weights_from_lookup(lookup).to(device) if use_class_weights else None
    amp = device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp)

    config = {"k": k, "image_size": image_size, "patch_size": patch_size, "patch_embed_dim": patch_embed_dim,
              "text_dim": 768, "temporal_dim": temporal_dim, "n_classes": N_CLASSES,
              "time_features": time_features, "node_embedding": node_embedding,
              "use_text": raw_twitter_root is not None,
              "visual_split": {f"{d.isoformat()}|{s}": v for (d, s), v in (visual_split or VISUAL_SPLIT).items()},
              "weather_columns": WEATHER_COLUMNS, "node_labels": node_labels, "camera_map": camera_map,
              "graph_node_ids": graph.node_ids}
    start_epoch, best = 0, float("-inf")
    if resume and Path(resume).exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = state["epoch"], state["best"]
        print(f"resumed from {resume} at epoch {start_epoch}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_path.with_suffix(".metrics.csv")
    history = []
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        running = []
        for batch in loaders["train"]:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(batch, a_hat, camera_index)
                loss = camera_node_loss(logits.float(), batch["target"], camera_index, weights)
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            running.append(loss.item())

        val = evaluate(model, loaders["val"], a_hat, camera_index, device, weights)
        row = {"epoch": epoch, "train_loss": float(np.mean(running)), "val": val}
        history.append(row)
        print(f"epoch {epoch}: train_loss={row['train_loss']:.4f} val={val}")
        with metrics_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if metrics_csv.stat().st_size == 0:
                writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_ordinal_mae"])
            writer.writerow([epoch, row["train_loss"]] + ([val["loss"], val["accuracy"], val["macro_f1"],
                                                            val["ordinal_mae"]] if val else ["", "", "", ""]))

        score = val["macro_f1"] if val else -row["train_loss"]
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "best": best, "config": config, "weather_min": scaler.min_,
                        "weather_max": scaler.max_}, out_path)

    if not out_path.exists():
        raise RuntimeError("No checkpoint was written (no epochs were run).")
    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, loaders["test"], a_hat, camera_index, device, weights)
    print(f"best epoch {checkpoint['epoch']}, test={test}")
    return {"history": history, "test": test, "best_epoch": checkpoint["epoch"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-root", type=Path, nargs="+",
                   help="one or more frames folders; the first wins when a frame is in several")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("--camera-csv", type=Path, default=REPO_ROOT / "configs/camera_nodes.csv")
    p.add_argument("--weather-csv", type=Path,
                   default=REPO_ROOT / "data/processed/temporal/weatherstack_historical.csv")
    p.add_argument("--raw-twitter", type=Path, default=REPO_ROOT / "data/raw/twitter")
    p.add_argument("--embeddings", type=Path, default=REPO_ROOT / "data/processed/embeddings.pt")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints/stgnn_congestion.pt")
    p.add_argument("--init-fusion", type=Path, default=REPO_ROOT / "checkpoints/cnn_lstm_congestion.pt")
    p.add_argument("--resume", type=Path)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr-fusion", type=float, default=1e-4)
    p.add_argument("--lr-graph", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-windows", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["official", "auto"], default="official",
                   help="official = the day split in alignment.py; auto = 70/15/15 by whole sessions over the labelled ones")
    p.add_argument("--min-session-labels", type=int, default=100,
                   help="with --split auto, a session needs this many labelled frames to be used")
    p.add_argument("--human-only", action="store_true", help="train only on frames reviewed by a person")
    p.add_argument("--no-time-features", action="store_true", help="do not give the model the time within the session")
    p.add_argument("--no-node-embedding", action="store_true", help="no learned per-intersection bias")
    p.add_argument("--graph-sizes", action="store_true", help="print subgraph sizes for k=1..10 and exit")
    a = p.parse_args()

    if a.graph_sizes:
        for k, nodes, edges in graph_sizes(a.spatial_dir, range(1, 11)):
            print(f"k={k:>2}: {nodes:>6} nodes, {edges:>6} edges, dense adjacency {nodes * nodes * 4 / 1e6:8.1f} MB")
        return
    if not a.frames_root:
        p.error("--frames-root is required")

    train(a.frames_root, a.weather_csv, None if a.no_text else a.raw_twitter,
          None if a.no_text else a.embeddings, a.out, a.spatial_dir, a.camera_csv, a.k, a.epochs,
          a.batch_size, a.lr_fusion, a.lr_graph, image_size=a.image_size, init_fusion=a.init_fusion,
          resume=a.resume, use_class_weights=a.class_weights, num_workers=a.num_workers,
          max_train_windows=a.max_train_windows, device=a.device, seed=a.seed, human_only=a.human_only, split=a.split, min_session_labels=a.min_session_labels,
          time_features=not a.no_time_features, node_embedding=not a.no_node_embedding)


if __name__ == "__main__":
    main()
