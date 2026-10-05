"""
Run a trained CNNLSTMFusion + RADR STGNN checkpoint over EVERY extracted session and write congestion
predictions and risk scores.

    python scripts/predict_risk.py --checkpoint checkpoints/stgnn_final.pt \
        --frames-root /content/frames/pilot /content/frames/a /content/frames/b --out-dir outputs/risk

Outputs (in --out-dir):
    risk_nodes.csv    one row per 30-minute window and camera intersection:
                      probabilities for Light / Medium / Heavy, predicted class, risk, camera_present,
                      split (train / val / test / infer) and, where you labelled it, the human label
    risk_edges.csv    (with --edges) one row per window and road edge: risk of the edge
    risk_summary.json accuracy / macro-F1 per split against your human labels, and the majority baseline

Risk score:
    node risk = P(Medium) * 0.5 + P(Heavy) * 1.0            (0 = free flowing, 1 = heavy)
    edge risk = mean of the risk of its two end nodes
Only the camera intersections were trained against labels. Other nodes of the road subgraph get a risk from
the graph model's propagation alone, so treat edge risks away from a camera as an estimate.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.alignment import WINDOW_STEPS, session_start  # noqa: E402
from src.data.graph_data import build_subgraph  # noqa: E402
from src.data.graph_dataset import GraphWindowDataset  # noqa: E402
from src.data.metrics import macro_f1  # noqa: E402
from src.data.training_data import (  # noqa: E402
    IGNORE_INDEX, InferenceWindowDataset, build_training_records, load_frames_table, load_label_lookup, session_of,
)
from src.models.cnn_lstm_fusion import CNNLSTMFusion  # noqa: E402
from src.models.radr_stgnn import RADRSTGNN  # noqa: E402
from src.models.traffic_risk_model import TrafficRiskModel  # noqa: E402

CLASS_NAMES = ["Light", "Medium", "Heavy"]
RISK_WEIGHTS = torch.tensor([0.0, 0.5, 1.0])
NODE_FIELDS = ["day", "session", "window_start", "window_end", "split", "intersection", "node_id", "camera_present",
               "p_light", "p_medium", "p_heavy", "predicted", "risk", "human_label"]
EDGE_FIELDS = ["day", "session", "window_end", "source_node_id", "target_node_id", "risk"]


def parse_session_key(key: str):
    from datetime import date
    day, session = key.split("|")
    return date.fromisoformat(day), session


def build_model(cfg: dict, device) -> TrafficRiskModel:
    fusion = CNNLSTMFusion(text_dim=cfg["text_dim"], temporal_dim=cfg["temporal_dim"], image_size=cfg["image_size"],
                           patch_size=cfg["patch_size"], patch_embed_dim=cfg["patch_embed_dim"])
    n_nodes = len(cfg["node_labels"]) if cfg.get("node_embedding") else 0
    return TrafficRiskModel(fusion, RADRSTGNN(in_features=fusion.lstm.hidden_size), cfg["n_classes"],
                            n_camera_nodes=n_nodes).to(device)


def node_risk(probs: torch.Tensor) -> torch.Tensor:
    """[..., 3] class probabilities -> risk in [0, 1]."""
    return (probs * RISK_WEIGHTS.to(probs.device)).sum(-1)


def summarise(rows: List[dict]) -> Dict[str, dict]:
    """Accuracy / macro-F1 against human labels, per split, plus the always-majority baseline."""
    out = {}
    for split in sorted({r["split"] for r in rows}):
        labelled = [r for r in rows if r["split"] == split and r["human_label"] != ""]
        entry = {"windows": sum(r["split"] == split for r in rows), "labelled_windows": len(labelled)}
        if labelled:
            y = np.array([CLASS_NAMES.index(r["human_label"]) for r in labelled])
            p = np.array([CLASS_NAMES.index(r["predicted"]) for r in labelled])
            majority = np.full_like(y, np.bincount(y, minlength=3).argmax())
            entry.update({"accuracy": float((y == p).mean()), "macro_f1": macro_f1(y, p),
                          "majority_baseline_accuracy": float((y == majority).mean()),
                          "majority_baseline_macro_f1": macro_f1(y, majority)})
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
    write_edges: bool = False,
    with_labels: bool = True,
    graph=None,
) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    node_labels, camera_map = cfg["node_labels"], cfg["camera_map"]

    graph = graph or build_subgraph(spatial_dir, cfg["k"])
    if not np.array_equal(graph.node_ids, cfg["graph_node_ids"]):
        raise ValueError("The road subgraph differs from the one this checkpoint was trained on "
                         "(different spatial files or k); scores would be misaligned.")
    camera_index = torch.tensor([graph.camera_nodes[label] for label in node_labels], dtype=torch.long)

    # Reuse the training split so weather is normalised exactly as in training; unseen sessions become 'infer'.
    saved = {parse_session_key(k): v for k, v in cfg["visual_split"].items()}
    frames = load_frames_table(frames_roots)
    sessions = sorted({session_of(t) for t in frames["timestamp"]})
    visual_split = {s: saved.get(s, "infer") for s in sessions}
    use_text = bool(cfg.get("use_text")) and raw_twitter_root is not None and embeddings_path is not None
    records, _, skipped = build_training_records(frames_roots, weather_csv, raw_twitter_root if use_text else None,
                                                 embeddings_path if use_text else None, visual_split=visual_split)
    lookup = load_label_lookup(frames_roots, human_only=True) if with_labels else None
    base = InferenceWindowDataset(records, cfg["image_size"], cfg.get("time_features", False), lookup or None)
    dataset = GraphWindowDataset(records, "infer", {}, node_labels, camera_map, cfg["image_size"],
                                 time_features=cfg.get("time_features", False), base=base)
    print(f"{len(sessions)} sessions ({skipped} skipped for missing weather), {len(dataset)} windows, "
          f"{len(node_labels)} intersections, text {'on' if use_text else 'off'}")

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    a_hat = graph.a_hat.to(device)
    camera_index_d = camera_index.to(device)
    edge_src, edge_dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    node_ids = graph.node_ids

    out_dir.mkdir(parents=True, exist_ok=True)
    node_rows: List[dict] = []
    edge_file = (out_dir / "risk_edges.csv").open("w", encoding="utf-8", newline="") if write_edges else None
    edge_writer = csv.writer(edge_file) if edge_file else None
    if edge_writer:
        edge_writer.writerow(EDGE_FIELDS)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(batch, a_hat, camera_index_d)
            probs = torch.softmax(logits.float(), dim=-1).cpu()                 # [B, N, 3]
            risk_all = node_risk(probs).numpy()                                   # [B, N]
            targets = batch["target"].cpu()
            present = batch["visual_mask"].any(dim=-1).cpu()                      # [B, K]
            for b in range(probs.shape[0]):
                day, session, start = dataset.keys[seen + b]
                begin = session_start(day, session) + pd.Timedelta(minutes=int(start))
                end = begin + pd.Timedelta(minutes=WINDOW_STEPS)
                for slot, label in enumerate(node_labels):
                    p = probs[b, camera_index[slot]]
                    human = int(targets[b, slot])
                    node_rows.append({
                        "day": day.isoformat(), "session": session, "window_start": begin.isoformat(),
                        "window_end": end.isoformat(), "split": visual_split.get((day, session), "infer"),
                        "intersection": label, "node_id": int(node_ids[camera_index[slot]]),
                        "camera_present": bool(present[b, slot]), "p_light": round(float(p[0]), 4),
                        "p_medium": round(float(p[1]), 4), "p_heavy": round(float(p[2]), 4),
                        "predicted": CLASS_NAMES[int(p.argmax())], "risk": round(float(risk_all[b, camera_index[slot]]), 4),
                        "human_label": CLASS_NAMES[human] if human != IGNORE_INDEX else "",
                    })
                if edge_writer:
                    edge_risk = 0.5 * (risk_all[b, edge_src] + risk_all[b, edge_dst])
                    edge_writer.writerows(zip([day.isoformat()] * len(edge_risk), [session] * len(edge_risk),
                                              [end.isoformat()] * len(edge_risk), node_ids[edge_src], node_ids[edge_dst],
                                              np.round(edge_risk, 4)))
            seen += probs.shape[0]
    if edge_file:
        edge_file.close()

    with (out_dir / "risk_nodes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NODE_FIELDS)
        writer.writeheader()
        writer.writerows(node_rows)
    summary = {"checkpoint": str(checkpoint), "windows": len(dataset), "sessions": len(sessions),
               "splits": summarise(node_rows)}
    (out_dir / "risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--frames-root", required=True, type=Path, nargs="+")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/risk_scores")
    p.add_argument("--weather-csv", type=Path,
                   default=REPO_ROOT / "data/processed/temporal/weatherstack_historical.csv")
    p.add_argument("--raw-twitter", type=Path, default=REPO_ROOT / "data/raw/twitter")
    p.add_argument("--embeddings", type=Path, default=REPO_ROOT / "data/processed/embeddings.pt")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("--edges", action="store_true", help="also write risk_edges.csv (large)")
    p.add_argument("--no-labels", action="store_true", help="do not attach your human labels")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    summary = predict(a.checkpoint, a.frames_root, a.weather_csv, a.out_dir, a.raw_twitter, a.embeddings,
                      a.spatial_dir, a.batch_size, a.num_workers, a.device, a.edges, not a.no_labels)
    print(json.dumps(summary["splits"], indent=2))
    print(f"wrote {a.out_dir / 'risk_nodes.csv'}" + (f" and {a.out_dir / 'risk_edges.csv'}" if a.edges else ""))


if __name__ == "__main__":
    main()
