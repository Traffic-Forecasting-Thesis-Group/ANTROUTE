"""
Train the CNN + LSTM module to predict per-minute congestion (Light / Medium / Heavy).

Inputs: frames + labels from extract_frames.py / autolabel_frames.py / label_viewer.py,
DistilBERT text embeddings, and daily weather (all aligned by alignment.build_sessions).
The 2D patch embedding is the first layer of CNNLSTMFusion and is trained with it.

    python scripts/train_cnn_lstm.py --frames-root "G:/My Drive/MMDA_FRAMES" --epochs 5
    python scripts/train_cnn_lstm.py --frames-root ... --no-text        # visual + weather only

Splits follow alignment.VISUAL_SPLIT (train: May 4-18, val: May 20 + 22 AM, test: rest).
Saves the best-validation checkpoint (macro-F1) to --out.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.training_data import (  # noqa: E402
    IGNORE_INDEX, WEATHER_COLUMNS, LabeledWindowDataset, assign_session_splits, build_training_records,
    describe_split, labelled_sessions, load_label_lookup,
)
from src.data.metrics import macro_f1  # noqa: E402
from src.models.cnn_lstm_fusion import CNNLSTMFusion  # noqa: E402

N_CLASSES = 3


class CongestionModel(nn.Module):
    """CNNLSTMFusion followed by a per-timestep congestion classifier."""

    def __init__(self, fusion: CNNLSTMFusion, hidden_dim: int, n_classes: int = N_CLASSES):
        super().__init__()
        self.fusion = fusion
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.fusion(
            batch["images"], batch["text"], batch["temporal"],
            visual_mask=batch["visual_mask"], text_mask=batch["text_mask"],
        )
        return self.head(features)  # [B, T, n_classes]


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: CongestionModel, loader: DataLoader, device) -> Optional[dict]:
    if len(loader.dataset) == 0:
        return None
    model.eval()
    losses, true, pred = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch)
        target = batch["target"]
        losses.append(F.cross_entropy(logits.reshape(-1, N_CLASSES), target.reshape(-1),
                                      ignore_index=IGNORE_INDEX).item())
        mask = target != IGNORE_INDEX
        true.append(target[mask].cpu().numpy())
        pred.append(logits.argmax(-1)[mask].cpu().numpy())
    true, pred = np.concatenate(true), np.concatenate(pred)
    return {"loss": float(np.mean(losses)), "accuracy": float((true == pred).mean()),
            "macro_f1": macro_f1(true, pred), "n_steps": int(len(true))}


def train(
    frames_root,
    weather_csv: Path,
    raw_twitter_root: Optional[Path],
    embeddings_path: Optional[Path],
    out_path: Path,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    image_size: int = 224,
    patch_size: int = 16,
    patch_embed_dim: int = 768,
    num_workers: int = 0,
    max_train_windows: Optional[int] = None,
    device: Optional[str] = None,
    seed: int = 0,
    human_only: bool = False,
    split: str = "official",
    min_session_labels: int = 100,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    lookup = load_label_lookup(frames_root, human_only)
    visual_split = None
    if split == "auto":
        visual_split = assign_session_splits(labelled_sessions(frames_root, lookup, min_session_labels))
        print("70/15/15 split over the labelled sessions:\n" + describe_split(visual_split))
    records, scaler, skipped = build_training_records(
        frames_root, weather_csv, raw_twitter_root, embeddings_path, visual_split)
    if not lookup:
        raise ValueError("No labelled frames (with human_only, review frames in the viewer first).")
    print(f"{len(records)} sessions built ({skipped} skipped for missing weather), "
          f"{len(lookup)} labelled frames")

    datasets = {s: LabeledWindowDataset(records, s, lookup, image_size) for s in ("train", "val", "test")}
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} windows")
    if len(datasets["train"]) == 0:
        raise ValueError("No labelled training windows. Extract, auto-label and review frames first.")

    if max_train_windows and len(datasets["train"]) > max_train_windows:
        keep = np.random.permutation(len(datasets["train"]))[:max_train_windows]
        datasets["train"].index = [datasets["train"].index[i] for i in sorted(keep)]

    loaders = {
        s: DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"), num_workers=num_workers)
        for s, ds in datasets.items()
    }

    fusion = CNNLSTMFusion(
        text_dim=768, temporal_dim=len(WEATHER_COLUMNS), image_size=image_size,
        patch_size=patch_size, patch_embed_dim=patch_embed_dim,
    )
    model = CongestionModel(fusion, fusion.lstm.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    config = {"image_size": image_size, "patch_size": patch_size, "patch_embed_dim": patch_embed_dim,
              "text_dim": 768, "temporal_dim": len(WEATHER_COLUMNS), "n_classes": N_CLASSES,
              "weather_columns": WEATHER_COLUMNS}
    history, best = [], float("-inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        for batch in loaders["train"]:
            batch = to_device(batch, device)
            logits = model(batch)
            loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), batch["target"].reshape(-1),
                                   ignore_index=IGNORE_INDEX)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running.append(loss.item())

        val = evaluate(model, loaders["val"], device)
        record = {"epoch": epoch, "train_loss": float(np.mean(running)), "val": val}
        history.append(record)
        score = val["macro_f1"] if val else -record["train_loss"]
        print(f"epoch {epoch}: train_loss={record['train_loss']:.4f} val={val}")

        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "config": config,
                        "weather_min": scaler.min_, "weather_max": scaler.max_, "epoch": epoch}, out_path)

    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, loaders["test"], device)
    print(f"best epoch {checkpoint['epoch']}, test={test}")
    return {"history": history, "test": test, "best_epoch": checkpoint["epoch"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-root", required=True, type=Path, nargs="+",
                   help="one or more frames folders; the first wins when a frame is in several")
    p.add_argument("--weather-csv", type=Path,
                   default=REPO_ROOT / "data/processed/temporal/weatherstack_historical.csv")
    p.add_argument("--raw-twitter", type=Path, default=REPO_ROOT / "data/raw/twitter")
    p.add_argument("--embeddings", type=Path, default=REPO_ROOT / "data/processed/embeddings.pt")
    p.add_argument("--no-text", action="store_true", help="skip tweets (text is masked everywhere)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints/cnn_lstm_congestion.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-windows", type=int, default=None, help="subsample for a quick run")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["official", "auto"], default="official",
                   help="official = the day split in alignment.py; auto = 70/15/15 by whole sessions over the labelled ones")
    p.add_argument("--min-session-labels", type=int, default=100,
                   help="with --split auto, a session needs this many labelled frames to be used")
    p.add_argument("--human-only", action="store_true", help="train only on frames reviewed by a person")
    a = p.parse_args()

    train(a.frames_root, a.weather_csv,
          None if a.no_text else a.raw_twitter, None if a.no_text else a.embeddings,
          a.out, a.epochs, a.batch_size, a.lr, a.image_size, num_workers=a.num_workers,
          max_train_windows=a.max_train_windows, device=a.device, seed=a.seed, human_only=a.human_only, split=a.split, min_session_labels=a.min_session_labels)


if __name__ == "__main__":
    main()
