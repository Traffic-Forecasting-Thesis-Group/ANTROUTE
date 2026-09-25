"""
ANTROUTE — Real (Date-Synchronized) CNN+LSTM Feature Caching Pipeline

Runs every real visual embedding file through CNNLSTMFusion, joined to real
data from the other two modalities by DATE (not a random smoke test):

  - Visual:   pre-extracted 2D patch embeddings on Google Drive
              (.pt, [36,196,256]), date + camera_id parsed from the file path.
  - Weather:  data/processed/temporal/weatherstack_historical.csv, joined by
              exact date (averaged across that date's grid_cell rows, since
              no per-camera geocoding exists).
  - Text:     data/processed/embeddings.pt (pre-computed DistilBERT vectors),
              joined by date via a replayed walk of data/raw/twitter (same
              order TwitterTrafficDataset used to build that file, so no
              model re-run is needed). Tweets are sparse (~168 total across
              a year) — dates with no tweets use CNNLSTMFusion's own
              missing_text placeholder + text_mask=False, exactly as that
              mechanism was designed for.

Output: traffic_system/data/processed/cnn_lstm_features.pt containing the
fused LSTM output, intermediate visual features, temporal features, and a
manifest row per sample (camera_id, date, source path, whether real text
was available that day).

Text is NOT camera-specific (tweets aren't geocoded to a camera/intersection
in this dataset) — it's the real same-day tweet pool, shared across every
camera active that day. See the project plan for the reasoning.
"""

import json
import pickle
import re
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]  # traffic_system/
import sys
sys.path.insert(0, str(REPO_ROOT))

from src.models.cnn_lstm_fusion import CNNLSTMFusion  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_ROOT = Path(r"G:\My Drive\MMDA_EMBEDDINGS\REQ. PUP STUDENT")
RAW_TWITTER_ROOT = REPO_ROOT / "data" / "raw" / "twitter"
TEXT_EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "embeddings.pt"
WEATHER_CSV_PATH = REPO_ROOT / "data" / "processed" / "temporal" / "weatherstack_historical.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "cnn_lstm_features.pt"

EXPECTED_VISUAL_SHAPE = (36, 196, 256)  # (T, num_patches, embed_dim)
T = EXPECTED_VISUAL_SHAPE[0]
PATCH_EMBED_DIM = 256
TEXT_DIM = 768
TEMPORAL_DIM = 3  # ws_temp_c, ws_precip_mm, ws_humidity_pct

DATE_IN_PATH_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\.\d{2}\.\d{2}\.\d{2}")
PROGRESS_EVERY = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Stage A — tweet date index (replays TwitterTrafficDataset's exact walk,
# so row i here lines up with row i of embeddings.pt without re-running the
# NLP model)
# ---------------------------------------------------------------------------
def build_tweet_date_index(raw_twitter_root: Path):
    dates = []
    paths = sorted(raw_twitter_root.rglob("tweets_*.json"))
    for path in paths:
        with path.open(encoding="utf-8") as f:
            raw_content = json.load(f)
        tweets = raw_content.get("data", raw_content) if isinstance(raw_content, dict) else raw_content
        for tweet in tweets:
            created_at = tweet.get("createdAt") if isinstance(tweet, dict) else None
            if created_at is None:
                dates.append(None)
                continue
            try:
                dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                dates.append(None)
                continue
            dates.append(dt.date())

    date_to_rows = defaultdict(list)
    for idx, d in enumerate(dates):
        if d is not None:
            date_to_rows[d].append(idx)
    return date_to_rows


# ---------------------------------------------------------------------------
# Stage A (cont.) — load the pre-computed text embedding pool
# ---------------------------------------------------------------------------
def load_text_pool(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    embeddings = payload["embeddings"] if isinstance(payload, dict) else payload
    return embeddings.float()


# ---------------------------------------------------------------------------
# Stage B — weather: date -> averaged [3] vector across that date's grid cells
# ---------------------------------------------------------------------------
def build_weather_by_date(csv_path: Path):
    import csv

    sums = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    counts = defaultdict(int)

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d = datetime.strptime(row["date"], "%Y-%m-%d").date()
                vec = np.array(
                    [float(row["ws_temp_c"]), float(row["ws_precip_mm"]), float(row["ws_humidity_pct"])],
                    dtype=np.float64,
                )
            except (KeyError, ValueError):
                continue
            sums[d] += vec
            counts[d] += 1

    return {d: (sums[d] / counts[d]).astype(np.float32) for d in sums}


# ---------------------------------------------------------------------------
# Stage C — visual manifest: scan Drive tree, parse date + camera_id from path
# ---------------------------------------------------------------------------
def parse_date_and_camera_id(pt_path: Path):
    match = DATE_IN_PATH_RE.search(str(pt_path))
    if not match:
        return None, None
    year, month, day = match.groups()
    try:
        parsed_date = datetime(int(year), int(month), int(day)).date()
    except ValueError:
        return None, None

    # .../<camera_id>/Dados/<file>.pt
    camera_id = pt_path.parent.parent.name if pt_path.parent.name == "Dados" else pt_path.parent.name
    return parsed_date, camera_id


def load_visual_tensor(path: Path):
    try:
        tensor = torch.load(path, map_location="cpu")
    except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as e:
        warnings.warn(f"Skipping unreadable/locked/corrupt file {path.name}: {e}")
        return None
    if not torch.is_tensor(tensor) or tuple(tensor.shape) != EXPECTED_VISUAL_SHAPE:
        warnings.warn(
            f"Skipping {path.name}: unexpected shape "
            f"{tuple(getattr(tensor, 'shape', ()))} != {EXPECTED_VISUAL_SHAPE}"
        )
        return None
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    return tensor


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("Building tweet date index (replaying TwitterTrafficDataset's walk)...")
    date_to_tweet_rows = build_tweet_date_index(RAW_TWITTER_ROOT)
    print(f"  {sum(len(v) for v in date_to_tweet_rows.values())} tweets across {len(date_to_tweet_rows)} dates")

    print("Loading pre-computed text embedding pool...")
    text_pool = load_text_pool(TEXT_EMBEDDINGS_PATH)
    print(f"  text_pool shape={tuple(text_pool.shape)}")

    print("Building weather-by-date lookup...")
    weather_by_date = build_weather_by_date(WEATHER_CSV_PATH)
    print(f"  {len(weather_by_date)} dates with weather data")

    print(f"Scanning visual embeddings under {EMBED_ROOT} ...")
    pt_paths = sorted(EMBED_ROOT.rglob("*.pt"))
    print(f"  found {len(pt_paths)} candidate files")

    fusion = CNNLSTMFusion(
        text_dim=TEXT_DIM,
        temporal_dim=TEMPORAL_DIM,
        visual_feature_dim=128,
        patch_embed_dim=PATCH_EMBED_DIM,
    ).to(DEVICE)
    fusion.eval()

    grid_size = int(EXPECTED_VISUAL_SHAPE[1] ** 0.5)
    assert grid_size * grid_size == EXPECTED_VISUAL_SHAPE[1]

    fused_outputs = []
    visual_features_all = []
    temporal_features_all = []
    had_real_text_all = []
    manifest = []

    n_skipped_corrupt = 0
    n_skipped_no_weather = 0
    n_with_real_text = 0
    n_processed = 0

    for i, pt_path in enumerate(pt_paths):
        if (i + 1) % PROGRESS_EVERY == 0:
            print(f"  ... {i + 1}/{len(pt_paths)} scanned, {n_processed} cached so far")

        visual_date, camera_id = parse_date_and_camera_id(pt_path)
        if visual_date is None:
            n_skipped_corrupt += 1
            continue

        visual_tensor = load_visual_tensor(pt_path)
        if visual_tensor is None:
            n_skipped_corrupt += 1
            continue

        weather_vec = weather_by_date.get(visual_date)
        if weather_vec is None:
            n_skipped_no_weather += 1
            continue

        with torch.no_grad():
            # [T,P,D] -> [T,D,grid,grid] -> CNN -> [T,128]
            grid = visual_tensor.transpose(1, 2).reshape(T, PATCH_EMBED_DIM, grid_size, grid_size)
            visual_features = fusion.visual_encoder.cnn(grid.to(DEVICE)).flatten(start_dim=1)  # [T,128]

            temporal_features = torch.from_numpy(np.tile(weather_vec, (T, 1))).float().to(DEVICE)  # [T,3]

            tweet_rows = date_to_tweet_rows.get(visual_date, [])
            if tweet_rows:
                # Sample with replacement (most dates have far fewer than T
                # real tweets) to fill all T timesteps with real same-day text.
                idx = torch.randint(low=0, high=len(tweet_rows), size=(T,))
                row_ids = torch.tensor([tweet_rows[j] for j in idx.tolist()])
                text_features = text_pool[row_ids].to(DEVICE)  # [T,768]
                text_mask = torch.ones(T, dtype=torch.bool, device=DEVICE)
                had_real_text = True
                n_with_real_text += 1
            else:
                text_features = torch.zeros(T, TEXT_DIM, device=DEVICE)
                text_mask = torch.zeros(T, dtype=torch.bool, device=DEVICE)
                had_real_text = False

            # Manual fusion (bypasses CNNLSTMFusion.forward(), which only
            # accepts raw images — visual data here is already patch-embedded)
            visual = fusion.visual_dropout(visual_features.unsqueeze(0))  # [1,T,128]
            text_proj = fusion.text_projection(text_features.unsqueeze(0))  # [1,T,text_proj_dim]
            text_proj = torch.where(
                text_mask.unsqueeze(0).unsqueeze(-1),
                text_proj,
                fusion.missing_text.to(text_proj.dtype).expand_as(text_proj),
            )
            temporal_proj = fusion.temporal_projection(temporal_features.unsqueeze(0))  # [1,T,temp_proj_dim]

            fused = fusion.fusion_norm(torch.cat((visual, text_proj, temporal_proj), dim=-1))
            lstm_out, _ = fusion.lstm(fused)
            lstm_out = fusion.output_dropout(lstm_out).squeeze(0)  # [T,128]

        fused_outputs.append(lstm_out.cpu())
        visual_features_all.append(visual_features.cpu())
        temporal_features_all.append(temporal_features.cpu())
        had_real_text_all.append(had_real_text)
        manifest.append(
            {
                "camera_id": camera_id,
                "date": visual_date.isoformat(),
                "source_path": str(pt_path),
                "had_real_text": had_real_text,
            }
        )
        n_processed += 1

    if n_processed == 0:
        raise RuntimeError("No samples were successfully processed — check EMBED_ROOT and data paths.")

    fused_tensor = torch.stack(fused_outputs, dim=0)          # [N,T,128]
    visual_tensor_all = torch.stack(visual_features_all, dim=0)  # [N,T,128]
    temporal_tensor_all = torch.stack(temporal_features_all, dim=0)  # [N,T,3]
    had_real_text_tensor = torch.tensor(had_real_text_all, dtype=torch.bool)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fused_output": fused_tensor,
            "visual_features": visual_tensor_all,
            "temporal_features": temporal_tensor_all,
            "had_real_text": had_real_text_tensor,
            "manifest": manifest,
        },
        OUTPUT_PATH,
    )

    print("\n=== Summary ===")
    print(f"Scanned:              {len(pt_paths)}")
    print(f"Cached samples:       {n_processed}")
    print(f"Skipped (corrupt):    {n_skipped_corrupt}")
    print(f"Skipped (no weather): {n_skipped_no_weather}")
    print(f"With real same-day text: {n_with_real_text} ({100 * n_with_real_text / n_processed:.1f}%)")
    print(f"fused_output shape: {tuple(fused_tensor.shape)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
