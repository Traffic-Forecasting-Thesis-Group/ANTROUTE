# ANTROUTE — Session Handoff (as of 2026-09-30)

Read this first in a new session before doing anything else. It says where the project actually
stands against the system architecture, what's already working, what's broken, and what's next.

Branch: `dev`. Latest commit at time of writing: `40b1074d`.

## Architecture status, stage by stage

| Diagram stage | Status |
|---|---|
| Unstructured Data (X/Twitter) → ... → DistilBERT | Done. `data/processed/embeddings.pt` regenerated fresh from all 46,098 real tweets (was stale before, fixed). Currently **not used in training** (`--no-text`) — the fresh embeddings file was never copied to Drive for Colab to read, and multi-modal text sync is unsynchronized-by-design (see "Known limitations" below). |
| Visual Data (MMDA CCTV) → Frame Extraction & Normalization | Done. Old approach (random per-run patch embeddings) was replaced: frames are now extracted as plain JPEGs (`scripts/extract_frames.py`), 1/min, across all 10 session dates (May 4–25, AM+PM). Real bugs found and fixed along the way: mid-stream `.dar` chunks with no PPS, 25-vs-30fps misread cameras, duplicate video copies, truncated MP4Box conversions. Extraction is essentially complete (a handful of chunks — mostly the Ayala `8337` camera — still fail on every method tried; low priority). |
| 2D Patch Embedding | Now the **first learned layer inside `CNNLSTMFusion`** (`src/vision/patch_embedder.py`), not a separate saved step. This matches the diagram's real intent — trained end-to-end, not a random fixed projection like the old notebook did. |
| CNN + LSTM (Visual and Event Feature Extraction) | Implemented, `src/models/cnn_lstm_fusion.py`. Had a real GPU-only bug (autocast dtype mismatch on boolean-mask assignment) — **fixed**, commit `b3f26904`. |
| Temporal Data (WeatherStack) → Sequence Norm | Done. `data/processed/temporal/weatherstack_historical.csv`, joined by date (city-wide average across grid cells — no per-camera geocoding exists). |
| Spatial Data (OSMnx, Project NOAH) → Geocoding & Topology | Done earlier. `data/processed/spatial/` has the full 59,521-node Metro Manila road graph + adjacency. RADR STGNN trains on a **k=8 hop subgraph around the 8 camera intersections** (1,409 nodes, 2,816 edges) — the full graph is too large for a dense adjacency, and only 8 nodes have real camera coverage. |
| RADR STGNN (GCN + GRU) | Implemented, `src/models/radr_stgnn.py`. Same autocast dtype bug existed one layer downstream (`scatter_camera_features`, both model variants) — **fixed**, commit `18a5ab80`. |
| MLP Decoder → Congestion Risk Score (per edge) | **This is the important recent change.** Two parallel implementations exist: <br>• `traffic_risk_model.py` / `train_stgnn.py` / `predict_risk.py` — older, per-node Light/Medium/Heavy classifier + a hand-written averaging formula for edge risk. Does **not** match the diagram (no MLP Decoder in the loop). <br>• `traffic_risk_model_edge.py` / `train_stgnn_edge.py` / `predict_congestion_risk.py` — merged from `feature/mlp-decoder-joint-edge-risk` (PR #6). Uses the real, previously-unused `MLPDecoder` to output risk **directly per edge**, matching the diagram exactly (RADR STGNN → MLP Decoder → risk per edge). **This is the one to use going forward.** The old one should probably be retired/kept only as an ablation comparison — team hasn't formally decided, but everyone was leaning this way. |
| Dynamic Weight Engine, Dynamic Routing Engine, Optimal Path | **Not started.** Blocked on having a trained, trusted risk-score model first. |

## Real training results so far

Two completed real training runs exist on Drive (`G:\My Drive\MMDA_CHECKPOINTS\` /
`/content/drive/MyDrive/MMDA_CHECKPOINTS/`), both using `train_stgnn_edge.py`
(`--human-only --split auto --no-text`, k=8 subgraph, 11 train / 2 val / 2 test sessions,
labels from real human annotation, not the raw YOLO auto-labels):

| Run | Epochs | Saved epoch | val_loss | val_mae | Note |
|---|---|---|---|---|---|
| `stgnn_edge_v1.pt` | 15 | 3 | 0.5088 | 0.260 | val_loss-best and val_mae-best epochs **disagree** (val_mae was actually lowest at epoch 15, 0.215, but that epoch wasn't saved — checkpoint selection uses val_loss). Val loss gets noisy/worse after epoch 3-ish — overfitting signature. |
| `stgnn_edge_v2.pt` | 15 | **4** | 0.5047 | **0.223** | val_loss-best and val_mae-best **agree** on epoch 4 this time — cleaner signal, genuinely the better of the two runs. |

**Baseline for comparison:** a naive "always predict 0.5" (Medium) guess gets MAE ≈ 0.33 for a
roughly balanced Light/Medium/Heavy label mix. Both real runs clearly beat that (v2: ~32% relative
improvement), so the model is learning something real, not noise.

**v2 is the best result in hand right now.** If picking one number to report today, use v2's
epoch 4 (val_mae 0.223). Nobody has looked closely at this result yet — it got buried under a long
stretch of Colab/notebook infrastructure debugging.

## What actually broke today, and what's fixed

Long story short: **the modeling code itself is in good shape now.** Almost everything debugged
today was Colab/Drive infrastructure, not the model. In order encountered:

1. **CNNLSTMFusion autocast dtype crash** (`Index put requires the source and destination dtypes
   match, got Float for the destination and Half for the source`) — fixed, `b3f26904`.
2. **Same bug one layer downstream**, in `scatter_camera_features` (both model variants) and the
   node-embedding addition line — fixed proactively, `18a5ab80`.
3. **Google Drive rate-limiting**: copying ~35,000 frame files from Drive to Colab's local disk in
   one session started silently hanging (no error, just stuck) after heavy cumulative I/O. Fixed
   with a 30-min `rsync` timeout (fails loud instead of hanging) and an `ONLY_DATES` filter so a
   folder only pulls the date subfolders actually needed by the current `--split auto` run, cutting
   volume. **This is still not fully solved** — it's recurred on different folders across different
   attempts today, consistent with a genuine Google-side rate-limit cooldown window, not a code bug.
   If it happens again: stop restart-looping, wait 20-30+ min before retrying.
4. **Stale Colab notebook tabs**: opening a notebook from GitHub creates a disconnected copy inside
   Colab. `git pull` inside a Colab cell only updates the `.py` scripts the notebook calls — it does
   **not** update the notebook's own cells. Several rounds of confusion today came from the user
   editing/running an old tab that never picked up notebook-level fixes (the Config-cell
   restructuring, the extended date filters, etc.). **Fix: re-open fresh from GitHub
   (File → Open notebook → GitHub) rather than trusting an already-open tab, whenever a notebook-level
   fix has been pushed.**
5. **Missing-frame crash**: `manifest.csv` lists every extracted frame, but an interrupted Drive
   copy can leave some of those files missing locally, which crashed the whole `DataLoader` mid-epoch
   on one `FileNotFoundError`. Fixed to skip that one frame (mask it as absent, same as a genuine
   gap) instead of crashing the whole run — `40b1074d`. This makes future runs much more resilient
   to Drive flakiness.
6. Smaller notebook bugs fixed along the way: `pandas` not imported in results cells, `RUN`/
   `CKPT_DIR`/`OUT` not defined after a session restart (now centralized in one Config cell near the
   top), results cell reporting a different epoch than the one actually saved.

## Data / labeling status

15 qualifying sessions (≥100 human labels each) as of this writing, 12,178 total merged human
labels across all labelers (`labels_<name>.csv` files, one per person, merged automatically —
this was built specifically so multiple people can label the same folder concurrently without
overwriting each other).

```
train (11): May 04 PM, May 06 AM, May 06 PM, May 08 AM, May 08 PM, May 11 AM, May 11 PM,
            May 13 AM, May 13 PM, May 18 AM, May 18 PM
val   (2):  May 20 AM, May 20 PM
test  (2):  May 25 AM, May 25 PM
```

Still unlabeled: **May 22** (both sessions, fully extracted, just untouched — highest-value next
target, would give a second independent test session). May 15 is low-value (weak/partial
extraction, only ~390 frames for the AM session) — not worth prioritizing.

This is enough to train on right now. Don't block training on more labeling.

## Known limitations (worth stating plainly in the paper, not hiding)

- **No true per-timestamp sync across modalities.** Visual frames, tweets, and weather are joined
  by *date* only (weather) or not synced at all (text — shared city-wide pool per day, not
  per-camera). This was an explicit, discussed tradeoff, not an oversight.
- **Only 8 of 1,409 subgraph nodes have real camera coverage.** Edges away from a camera get a risk
  prediction from graph propagation alone, never checked against ground truth.
- **Congestion labels are per-camera-relative.** "Heavy" on one camera's tercile split isn't
  necessarily the same real-world congestion level as "Heavy" on another camera.
- **Text branch is currently unused in training** (`--no-text`). The regenerated, correct
  `embeddings.pt` exists locally but was never copied to Drive for Colab to use.
- **Camera → intersection mapping is manual** (`configs/camera_nodes.csv`), derived partly from
  folder-name heuristics, partly hand-filled. All 16 extracted cameras are currently mapped.
- **Ortigas and Shaw share one graph node** (`EDSA-Ortigas-Shaw`) even though they're different
  intersections — 5 of your 16 cameras collapse onto that one node. Not fixed; would need spatial
  data regeneration to properly separate.

## Immediate next steps

1. **Confirm the v2 result holds up** — nobody has scored it yet (`predict_congestion_risk.py`
   was never run against `stgnn_edge_v2.pt`). Do that first; it's the actual next milestone, not
   more training.
2. **Get a completed, uninterrupted training run using the *current* notebook** (post all of
   today's fixes) as the "official" result for the paper — v1/v2 both predate the missing-frame
   resilience fix, so they may not reflect the most reliable pipeline available now.
3. **Try `--no-time-features` as an ablation** once a clean run exists, for the results table.
4. **Decide: retire the old node+formula pipeline, or keep it as a documented ablation?** Not yet
   decided as a team.
5. **Only after a trusted risk-score model**: start the Dynamic Weight Engine
   (`W = distance × (1 + λ·Risk)`) and Dynamic Routing Engine stages — nothing built there yet.
6. Optional, lower priority: label May 22 for a second test session; copy fresh `embeddings.pt` to
   Drive and try a text-on training run.

## Where things live

- Frames: `MMDA_FRAMES_PILOT` (May 4), `MMDA_FRAMES` (May 4/6/8/11/13/15), `MMDA_FRAMES_B`
  (May 18/20/22/25) — all on Drive, `G:\My Drive\...` locally / `/content/drive/MyDrive/...` in
  Colab.
- Checkpoints: `G:\My Drive\MMDA_CHECKPOINTS\` (`stgnn_edge_v1.pt`, `stgnn_edge_v2.pt` + matching
  `.metrics.csv`).
- Training/scoring notebook: `traffic_system/notebooks/train_and_score_colab.ipynb` — **always
  re-open fresh from GitHub after any notebook-level fix**, don't trust an already-open tab.
- Camera → intersection map: `traffic_system/configs/camera_nodes.csv`.
- Full test suite: `traffic_system/tests/` — passes clean as of `40b1074d` (run from `traffic_system/`
  with `../.venv/Scripts/python.exe -m pytest tests -q`; on Windows, running the *whole* suite in
  one go can be flaky due to multiprocessing test interference — this is a known, harmless local
  quirk, not a real failure; individual test files always pass in isolation).
