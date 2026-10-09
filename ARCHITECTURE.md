# ANTROUTE — Architecture Diagram to Code Map

This is the bridge between the system architecture diagram and the actual files.
Every block on the diagram is numbered here, and every file listed carries a matching
`ARCHITECTURE BLOCK <n>` banner at the top plus `# [BLOCK n]` anchors on the lines that
do the real work. So you can search the codebase for `BLOCK 15` and land exactly on the
RADR STGNN.

**Where to start reading:** [`src/models/traffic_risk_model_edge.py`](traffic_system/src/models/traffic_risk_model_edge.py).
Its `TrafficRiskModel.forward()` is the entire bottom half of the diagram in about twenty
lines — CNN+LSTM → RADR STGNN → MLP Decoder → Congestion Risk Score.

---

## The map

### Text branch (left column)

| # | Diagram block | File | Key names |
|---|---|---|---|
| 1 | Unstructured Data (X/Twitter) | [`src/ingestion/twitter_scraper.py`](traffic_system/src/ingestion/twitter_scraper.py) | `build_query()`, `retrieve_twitter_data()` |
| 2 | Textual Pre-cleaning | [`src/preprocessing/text_cleaner.py`](traffic_system/src/preprocessing/text_cleaner.py) | `remove_noise()`, `replace_slang()`, `extract_and_preserve_entities()` |
| 3 | MarianMT (Tagalog→English) | [`src/preprocessing/nlp_pipeline.py`](traffic_system/src/preprocessing/nlp_pipeline.py) | `_is_english()`, step 1 of `process_batch()` |
| 4 | Tokenizer | same file | step 2 of `process_batch()` |
| 5 | DistilBERT (Text Embedding) | same file | step 3 of `process_batch()` |

> Blocks 3–5 are three boxes on the diagram but one class in code (`BatchNLPPipeline`),
> because they always run together on the same batch. Read `process_batch()` top to
> bottom and you are walking all three.

### Visual branch

| # | Diagram block | File | Key names |
|---|---|---|---|
| 6 | Visual Data (MMDA CCTV) | [`src/vision/frame_extractor.py`](traffic_system/src/vision/frame_extractor.py) | `discover_segments()` |
| 7 | Frame Extraction & Normalization | same file | `sample_frames()`, `resize_max_side()`, `extract_all()` |
| 8 | 2D Patch Embedding | [`src/vision/patch_embedder.py`](traffic_system/src/vision/patch_embedder.py) | `PatchEmbedder` |
| 9 | CNN + LSTM | [`src/models/cnn_lstm_fusion.py`](traffic_system/src/models/cnn_lstm_fusion.py) | `VisualCNNEncoder`, `CNNLSTMFusion` |

> Block 8 is **not** a separate saved step. It is the first learned layer inside block 9,
> trained end-to-end.
>
> Block 9 is where the diagram's three input arrows converge — look for `self.fused_input_dim`.

### Temporal & Spatial branches

| # | Diagram block | File | Key names |
|---|---|---|---|
| 10 | Temporal Data (WeatherStack) | [`src/ingestion/temporal_weather.py`](traffic_system/src/ingestion/temporal_weather.py) | `build_grid_points()` |
| 11 | Sequence Norm (Weather) | same file | `zscore_normalize()`, `build_sequences()` |
| 12 | Spatial Data (OSMnx, Project NOAH) | [`src/ingestion/spatial_topology.py`](traffic_system/src/ingestion/spatial_topology.py) | `load_or_build_graph()`, `fetch_flood_hazard()` |
| 13 | Geocoding & Topology | same file | `locate_key_intersections()`, `build_adjacency()` |

> **How these actually reach the model:** [`src/data/node_context.py`](traffic_system/src/data/node_context.py).
> The arrows from the Temporal and Spatial branches into the RADR STGNN *are* this file —
> it attaches 17 features to every node in 5 groups (weather, flood, events, temporal, spatial).

### Training target

| # | Diagram block | File | Key names |
|---|---|---|---|
| 14 | YOLOv8 Auto-Labeling | [`src/vision/congestion_autolabel.py`](traffic_system/src/vision/congestion_autolabel.py) | `detect_batch()`, `occupancy()`, `tercile_thresholds()` |

> Its arrow is **dashed** on the diagram and labelled *(training target)*. The model never
> sees these labels as a feature — they are what predictions are scored against in the loss.
> Confusing this for an input is the easiest way to misread the architecture.

### Core model

| # | Diagram block | File | Key names |
|---|---|---|---|
| 15 | RADR STGNN (GCN + GRU) | [`src/models/radr_stgnn.py`](traffic_system/src/models/radr_stgnn.py) | `GCNEncoder`, `GRUTemporalLayer`, `RADRSTGNN.forward()` |
| 16 | MLP Decoder | [`src/models/mlp_decoder.py`](traffic_system/src/models/mlp_decoder.py) | `MLPDecoder.forward()` |
| 17 | Congestion Risk Score | [`src/models/congestion_risk_score.py`](traffic_system/src/models/congestion_risk_score.py) | `compute_congestion_risk_scores()` |
| — | **all of the above, assembled** | [`src/models/traffic_risk_model_edge.py`](traffic_system/src/models/traffic_risk_model_edge.py) | `TrafficRiskModel.forward()`, `edge_risk_loss()` |
| — | writes block 17 to disk | [`scripts/predict_congestion_risk.py`](traffic_system/scripts/predict_congestion_risk.py) | → `risk_edges.csv` |

> **Nodes in, edges out:** the STGNN produces one embedding per *junction*, but risk is
> predicted per *road segment*. Block 16 is the bridge — see the four-part edge
> representation in `MLPDecoder.forward()`.

### Routing

| # | Diagram block | File | Key names |
|---|---|---|---|
| 18 | Dynamic Weight Engine | [`src/routing/dynamic_weight.py`](traffic_system/src/routing/dynamic_weight.py) | `dynamic_weight()`, `DEFAULT_LAMBDA = 2.0`, `fill_unscored_risk()` |
| 19 | Dynamic Routing Engine | [`src/routing/aco_routing.py`](traffic_system/src/routing/aco_routing.py) | `ant_colony_shortest_path()`, `multi_stop_route()` |
| 19 | └ ETA Estimation | [`src/routing/eta_engine.py`](traffic_system/src/routing/eta_engine.py) | `congested_eta()`, `DEFAULT_GAMMA = 1.0` |
| 20 | Optimal Path | [`src/routing/aco_routing.py`](traffic_system/src/routing/aco_routing.py) | `diverse_routes()` |
| 20 | └ served live | [`app/risk_routing.py`](traffic_system/app/risk_routing.py) | `plan_real_routes()` |

> **Block 18 is the hinge of the whole system.** Everything above it predicts congestion;
> everything below it finds routes. `W = distance × (1 + λ·Risk)` is the one line where a
> prediction becomes a routing decision.

---

## Not on the diagram, but you will trip over it

| File | What it is |
|---|---|
| [`src/routing/baseline_iaco.py`](traffic_system/src/routing/baseline_iaco.py) | **The comparison baseline**, Cheng (2023) Improved ACO — *not* our method. Both are ant colony algorithms, which is why they get mixed up. The difference: ours routes on **predicted** risk, this one on **observed** traffic. |
| [`src/routing/event_layer.py`](traffic_system/src/routing/event_layer.py) | Rule-based MMDA incident penalty. Its own header says it is **not part of the thesis method** — check `SESSION_HANDOFF.md` before reporting any result that includes it. |
| [`src/models/risk_calibration.py`](traffic_system/src/models/risk_calibration.py) | Affine map correcting the risk score's offset. Monotone, so it never changes a route. |
| [`src/models/risk_baselines.py`](traffic_system/src/models/risk_baselines.py) | Constant / train-mean / history baselines the model must beat. |
| [`src/models/traffic_risk_model.py`](traffic_system/src/models/traffic_risk_model.py) | **Superseded** per-node pipeline, kept pending a team decision. The per-edge model above is the one in use. |

---

## Reading order for a first pass

1. This file, to locate things.
2. `src/models/traffic_risk_model_edge.py` — the whole model in one `forward()`.
3. Follow one branch backwards from there, e.g. block 9 → 8 → 7.
4. `src/routing/dynamic_weight.py` — where prediction becomes routing.
5. `src/routing/aco_routing.py` — how the route is actually found.
6. `SESSION_HANDOFF.md` — current status, real numbers, and what is still open.
