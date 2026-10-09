"""
===============================================================================
ARCHITECTURE BLOCK 17: Congestion Risk Score (predicted risk per edge)
===============================================================================
Diagram path:   MLP Decoder (block 16)
                  -> [CONGESTION RISK SCORE]
                  -> Dynamic Weight Engine (block 18)
                  ^
                  +-- compared in the training loss against
                      YOLOv8 Auto-Labeling (block 14, the training target)

THIS IS THE CENTRE OF THE THESIS
    Everything upstream exists to produce this number; everything downstream
    consumes it. For each directed road segment it is ONE value in [0, 1]:

        0.0  Light    free flowing
        0.5  Medium
        1.0  Heavy    congested

    Those three anchor values are exactly what the Light/Medium/Heavy labels
    from block 14 are mapped to, which is how prediction and target become
    comparable in the training loss.

THE SIGMOID IS THE WHOLE CONVERSION
    The decoder emits an unbounded logit; torch.sigmoid squashes it into
    [0, 1]. That bound is what makes the routing formula well behaved: in
    W = distance * (1 + lambda * Risk) a risk of 1.0 with lambda = 2.0 makes
    an edge cost exactly 3x its length, and no edge can cost more.

TWO SCORE COLUMNS EXIST DOWNSTREAM
    The raw score above, plus `risk_calibrated` from
    src/models/risk_calibration.py. The calibration is affine and monotone,
    so it changes the NUMBERS but never the ranking -- and therefore never
    the chosen route. Both are kept so the paper can report them side by side.

NOTE ON SCALE (state this with any routing result)
    In the shipped scoring run only ~1.5% of city edges (42 of 2,894) are
    directly scored against a camera. The rest are filled in by the routing
    layer from the nearest scored road; see dynamic_weight.fill_unscored_risk.

INPUT   <- node embeddings (block 15) + the trained MLP decoder (block 16)
OUTPUT  -> risk per edge; written to data/processed/risk_scores/risk_edges.csv
           by scripts/predict_congestion_risk.py, read by block 18

KEY NAMES
    compute_congestion_risk_scores()  logit -> sigmoid -> risk in [0, 1]
    risk_scores_to_dataframe()        tensor -> (source, target, score) rows
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn


def edge_index_from_adjacency(adj: sp.csr_matrix) -> torch.Tensor:
    coo = adj.tocoo()
    edge_index = np.vstack([coo.row, coo.col])
    return torch.tensor(edge_index, dtype=torch.long)


def compute_congestion_risk_scores(decoder: nn.Module, node_embeddings: torch.Tensor,
                                    edge_index: torch.Tensor) -> torch.Tensor:
    logits = decoder(node_embeddings, edge_index)
    # [BLOCK 17] The one line that makes it a "score": sigmoid bounds the
    # decoder's logit into [0, 1], matching the Light/Medium/Heavy -> 0/0.5/1
    # scale of the training target and keeping the routing weight bounded.
    risk_scores = torch.sigmoid(logits)
    return risk_scores


def risk_scores_to_dataframe(risk_scores: torch.Tensor, edge_index: torch.Tensor,
                              node_order: list, batch_index: int = 0) -> pd.DataFrame:
    src_idx, dst_idx = edge_index[0].tolist(), edge_index[1].tolist()
    scores = risk_scores[batch_index].detach().cpu().numpy()
    return pd.DataFrame({
        "source_node_id": [node_order[i] for i in src_idx],
        "target_node_id": [node_order[i] for i in dst_idx],
        "congestion_risk_score": scores,
    })