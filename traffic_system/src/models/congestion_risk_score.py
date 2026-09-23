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