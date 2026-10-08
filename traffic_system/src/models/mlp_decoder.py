from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    Edge risk logit from its two endpoint embeddings and, when edge_feature_dim > 0, the
    edge's own road attributes (node_context.edge_features: length, free-flow speed), so two
    roads leaving the same junction can differ by what kind of road they are.
    """

    def __init__(self, node_embedding_dim: int, hidden_dim: int = 64, dropout: float = 0.1,
                 edge_feature_dim: int = 0):
        super().__init__()
        self.edge_feature_dim = edge_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(node_embedding_dim * 4 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        h_src = node_embeddings[:, src_idx, :]
        h_dst = node_embeddings[:, dst_idx, :]

        parts = [h_src, h_dst, h_src - h_dst, h_src * h_dst]
        if self.edge_feature_dim:
            if edge_features is None or edge_features.shape != (edge_index.shape[1], self.edge_feature_dim):
                raise ValueError(f"edge_features must be [{edge_index.shape[1]}, {self.edge_feature_dim}]")
            parts.append(edge_features.to(h_src.dtype).unsqueeze(0).expand(h_src.shape[0], -1, -1))
        edge_features_all = torch.cat(parts, dim=-1)

        logits = self.mlp(edge_features_all).squeeze(-1)
        return logits
