from __future__ import annotations

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self, node_embedding_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        h_src = node_embeddings[:, src_idx, :]
        h_dst = node_embeddings[:, dst_idx, :]

        edge_features = torch.cat(
            [h_src, h_dst, h_src - h_dst, h_src * h_dst],
            dim=-1,
        )

        logits = self.mlp(edge_features).squeeze(-1)
        return logits