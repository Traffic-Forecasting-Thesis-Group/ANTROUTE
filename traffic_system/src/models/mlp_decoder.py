"""
===============================================================================
ARCHITECTURE BLOCK 16: MLP Decoder
===============================================================================
Diagram path:   RADR STGNN (block 15)
                  -> [MLP DECODER]
                  -> Congestion Risk Score, per edge (block 17)

THE JOB: NODES IN, EDGES OUT
    The STGNN produces one embedding per JUNCTION (node). But the thesis
    predicts risk per ROAD SEGMENT (edge), and routing needs a cost on edges.
    This decoder is the bridge: given an edge's two endpoint embeddings, it
    outputs a single risk logit for that edge.

HOW AN EDGE IS REPRESENTED (the four parts in forward())
        h_src              the "from" junction
        h_dst              the "to" junction
        h_src - h_dst      the difference  -> direction / asymmetry
        h_src * h_dst      the interaction -> agreement between the two ends
    Concatenating all four (hence node_embedding_dim * 4) lets the decoder
    distinguish A->B from B->A, which matters because EDSA northbound and
    southbound are genuinely different roads at rush hour.

    When edge_feature_dim > 0 the edge's own road attributes (length,
    free-flow speed, from node_context.edge_features) are appended too, so
    two roads leaving the SAME junction can differ by what kind of road
    they are -- a service road and a main carriageway are not alike.

OUTPUT IS A LOGIT, NOT YET A SCORE
    forward() returns an unbounded logit. A sigmoid maps it into [0, 1] in
    the training loss / scoring step -- that bounded value is the Congestion
    Risk Score of block 17.

INPUT   <- node_embeddings [B, N, D] from block 15, edge_index [2, E],
           optional edge_features [E, edge_feature_dim]
OUTPUT  -> [B, E] risk logits, one per directed road segment
===============================================================================
"""

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

        # [BLOCK 16] How an edge is built from its two endpoints:
        #   h_src, h_dst        the two junctions
        #   h_src - h_dst       difference  -> makes A->B differ from B->A
        #   h_src * h_dst       interaction -> how much the two ends agree
        parts = [h_src, h_dst, h_src - h_dst, h_src * h_dst]
        if self.edge_feature_dim:
            if edge_features is None or edge_features.shape != (edge_index.shape[1], self.edge_feature_dim):
                raise ValueError(f"edge_features must be [{edge_index.shape[1]}, {self.edge_feature_dim}]")
            parts.append(edge_features.to(h_src.dtype).unsqueeze(0).expand(h_src.shape[0], -1, -1))
        edge_features_all = torch.cat(parts, dim=-1)

        logits = self.mlp(edge_features_all).squeeze(-1)
        return logits
