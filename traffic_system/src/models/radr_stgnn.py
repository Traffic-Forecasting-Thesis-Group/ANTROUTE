"""
===============================================================================
ARCHITECTURE BLOCK 15: RADR STGNN
                       (Spatial GCN + Temporal GRU Modeling)
===============================================================================
Diagram path:   CNN + LSTM          (block 9)  ---+
                Sequence Norm       (block 11) ---+--> [RADR STGNN]
                Geocoding & Topology(block 13) ---+          |
                                                             v
                                                  MLP Decoder (block 16)

WHAT "SPATIO-TEMPORAL" ACTUALLY MEANS HERE
    Two stacked ideas, and forward() below is literally these two steps:

      SPATIAL (GCN)   For each timestep independently, a Graph Convolutional
                      Network passes information between CONNECTED ROADS.
                      This is what lets congestion spread: a jam at EDSA x
                      Shaw raises the predicted risk of its neighbours even
                      though no camera watches them. Without it, the 8-9
                      camera junctions would be the only roads we could say
                      anything about.

      TEMPORAL (GRU)  The sequence of per-timestep spatial embeddings is then
                      read by a GRU, so the model sees how the whole network's
                      state is EVOLVING -- building vs clearing -- rather than
                      judging each timestep alone.

    Order matters: GCN first (space), then GRU over the result (time). See
    RADRSTGNN.forward(): loop the GCN over timesteps, stack, then run the GRU
    and keep its final hidden state.

WHY THE ADJACENCY IS NORMALISED
    normalize_adjacency() builds A_hat = D^-1/2 (A + I) D^-1/2. Adding I keeps
    a road's own features (self-loop); the D^-1/2 scaling stops high-degree
    junctions from dominating purely because they have more neighbours.

INPUT   <- x: [B, T, N, F] per-node features. Camera nodes carry CNN+LSTM
           output (block 9); every node also carries 17 context features
           (weather / flood / events / temporal / spatial) from
           src/data/node_context.py.
           a_hat: normalised adjacency from block 13.
OUTPUT  -> [B, N, gru_hidden] one embedding per road junction, handed to the
           MLP Decoder (block 16)

KEY NAMES
    normalize_adjacency()  A_hat, the symmetric normalisation
    GCNLayer / GCNEncoder  the SPATIAL half
    GRUTemporalLayer       the TEMPORAL half
    RADRSTGNN.forward()    the two halves in order
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_adjacency_npz(path: Path) -> sp.csr_matrix:
    return sp.load_npz(path).tocsr()


# [BLOCK 15] A_hat = D^-1/2 (A + I) D^-1/2. The +I keeps each road's own
# features; the D^-1/2 scaling stops busy junctions dominating by degree alone.
def normalize_adjacency(adj: sp.csr_matrix) -> torch.Tensor:
    n = adj.shape[0]
    adj = adj.astype(bool).astype(np.float32)
    adj_tilde = adj + sp.eye(n, format="csr", dtype=np.float32)
    deg = np.asarray(adj_tilde.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    a_hat = D_inv_sqrt @ adj_tilde @ D_inv_sqrt
    return torch.tensor(a_hat.toarray(), dtype=torch.float32)


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=F.relu, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        propagated = torch.einsum("ij,bjf->bif", a_hat, support)
        out = self.activation(propagated) if self.activation is not None else propagated
        return self.dropout(out)


class GCNEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert num_layers >= 1
        layers = []
        dims = [in_features] + [hidden_features] * (num_layers - 1) + [out_features]
        for i in range(num_layers):
            is_last = i == num_layers - 1
            layers.append(GCNLayer(
                dims[i], dims[i + 1],
                activation=None if is_last else F.relu,
                dropout=0.0 if is_last else dropout,
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, a_hat)
        return x


class GRUTemporalLayer(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_features = hidden_features
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_features,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, n, f = x.shape
        x_folded = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        seq_out, h_n = self.gru(x_folded)
        seq_out = seq_out.reshape(b, n, t, self.hidden_features).permute(0, 2, 1, 3)
        final_state = h_n[-1].reshape(b, n, self.hidden_features)
        return seq_out, final_state


class RADRSTGNN(nn.Module):
    def __init__(
        self,
        in_features: int = 128,
        gcn_hidden: int = 64,
        gcn_out: int = 32,
        gcn_layers: int = 2,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gcn = GCNEncoder(in_features, gcn_hidden, gcn_out, num_layers=gcn_layers, dropout=dropout)
        self.gru = GRUTemporalLayer(gcn_out, gru_hidden, num_layers=gru_layers, dropout=dropout)
        self.output_dim = gru_hidden

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # [BLOCK 15] The whole spatio-temporal idea, in seven lines.
        b, t, n, f = x.shape

        # 1. SPATIAL: run the GCN at each timestep separately, so congestion
        #    propagates between connected roads within that snapshot.
        spatial_embeddings = []
        for step in range(t):
            spatial_embeddings.append(self.gcn(x[:, step], a_hat))
        spatial_seq = torch.stack(spatial_embeddings, dim=1)

        # 2. TEMPORAL: read that sequence of network snapshots over time.
        #    Only the final hidden state is kept -- the network's state at the
        #    end of the window, which is what we predict risk from.
        _, final_state = self.gru(spatial_seq)
        return final_state