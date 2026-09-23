from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_adjacency_npz(path: Path) -> sp.csr_matrix:
    return sp.load_npz(path).tocsr()


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
        b, t, n, f = x.shape
        spatial_embeddings = []
        for step in range(t):
            spatial_embeddings.append(self.gcn(x[:, step], a_hat))
        spatial_seq = torch.stack(spatial_embeddings, dim=1)

        _, final_state = self.gru(spatial_seq)
        return final_state