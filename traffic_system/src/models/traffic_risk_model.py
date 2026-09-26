"""
CNNLSTMFusion -> RADR STGNN -> per-node congestion classifier.

Only the camera nodes carry images, so the fusion runs on those K nodes alone and its
[B, T, K, 128] output is scattered into a [B, T, N, 128] tensor for the whole subgraph.
Every other node receives one learned placeholder vector. (build_stgnn_input on all N nodes
would allocate [B, N, T, 3, 224, 224] images: ~18 GB at N = 1000.)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.radr_stgnn import RADRSTGNN
from src.models.stgnn_input_builder import build_stgnn_input

IGNORE_INDEX = -100


def scatter_camera_features(camera_features: torch.Tensor, camera_index: torch.Tensor,
                            n_nodes: int, fill: torch.Tensor) -> torch.Tensor:
    """[B, T, K, F] -> [B, T, N, F]: camera nodes get their features, all others `fill` [F]."""
    b, t, _, f = camera_features.shape
    out = fill.view(1, 1, 1, f).expand(b, t, n_nodes, f).clone()
    out[:, :, camera_index, :] = camera_features
    return out


class TrafficRiskModel(nn.Module):
    def __init__(self, fusion: nn.Module, stgnn: RADRSTGNN, n_classes: int = 3):
        super().__init__()
        self.fusion = fusion
        self.stgnn = stgnn
        self.placeholder = nn.Parameter(torch.zeros(fusion.lstm.hidden_size))
        self.head = nn.Linear(stgnn.output_dim, n_classes)

    def forward(self, batch: Dict[str, torch.Tensor], a_hat: torch.Tensor,
                camera_index: torch.Tensor) -> torch.Tensor:
        """batch tensors are [B, K, T, ...]; returns logits [B, N, n_classes]."""
        features = build_stgnn_input(
            self.fusion, batch["images"], batch["text"], batch["temporal"],
            visual_mask=batch["visual_mask"], text_mask=batch["text_mask"],
        )                                                    # [B, T, K, 128]
        x = scatter_camera_features(features, camera_index, a_hat.shape[0], self.placeholder)
        return self.head(self.stgnn(x, a_hat))               # [B, N, n_classes]


def camera_node_loss(logits: torch.Tensor, targets: torch.Tensor, camera_index: torch.Tensor,
                     class_weights: torch.Tensor = None) -> torch.Tensor:
    """Cross-entropy on the camera nodes only; targets [B, K] use IGNORE_INDEX for unlabeled nodes."""
    at_cameras = logits[:, camera_index]                     # [B, K, C]
    return F.cross_entropy(at_cameras.reshape(-1, at_cameras.shape[-1]), targets.reshape(-1),
                           weight=class_weights, ignore_index=IGNORE_INDEX)
