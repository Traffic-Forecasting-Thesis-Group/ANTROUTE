from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mlp_decoder import MLPDecoder
from src.models.radr_stgnn import RADRSTGNN
from src.models.stgnn_input_builder import build_stgnn_input

IGNORE_INDEX = -100
CLASS_RISK = torch.tensor([0.0, 0.5, 1.0])


def scatter_camera_features(
    camera_features: torch.Tensor, camera_index: torch.Tensor, n_nodes: int, fill: torch.Tensor
) -> torch.Tensor:
    b, t, _, f = camera_features.shape
    out = fill.view(1, 1, 1, f).expand(b, t, n_nodes, f).clone()
    out[:, :, camera_index, :] = camera_features
    return out


class TrafficRiskModel(nn.Module):

    def __init__(self, fusion: nn.Module, stgnn: RADRSTGNN, decoder: Optional[MLPDecoder] = None):
        super().__init__()
        self.fusion = fusion
        self.stgnn = stgnn
        self.decoder = decoder or MLPDecoder(node_embedding_dim=stgnn.output_dim)
        self.placeholder = nn.Parameter(torch.zeros(fusion.lstm.hidden_size))

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        a_hat: torch.Tensor,
        camera_index: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        features = build_stgnn_input(
            self.fusion,
            batch["images"],
            batch["text"],
            batch["temporal"],
            visual_mask=batch["visual_mask"],
            text_mask=batch["text_mask"],
        )
        x = scatter_camera_features(features, camera_index, a_hat.shape[0], self.placeholder)
        nodes = self.stgnn(x, a_hat)
        return self.decoder(nodes, edge_index)


def camera_edge_ids(edge_index: torch.Tensor, camera_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    is_camera = torch.zeros(n_nodes, dtype=torch.bool)
    is_camera[camera_index.cpu()] = True
    src, dst = (edge_index[0].cpu(), edge_index[1].cpu())
    touching = is_camera[src] | is_camera[dst]
    ids = touching.nonzero(as_tuple=True)[0]
    if len(ids) == 0:
        raise ValueError("No graph edge touches a camera node.")
    return ids


def camera_edge_targets(
    target: torch.Tensor,
    camera_index: torch.Tensor,
    n_nodes: int,
    edge_index: torch.Tensor,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    known = target != IGNORE_INDEX
    values = CLASS_RISK.to(target.device)[target.clamp(min=0)]
    values = torch.where(known, values, torch.full_like(values, float("nan")))
    node_risk = torch.full((target.shape[0], n_nodes), float("nan"), device=target.device)
    node_risk[:, camera_index.to(target.device)] = values
    src, dst = (edge_index[0, edge_ids], edge_index[1, edge_ids])
    pair = torch.stack([node_risk[:, src], node_risk[:, dst]], dim=-1)
    return torch.nanmean(pair, dim=-1)


def edge_risk_loss(
    logits: torch.Tensor, edge_ids: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    at_camera_edges = logits[:, edge_ids]
    mask = ~torch.isnan(targets)
    count = int(mask.sum())
    if count == 0:
        return (at_camera_edges.sum() * 0.0, 0)
    loss = F.binary_cross_entropy_with_logits(at_camera_edges[mask], targets[mask], reduction="sum")
    return (loss / count, count)