from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mlp_decoder import MLPDecoder
from src.models.radr_stgnn import RADRSTGNN
from src.models.stgnn_input_builder import build_stgnn_input

IGNORE_INDEX = -100
CLASS_RISK = torch.tensor([0.0, 0.5, 1.0])
N_CLASSES = 3


def scatter_camera_features(
    camera_features: torch.Tensor, camera_index: torch.Tensor, n_nodes: int, fill: torch.Tensor
) -> torch.Tensor:
    b, t, _, f = camera_features.shape
    out = fill.view(1, 1, 1, f).expand(b, t, n_nodes, f).clone()
    # Under autocast camera_features comes back float16 while `out` (built from the plain fp32
    # placeholder parameter) is float32; fancy-index assignment needs an exact dtype match.
    out[:, :, camera_index, :] = camera_features.to(out.dtype)
    return out


class TrafficRiskModel(nn.Module):
    """
    CNN+LSTM -> RADR STGNN -> MLP decoder, one Congestion Risk logit per edge.

    `flood_level` ([N] long, Project NOAH hazard level per graph node) is the static spatial
    node attribute the thesis feeds the STGNN alongside the fused camera features. Each level
    gets a learned vector of the fused-feature size that is added to that node's input at every
    timestep, so the GCN input stays at the CNN+LSTM output size (Table 2: 128). The vectors
    start at zero, so an untrained model behaves exactly as without the layer.
    """

    def __init__(
        self,
        fusion: nn.Module,
        stgnn: RADRSTGNN,
        decoder: Optional[MLPDecoder] = None,
        flood_level: Optional[torch.Tensor] = None,
        n_flood_levels: int = 4,
    ):
        super().__init__()
        self.fusion = fusion
        self.stgnn = stgnn
        self.decoder = decoder or MLPDecoder(node_embedding_dim=stgnn.output_dim)
        hidden = fusion.lstm.hidden_size
        self.placeholder = nn.Parameter(torch.zeros(hidden))
        self.flood_embedding = None
        if flood_level is not None:
            flood_level = torch.as_tensor(flood_level, dtype=torch.long)
            if flood_level.ndim != 1 or flood_level.min() < 0 or flood_level.max() >= n_flood_levels:
                raise ValueError(f"flood_level must be [N] with values in [0, {n_flood_levels - 1}]")
            self.register_buffer("flood_level", flood_level)
            self.flood_embedding = nn.Embedding(n_flood_levels, hidden)
            nn.init.zeros_(self.flood_embedding.weight)

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
        if self.flood_embedding is not None:
            if self.flood_level.shape[0] != a_hat.shape[0]:
                raise ValueError(
                    f"flood_level covers {self.flood_level.shape[0]} nodes but the graph has {a_hat.shape[0]}"
                )
            x = x + self.flood_embedding(self.flood_level).to(x.dtype)
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


def sample_weights_from_targets(
    targets: torch.Tensor, class_weights: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Per-edge BCE weight from its Light/Medium/Heavy bucket (targets are only ever
    0.0, 0.5, 1.0 or nan, since they come from CLASS_RISK). None if class_weights is None."""
    if class_weights is None:
        return None
    # round(target * 2) maps 0.0/0.5/1.0 -> 0/1/2; nan rows are excluded by the loss's own
    # mask before this is indexed, so the clamp here only has to avoid an out-of-range read.
    class_idx = torch.nan_to_num(targets * 2, nan=0.0).round().long().clamp(0, N_CLASSES - 1)
    return class_weights.to(targets.device)[class_idx]


def edge_risk_loss(
    logits: torch.Tensor,
    edge_ids: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    at_camera_edges = logits[:, edge_ids]
    mask = ~torch.isnan(targets)
    count = int(mask.sum())
    if count == 0:
        return (at_camera_edges.sum() * 0.0, 0)
    weight = sample_weights_from_targets(targets, class_weights)
    loss = F.binary_cross_entropy_with_logits(
        at_camera_edges[mask], targets[mask],
        weight=weight[mask] if weight is not None else None,
        reduction="sum",
    )
    return (loss / count, count)