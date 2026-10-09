from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mlp_decoder import MLPDecoder
from src.models.radr_stgnn import RADRSTGNN
from src.models.stgnn_input_builder import build_stgnn_input, scatter_camera_features

IGNORE_INDEX = -100
CLASS_RISK = torch.tensor([0.0, 0.5, 1.0])
N_CLASSES = 3


class ContextEncoder(nn.Module):
    """
    Per-node, per-minute context (src/data/node_context.py: daily weather of the node's cell,
    flood hazard and its interaction with rain, live incidents, clock, road attributes) ->
    a vector of the CNN+LSTM output size, added to every node's STGNN input.

    This is what lets the non-visual sources reach every road: a node without a camera used
    to get one shared placeholder, so its risk could only change through camera features
    carried at most two GCN hops.
    """

    def __init__(self, context_dim: int, hidden: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim, width), nn.ReLU(), nn.Linear(width, hidden))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class TrafficRiskModel(nn.Module):
    """
    CNN+LSTM -> RADR STGNN -> MLP decoder, one Congestion Risk logit per edge.

    STGNN input per node and timestep:
        camera nodes:  CNN+LSTM fused features (CCTV + its text + its temporal input)
        other nodes:   a learned placeholder ("no camera here")
        every node:    + ContextEncoder(context) when context_dim > 0
                       + the legacy flood_level embedding, for checkpoints trained with it

    `flood_level` ([N] long, Project NOAH hazard level per graph node) is the older way flood
    entered: a learned vector per level, zero-initialised. New models carry flood inside the
    context instead (ordinal and x rainfall), so it is kept only to load those checkpoints.

    `edge_features` ([E, D], node_context.edge_features) are per-edge road attributes passed to
    the decoder, aligned to the edge_index given at forward time.
    """

    def __init__(
        self,
        fusion: nn.Module,
        stgnn: RADRSTGNN,
        decoder: Optional[MLPDecoder] = None,
        flood_level: Optional[torch.Tensor] = None,
        n_flood_levels: int = 4,
        context_dim: int = 0,
        edge_features: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.fusion = fusion
        self.stgnn = stgnn
        edge_feature_dim = 0 if edge_features is None else int(edge_features.shape[1])
        self.decoder = decoder or MLPDecoder(node_embedding_dim=stgnn.output_dim, edge_feature_dim=edge_feature_dim)
        if self.decoder.edge_feature_dim != edge_feature_dim:
            raise ValueError("the decoder's edge_feature_dim must match edge_features")
        hidden = fusion.lstm.hidden_size
        self.placeholder = nn.Parameter(torch.zeros(hidden))
        self.context_encoder = ContextEncoder(context_dim, hidden) if context_dim else None
        if edge_features is not None:
            self.register_buffer("edge_features", torch.as_tensor(edge_features, dtype=torch.float32))
        else:
            self.edge_features = None
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
        if self.context_encoder is not None:
            context = batch.get("context")
            if context is None or tuple(context.shape[:3]) != tuple(x.shape[:3]):
                raise ValueError(
                    f"this model needs batch['context'] of shape [B, T, N, C] matching {tuple(x.shape[:3])}"
                )
            x = x + self.context_encoder(context.to(x.dtype)).to(x.dtype)
        nodes = self.stgnn(x, a_hat)
        return self.decoder(nodes, edge_index, self.edge_features)


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