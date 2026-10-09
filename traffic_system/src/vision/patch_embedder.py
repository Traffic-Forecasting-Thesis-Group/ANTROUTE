from typing import Optional, Sequence

import torch
import torch.nn as nn

# The input statistics torchvision's ImageNet-1k ViT-B/16 weights were trained with. Frames
# reach the encoder in [0, 1]; pretrained patch filters expect them standardised like this.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchEmbedder(nn.Module):
    """
    Phase 2: 2D Patch Embedding.
    Transforms normalized CCTV frames (C, H, W) into dense 2D patch embeddings.

    `input_mean` / `input_std` (optional) standardise the [0, 1] frames per channel first. They
    are only set for pretrained patch weights (load_vit_b16_patch_weights), and are kept out of
    the state dict (non-persistent), so checkpoints trained without them load unchanged.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768,
                 input_mean: Optional[Sequence[float]] = None, input_std: Optional[Sequence[float]] = None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patchification and Linear Projection using a single 2D Convolution layer.
        # Kernel size and stride equal to patch size accomplishes non-overlapping patches.
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Standard Vision Transformer positional embeddings (optional but highly recommended for spatial awareness)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.normalize_input = input_mean is not None and input_std is not None
        mean = torch.tensor(input_mean if self.normalize_input else [0.0] * in_chans).view(1, in_chans, 1, 1)
        std = torch.tensor(input_std if self.normalize_input else [1.0] * in_chans).view(1, in_chans, 1, 1)
        self.register_buffer("input_mean", mean, persistent=False)
        self.register_buffer("input_std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input normalized frames of shape (Batch, Channels, Height, Width) -> e.g., (B, 3, 224, 224)
        :return: Sequence of patch embeddings of shape (Batch, Num_Patches, Embed_Dim) -> e.g., (B, 196, 768)
        """
        # x is (B, 3, 224, 224)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        if self.normalize_input:
            x = (x - self.input_mean.to(x.dtype)) / self.input_std.to(x.dtype)

        # Linear projection: (B, Embed_Dim, H/Patch, W/Patch) -> (B, 768, 14, 14)
        x = self.proj(x)

        # Flatten spatial dimensions: (B, Embed_Dim, Num_Patches) -> (B, 768, 196)
        x = x.flatten(2)

        # Transpose to yield sequential features: (B, Num_Patches, Embed_Dim) -> (B, 196, 768)
        x = x.transpose(1, 2)

        # Add spatial positional embeddings
        x = x + self.pos_embed

        return x


def vit_b16_patch_weights():
    """
    (patch projection weight [768, 3, 16, 16], bias [768], patch position embeddings
    [1, 196, 768]) of torchvision's ImageNet-1k ViT-B/16 (downloads on first use).
    """
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    return (model.conv_proj.weight.detach().clone(), model.conv_proj.bias.detach().clone(),
            model.encoder.pos_embedding[:, 1:, :].detach().clone())   # drop the class-token position


@torch.no_grad()
def use_imagenet_normalisation(embedder: PatchEmbedder) -> None:
    """Standardise input frames with ImageNet statistics. A checkpoint trained from ViT patch
    weights needs this switched back on when it is rebuilt (the statistics are not saved)."""
    embedder.normalize_input = True
    embedder.input_mean.copy_(torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
    embedder.input_std.copy_(torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))


@torch.no_grad()
def load_vit_b16_patch_weights(embedder: PatchEmbedder, weights=None) -> dict:
    """
    Initialise `embedder`'s patch projection and position embeddings from ViT-B/16 and switch
    on the ImageNet input standardisation those weights expect.

    Matching shapes is necessary, not sufficient: in ViT these filters feed a transformer, here
    a CNN (trained from scratch) reads them as a 14 x 14 feature map. First-layer patch filters
    are generic colour / edge detectors and usually transfer, but whether they help on these
    CCTV frames is an empirical question -- compare against random initialisation on the
    validation split over several seeds (scripts/summarize_runs.py) before relying on it.
    """
    weight, bias, pos = weights if weights is not None else vit_b16_patch_weights()
    expected = (tuple(embedder.proj.weight.shape), tuple(embedder.proj.bias.shape), tuple(embedder.pos_embed.shape))
    received = (tuple(weight.shape), tuple(bias.shape), tuple(pos.shape))
    if expected != received:
        raise ValueError(f"ViT-B/16 patch weights {received} do not fit this embedder {expected}")
    embedder.proj.weight.copy_(weight)
    embedder.proj.bias.copy_(bias)
    embedder.pos_embed.copy_(pos)
    use_imagenet_normalisation(embedder)
    return {"source": "torchvision ViT_B_16_Weights.IMAGENET1K_V1", "layers": ["proj", "pos_embed"],
            "input_normalisation": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}}


@torch.no_grad()
def patch_feature_stats(embedder: PatchEmbedder, frames: torch.Tensor) -> dict:
    """
    How the patch tokens of real frames look (a quick compatibility check, not a quality
    measure): mean / std of the tokens, the share of channels that barely vary across the
    frames (near-dead filters), and the mean cosine similarity between different frames'
    pooled tokens (close to 1 means the filters cannot tell the frames apart).
    """
    tokens = embedder(frames)                                    # [N, P, D]
    pooled = tokens.mean(dim=1)                                  # [N, D]
    channel_sd = tokens.reshape(-1, tokens.shape[-1]).std(dim=0)
    unit = torch.nn.functional.normalize(pooled - pooled.mean(0, keepdim=True), dim=1)
    sim = unit @ unit.T
    n = sim.shape[0]
    off_diag = (sim.sum() - sim.diagonal().sum()) / max(n * (n - 1), 1)
    return {"token_mean": float(tokens.mean()), "token_sd": float(tokens.std()),
            "near_dead_channels": float((channel_sd < 1e-3 * channel_sd.max()).float().mean()),
            "between_frame_cosine": float(off_diag)}


if __name__ == "__main__":
    # Test the embedder locally
    model = PatchEmbedder()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"PatchEmbedder Input Shape: {dummy_input.shape}")
    print(f"PatchEmbedder Output Shape: {output.shape} (Batch, Num_Patches, Embed_Dim)")
