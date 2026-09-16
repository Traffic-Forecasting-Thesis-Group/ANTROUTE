import torch
import torch.nn as nn

class PatchEmbedder(nn.Module):
    """
    Phase 2: 2D Patch Embedding.
    Transforms normalized CCTV frames (C, H, W) into dense 2D patch embeddings.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input normalized frames of shape (Batch, Channels, Height, Width) -> e.g., (B, 3, 224, 224)
        :return: Sequence of patch embeddings of shape (Batch, Num_Patches, Embed_Dim) -> e.g., (B, 196, 768)
        """
        # x is (B, 3, 224, 224)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        # Linear projection: (B, Embed_Dim, H/Patch, W/Patch) -> (B, 768, 14, 14)
        x = self.proj(x)

        # Flatten spatial dimensions: (B, Embed_Dim, Num_Patches) -> (B, 768, 196)
        x = x.flatten(2)

        # Transpose to yield sequential features: (B, Num_Patches, Embed_Dim) -> (B, 196, 768)
        x = x.transpose(1, 2)

        # Add spatial positional embeddings
        x = x + self.pos_embed

        return x

if __name__ == "__main__":
    # Test the embedder locally
    model = PatchEmbedder()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"PatchEmbedder Input Shape: {dummy_input.shape}")
    print(f"PatchEmbedder Output Shape: {output.shape} (Batch, Num_Patches, Embed_Dim)")
