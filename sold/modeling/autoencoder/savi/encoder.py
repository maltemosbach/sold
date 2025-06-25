from modeling.autoencoder.cnn.encoder import CnnEncoder
import torch
import torch.nn as nn
from typing import List, Tuple


class ImagePositionalEmbedding(nn.Module):
    """Soft image positional embedding as proposed by Slot Attention (https://arxiv.org/pdf/2006.15055, Section E.2)."""

    def __init__(self, feature_dim: int, image_size: Tuple[int, int]) -> None:
        """Initializes the positional embedding based on the image and feature dimensions.

        Args:
            feature_dim (int): Dimension of the image features the embedding is added to.
            image_size (Tuple[int, int]): Size of the input image.
        """
        super().__init__()
        self.projection = nn.Conv2d(4, feature_dim, kernel_size=1)

        # Create a grid of shape (1, 4, H, W) with gradients  [0, 1] in [x, y, -x, -y] directions.
        y, x = [torch.linspace(-1.0, 1.0, steps=resolution) for resolution in image_size]
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        position = torch.stack([yy, xx], dim=-1)  # Shape: (H, W, 2).
        grid = torch.cat([position, 1.0 - position], dim=-1)
        self.register_buffer("grid", grid.unsqueeze(0).permute(0, 3, 1, 2))  # Shape: (1, 4, H, W).

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, num_channels = image_embeddings.size()
        grid = self.grid.expand(batch_size, -1, -1, -1)
        projected_grid = self.projection(grid).permute(0, 2, 3, 1)  # Shape: (B, H, W, feature_dim)
        return image_embeddings + projected_grid


class SaviCnnEncoder(CnnEncoder):
    """CNN Encoder proposed by Slot Attention (https://arxiv.org/pdf/2006.15055, Section E.1)."""

    def __init__(self, image_size: Tuple[int, int], num_channels: List[int],
                 kernel_sizes: List[int], strides: List[int], feature_dim: int) -> None:
        super().__init__(num_channels, kernel_sizes, strides)
        self.positional_embedding = ImagePositionalEmbedding(feature_dim=num_channels[-1], image_size=image_size)
        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(num_channels[-1]),
            nn.Linear(num_channels[-1], feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = super().forward(images).permute(0, 2, 3, 1)  # Shape: (B, H, W, num_channels[-1]).
        embeddings = self.positional_embedding(embeddings)
        embeddings = embeddings.flatten(1, 2)  # Flatten x, y pos -> Shape: (B, H * W, num_channels[-1]).
        embeddings = self.shared_mlp(embeddings)  # 1 x 1 convolutions implemented as a shared MLP.
        return embeddings
