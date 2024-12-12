from abc import ABC, abstractmethod
from modeling.blocks import ConvBlock
from modeling.positional_encoding import SoftPositionEmbed
import torch
import torch.nn as nn
from typing import Iterable, Tuple


class Encoder(nn.Module, ABC):
    def __init__(self, image_size: Tuple[int, int], num_channels: Iterable[int], kernel_size: int, feature_dim: int
                 ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = 3  # Fix in_channels to 3 for RGB images
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pass


class FullyConvolutionalEncoder(Encoder):
    def __init__(self, image_size: Tuple[int, int], num_channels: Iterable[int], kernel_size: int, feature_dim: int,
                 stride: int = 1, batch_norm: bool = False, max_pool: bool = False) -> None:
        super().__init__(image_size, num_channels, kernel_size, feature_dim)
        self.stride = stride
        self.batch_norm = batch_norm
        self.max_pool = max_pool
        self.conv = self._build_encoder()

        self.positional_encoding = SoftPositionEmbed(
            hidden_size=self.num_channels[-1],
            resolution=self.image_size
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.num_channels[-1]),
            nn.Linear(self.num_channels[-1], self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

    def _build_encoder(self):
        modules = []
        in_channels = self.in_channels
        for h_dim in self.num_channels:
            block = ConvBlock(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.kernel_size // 2,
                    batch_norm=self.batch_norm,
                    max_pool=self.max_pool
                )
            modules.append(block)
            in_channels = h_dim
        encoder = nn.Sequential(*modules)
        return encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)  # x ~ (B,C,H,W)
        x = x.permute(0, 2, 3, 1)
        x = self.positional_encoding(x)  # x ~ (B,H,W,C)

        # further encoding with 1x1 Conv (implemented as shared MLP)
        x = torch.flatten(x, 1, 2)
        x = self.mlp(x)  # x ~ (B, N, Dim)
        return x
