from abc import ABC, abstractmethod
from sold.utils.model_blocks import ConvBlock
import torch
import torch.nn as nn
from typing import Iterable


class Encoder(nn.Module, ABC):
    def __init__(self, num_channels: Iterable[int], kernel_size: int) -> None:
        super().__init__()
        self.in_channels = 3  # Fix in_channels to 3 for RGB images
        self.num_channels = num_channels
        self.kernel_size = kernel_size

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pass


class FullyConvolutionalEncoder(Encoder):
    def __init__(self, num_channels: Iterable[int], kernel_size: int, stride: int = 1, batch_norm: bool = False,
                 max_pool: bool = False) -> None:
        super().__init__(num_channels, kernel_size)
        self.stride = stride
        self.batch_norm = batch_norm
        self.max_pool = max_pool
        self.encoder = self._build_encoder()

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
        y = self.encoder(image)
        return y
