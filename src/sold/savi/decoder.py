from abc import ABC, abstractmethod
from sold.utils.model_blocks import ConvBlock
import torch
import torch.nn as nn
from typing import Iterable


class Decoder(nn.Module, ABC):
    def __init__(self, in_channels: int, num_channels: Iterable[int], kernel_size: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 4  # Fix out_channels to 4 for RGB + Mask
        self.num_channels = num_channels
        self.kernel_size = kernel_size

    @abstractmethod
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        pass


class FullyConvolutionalDecoder(Decoder):
    def __init__(self, in_channels: int, num_channels: Iterable[int], kernel_size: int, stride: int = 1,
                 batch_norm: bool = False) -> None:
        super().__init__(in_channels, num_channels, kernel_size)
        self.stride = stride
        self.batch_norm = batch_norm
        self.decoder = self._build_decoder()

    def _build_decoder(self):
        """
        Creating convolutional decoder given dimensionality parameters
        By default, it maps feature maps to a 5dim output, containing
        RGB objects and binary mask:
           (B,C,H,W)  -- > (B, N_S, 4, H, W)
        """
        modules = []
        in_channels = self.in_channels

        # adding convolutional layers to decoder
        for i in range(len(self.num_channels) - 1, -1, -1):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=self.num_channels[i],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                batch_norm=self.batch_norm,
            )
            in_channels = self.num_channels[i]
            modules.append(block)

        # final conv layer
        final_conv = nn.Conv2d(
            in_channels=self.num_channels[-1],
            out_channels=self.out_channels,  # RGB + Mask
            kernel_size=3,
            stride=1,
            padding=1
        )
        modules.append(final_conv)

        decoder = nn.Sequential(*modules)
        return decoder

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        y = self.decoder(slots)
        return y
