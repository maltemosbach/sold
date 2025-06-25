import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class CnnEncoder(nn.Module):
    def __init__(self, num_channels: List[int], kernel_sizes: List[int], strides: List[int], in_channels: int = 3
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        if not (len(num_channels) == len(kernel_sizes) == len(strides)):
            raise ValueError(f"Expected num_channels, kernel_sizes, and strides to have the same length, but got "
                             f"{len(num_channels)}, {len(kernel_sizes)}, {len(strides)}.")

        layers = []
        for layer_num, (out_channels, kernel_size, stride) in enumerate(zip(num_channels, kernel_sizes, strides)):
            layers.append(SamePadConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
            layers.append(nn.ReLU(inplace=False) if layer_num < len(num_channels) - 1 else nn.Identity())
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = images.size()
        if num_channels != self.in_channels:
            raise ValueError(f"Expected in_channels to be {self.in_channels}, but got {num_channels}.")
        embeddings = self.encoder(images)
        return embeddings


class SamePadConv2d(torch.nn.Conv2d):
    """Apply same padding so that out_size = in_size / stride."""
    @staticmethod
    def get_same_padding(size: int, kernel_size: int, stride: int, dilation: int) -> int:
        return max((math.ceil(size / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - size, 0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.size()[-2:]
        height_padding = self.get_same_padding(height, self.kernel_size[0], self.stride[0], self.dilation[0])
        width_padding = self.get_same_padding(width, self.kernel_size[1], self.stride[1], self.dilation[1])

        if height_padding > 0 or width_padding > 0:
            image = F.pad(
                image, [width_padding // 2, width_padding - width_padding // 2, height_padding // 2, height_padding - height_padding // 2]
            )
        return F.conv2d(image, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
