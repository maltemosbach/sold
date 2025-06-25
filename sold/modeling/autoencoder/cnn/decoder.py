from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
from typing import List, Tuple


class Decoder(nn.Module, ABC):
    def __init__(self, image_size: Tuple[int, int], out_channels: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        pass


class CnnDecoder(Decoder):
    def __init__(self, embedding_dim: int, image_size: Tuple[int, int], num_channels: List[int],
                 kernel_sizes: List[int], strides: List[int]) -> None:
        super().__init__(image_size, 3)
        assert len(num_channels) == len(kernel_sizes) == len(strides), \
            "num_channels, kernel_sizes, and strides must have the same length"
        self.image_size = image_size
        self.feature_map_size = torch.LongTensor(image_size)
        for stride in strides:
            self.feature_map_size //= stride

        self.embeddings_to_feature_map = nn.Linear(embedding_dim, int(num_channels[0] * self.feature_map_size.prod()))

        layers = []
        in_channels = num_channels[0]
        for i, (out_channels, kernel, stride) in enumerate(zip(num_channels[1:] + [self.out_channels], kernel_sizes, strides)):
            padding, output_padding = self.get_same_padding(kernel, stride, 1)

            layers.append(nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel, stride=stride,
                padding=padding, output_padding=output_padding
            ))
            is_last = i == len(num_channels) - 1
            layers.append(nn.Sigmoid() if is_last else nn.ReLU(inplace=False))
            in_channels = out_channels

        self.decoder = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = embeddings.size(0)
        feature_map = self.embeddings_to_feature_map(embeddings).view(batch_size, -1, *self.feature_map_size)
        return self.decoder(feature_map).unsqueeze(1)

    @staticmethod
    def get_same_padding(kernel_size: int, stride: int, dilation: int) -> Tuple[int, int]:
        val = dilation * (kernel_size - 1) - stride + 1
        padding = math.ceil(val / 2)
        output_padding = padding * 2 - val
        return padding, output_padding

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]
