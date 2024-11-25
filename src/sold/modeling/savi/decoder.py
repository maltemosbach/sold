from abc import ABC, abstractmethod
from sold.modeling.positional_encoding import SoftPositionEmbed
from sold.modeling.blocks import ConvBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple


class Decoder(nn.Module, ABC):
    def __init__(self, image_size: Tuple[int, int], slot_dim: int, in_channels: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.slot_dim = slot_dim
        self.in_channels = in_channels
        self.out_channels = 4  # Fix out_channels to 4 for RGB + Mask

    @abstractmethod
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]


class FullyConvolutionalDecoder(Decoder):
    def __init__(self, image_size: Tuple[int, int], slot_dim: int, in_channels: int, num_channels: Iterable[int],
                 kernel_size: int, stride: int = 1, batch_norm: bool = False, upsample: Optional[int] = None) -> None:
        super().__init__(image_size, slot_dim, in_channels)
        self.stride = stride
        self.batch_norm = batch_norm
        self.decoder = self._build_decoder(in_channels, num_channels, kernel_size, stride, batch_norm, upsample)
        self.positional_encoding = SoftPositionEmbed(
            hidden_size=slot_dim,
            resolution=image_size
        )

    def _build_decoder(self, in_channels: int, num_channels: Iterable[int], kernel_size: int, stride: int, batch_norm: bool, upsample: int) -> nn.Sequential:
        """
        Creating convolutional decoder given dimensionality parameters
        By default, it maps feature maps to a 5dim output, containing
        RGB objects and binary mask:
           (B,C,H,W)  -- > (B, N_S, 4, H, W)
        """
        modules = []

        # adding convolutional layers to decoder
        for i in range(len(num_channels) - 1, -1, -1):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=num_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                batch_norm=batch_norm,
            )
            in_channels = num_channels[i]
            modules.append(block)

            if upsample is not None and i > 0:
                modules.append(Upsample(scale_factor=upsample))

        # final conv layer
        final_conv = nn.Conv2d(
            in_channels=num_channels[-1],
            out_channels=self.out_channels,  # RGB + Mask
            kernel_size=3,
            stride=1,
            padding=1
        )
        modules.append(final_conv)

        decoder = nn.Sequential(*modules)
        return decoder

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_slots, slot_dim = slots.shape

        assert slot_dim == self.slot_dim, f"Slot dimension mismatch: {slot_dim} != {self.slot_dim}"

        # adding broadcasting for the disentangled decoder
        slots = slots.reshape((-1, 1, 1, slot_dim))
        slots = slots.repeat(
            (1, self.image_size[0], self.image_size[1], 1)
        )  # slots ~ (B*N_slots, H, W, Slot_dim)

        # adding positional embeddings to reshaped features
        slots = self.positional_encoding(slots)  # slots ~ (B*N_slots, H, W, Slot_dim)
        slots = slots.permute(0, 3, 1, 2)

        y = self.decoder(slots)

        # recons and masks have shapes [B, N_S, C, H, W] & [B, N_S, 1, H, W] respectively
        y_reshaped = y.reshape(batch_size, -1, 3 + 1, y.shape[2], y.shape[3])
        rgb, masks = y_reshaped.split([3, 1], dim=2)

        masks = F.softmax(masks, dim=1)
        return rgb, masks


class Upsample(nn.Module):
    """
    Overriding the upsample class to avoid an error of nn.Upsample with large tensors
    """

    def __init__(self, scale_factor):
        """
        Module initializer
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Forward pass
        """
        y = F.interpolate(x.contiguous(), scale_factor=self.scale_factor, mode='nearest')
        return y

    def __repr__(self):
        """ """
        str = f"Upsample(scale_factor={self.scale_factor})"
        return str