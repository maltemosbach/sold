"""
Basic building blocks for neural nets
"""

import math
import torch
import torch.nn as nn

from .model_utils import build_grid

__all__ = ["ConvBlock", "ConvTransposeBlock"]


class ConvBlock(nn.Module):
    """
    Simple convolutional block for conv. encoders

    Args:
    -----
    in_channels: int
        Number of channels in the input feature maps.
    out_channels: int
        Number of convolutional kernels in the conv layer
    kernel_size: int
        Size of the kernel for the conv layer
    stride: int
        Amount of strid applied in the convolution
    padding: int/None
        Whether to pad the input feature maps, and how much padding to use.
    batch_norm: bool
        If True, Batch Norm is applied after the convolutional layer
    max_pool: int/tuple/None
        If not None, output feature maps are downsampled by this amount via max pooling
    activation: bool
        If True, output feature maps are activated via a ReLU nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, max_pool=None, activation=True):
        """
        Module initializer
        """
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        # adding conv-(bn)-(pool)-act layer
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if max_pool:
            assert isinstance(max_pool, (int, tuple, list))
            layers.append(nn.MaxPool2d(kernel_size=max_pool, stride=max_pool))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """
        Forward pass
        """
        y = self.block(x)
        return y


class ConvTransposeBlock(nn.Module):
    """
    Simple transposed-convolutional block for conv. decoders

    Args:
    -----
    in_channels: int
        Number of channels in the input feature maps.
    out_channels: int
        Number of convolutional kernels in the conv layer
    kernel_size: int
        Size of the kernel for the conv layer
    stride: int
        Amount of strid applied in the convolution
    padding: int/None
        Whether to pad the input feature maps, and how much padding to use.
    batch_norm: bool
        If True, Batch Norm is applied after the convolutional layer
    upsample: int/tuple/None
        If not None, output feature maps are upsampled by this amount via (nn.) Upsampling
    activation: bool
        If True, output feature maps are activated via a ReLU nonlinearity.
    conv_transpose_2d: bool
        If True, Transposed convolutional layers are used.
        Otherwise, standard convolutions (combined with Upsampling) are applied.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, upsample=None, activation=True, conv_transpose_2d=True):
        """ Module initializer """
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        # adding conv-(bn)-(pool)-act layer
        layers = []
        if conv_transpose_2d:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
            )
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if upsample:
            assert isinstance(upsample, (int, tuple, list))
            layers.append(nn.Upsample(scale_factor=upsample))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """
        Forward pass
        """
        y = self.block(x)
        return y


class SoftPositionEmbed(nn.Module):
    """
    Soft positional embedding with learnable linear projection.
        1. The positional encoding corresponds to a 4-channel grid with coords [-1, ..., 1] and
           [1, ..., -1] in the vertical and horizontal directions
        2. The 4 channels are projected into a hidden_dimension via a linear layer (or Conv-1D)


    Args:
    -----
    hidden_size: int
        Number of output channels
    resolution: list/tuple of integers
        Number of elements in the positional embedding. Corresponds to a spatial size
    vmin, vmax: int
        Minimum and maximum values in the grids. By default vmin=-1 and vmax=1
    """

    def __init__(self, hidden_size, resolution, vmin=-1., vmax=1.):
        """
        Soft positional encoding
        """
        super().__init__()
        self.projection = nn.Conv2d(4, hidden_size, kernel_size=1)
        self.grid = build_grid(resolution, vmin=-1., vmax=1.).permute(0, 3, 1, 2)
        return

    def forward(self, inputs, channels_last=True):
        """
        Projecting grid and adding to inputs
        """
        b_size = inputs.shape[0]
        if self.grid.device != inputs.device:
            self.grid = self.grid.to(inputs.device)
        grid = self.grid.repeat(b_size, 1, 1, 1)
        emb_proj = self.projection(grid)
        if channels_last:
            emb_proj = emb_proj.permute(0, 2, 3, 1)
        return inputs + emb_proj




#
