from modeling.autoencoder.cnn.encoder import CnnEncoder
from modeling.autoencoder.savi.encoder import ImagePositionalEmbedding
import torch
import torch.nn.functional as F
from typing import List, Tuple


class SaviCnnDecoder(CnnEncoder):
    """CNN Decoder proposed by Slot Attention (https://arxiv.org/pdf/2006.15055, Section E.3)."""
    def __init__(self, image_size: Tuple[int, int],  num_channels: List[int], kernel_sizes: List[int],
                 strides: List[int], in_channels: int) -> None:
        super().__init__(num_channels + [4], kernel_sizes + [3,], strides + [1,], in_channels)  # Add final convolutional layer projecting to RGB + mask.
        self.image_size = image_size
        self.slot_dim = in_channels
        self.positional_embedding = ImagePositionalEmbedding(feature_dim=in_channels, image_size=image_size)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_slots, slot_dim = slots.size()
        if slot_dim != self.slot_dim:
            raise ValueError(f"Expected slot dimension to be {self.slot_dim}, but got {slot_dim}")

        slots = slots.reshape(-1, 1, 1, slot_dim).expand(-1, *self.image_size, -1)  # Spatial broadcasting -> Shape: (B * num_slots, H, W, slot_dim)
        slots = self.positional_embedding(slots)
        slots = slots.permute(0, 3, 1, 2)  # Shape: (B*N_slots, slot_dim, H, W)
        decoded = super().forward(slots).reshape(batch_size, -1, 4, *self.image_size)  # Shape: (B, num_slots, 4, H, W).
        rgbs, mask_logits = decoded.split([3, 1], dim=2)
        masks = F.softmax(mask_logits, dim=1)
        return torch.clamp(rgbs, 0, 1), masks
