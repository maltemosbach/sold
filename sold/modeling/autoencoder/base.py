"""Base class defining the interface for all (object-centric and holistic) autoencoders."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Optional
from utils.visualizations import make_row, stack_rows


class Autoencoder(ABC, nn.Module):
    @property
    @abstractmethod
    def num_slots(self) -> int:
        pass

    @property
    @abstractmethod
    def slot_dim(self) -> int:
        pass

    @abstractmethod
    def encode(self, images: torch.Tensor, actions: torch.Tensor,
               prior_slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode images into slots.
        Args:
            images (torch.Tensor): Image sequence of shape (batch_size, sequence_length, 3, height, width).
            actions (torch.Tensor): Action sequence of shape (batch_size, sequence_length - 1, action_dim).
            prior_slots (torch.Tensor, optional): Initializations of slots are either queried from the initializer or
                predicted from the prior_slots (and actions), before slot attention is applied.

        Returns:
            torch.Tensor: Slots encoded at every time step of shape (batch_size, sequence_length, num_slots, slot_dim)
        """
        pass

    @abstractmethod
    def decode(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode slots into images.
        Args:
            slots (torch.Tensor): Slots of shape (batch_size, sequence_length, num_slots, slot_dim).

        Returns:
            dict(str, torch.Tensor): Dictionary containing the reconstructed images and optionally reconstructions of
            individual objects.
        """
        pass

    def forward(self, images: torch.Tensor, actions: torch.Tensor, prior_slots: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length, num_channels, height, width = images.size()
        slots = self.encode(images, actions, prior_slots)
        outputs = self.decode(slots)
        if "reconstructions" not in outputs:
            raise ValueError("Expected 'reconstructions' to be present in decode outputs.")
        return {**outputs, "slots": slots}

    def visualize_reconstruction(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        rows = []
        if "images" in outputs:
            rows.append(make_row(outputs["images"].cpu()))
        rows.append(make_row(outputs["reconstructions"].cpu()))
        return stack_rows(rows)
