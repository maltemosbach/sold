from abc import ABC, abstractmethod
from math import sqrt
import torch
import torch.nn as nn


class SlotInitializer(nn.Module, ABC):
    def __init__(self, num_slots: int, slot_dim: int) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

    @abstractmethod
    def forward(self, batch_size: int) -> torch.Tensor:
        """Return slot initializations."""
        pass


class LearnedRandom(SlotInitializer):
    """
    Learned random initialization. This is the default mode used in SlotAttention.
    Slots are randomly sampled from a Gaussian distribution. However, the statistics of this
    distribution (mean vector and diagonal of covariance) are learned via backpropagation
    """

    def __init__(self, num_slots: int, slot_dim: int) -> None:
        super().__init__(num_slots, slot_dim)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        with torch.no_grad():
            limit = sqrt(6.0 / (1 + slot_dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_sigma, -limit, limit)
        return

    def forward(self, batch_size: int) -> torch.Tensor:
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_sigma.expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=self.slots_mu.device)
        return slots


class Learned(SlotInitializer):
    """
    Learned initialization.
    For each slot a discrete initialization is learned via backpropagation.
    """

    def __init__(self, num_slots: int, slot_dim: int) -> None:
        super().__init__(num_slots, slot_dim)

        self.initial_slots = torch.nn.Parameter(torch.randn(1, self.num_slots, self.slot_dim))

        with torch.no_grad():
            limit = sqrt(6.0 / (1 + slot_dim))
            torch.nn.init.uniform_(self.initial_slots, -limit, limit)
        return

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.initial_slots.expand(batch_size, -1, -1)
