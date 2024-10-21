from abc import ABC, abstractmethod
from sold.utils.transformer import TransformerBlock
import torch
import torch.nn as nn


class Predictor(nn.Module, ABC):
    def __init__(self, slot_dim: int) -> None:
        super().__init__()
        self.slot_dim = slot_dim

    @abstractmethod
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        pass


class TransformerPredictor(Predictor):
    def __init__(self, slot_dim: int, num_heads: int = 4, mlp_size: int = 256) -> None:
        super().__init__(slot_dim)
        self.transformer = TransformerBlock(slot_dim, num_heads, mlp_size)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.transformer(slots)
