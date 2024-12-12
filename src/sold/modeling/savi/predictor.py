from abc import ABC, abstractmethod
from modeling.blocks import TransformerBlock
import torch
import torch.nn as nn


class Predictor(nn.Module, ABC):
    def __init__(self, slot_dim: int, action_dim: int) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.action_dim = action_dim

    @abstractmethod
    def forward(self, slots: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict the slots/slot initializations at the next time-step.

        Args:
            slots (torch.Tensor): Slots at the current time-step of shape (B, num_slots, slot_dim).
            actions (torch.Tensor): Actions at the current time-step of shape (B, action_dim).

        Returns:
            torch.Tensor: Predicted slots at the next time-step of shape (B, num_slots, slot_dim).
        """
        pass


class TransformerPredictor(Predictor):
    def __init__(self, slot_dim: int, action_dim: int, num_heads: int = 4, mlp_size: int = 256) -> None:
        super().__init__(slot_dim, action_dim)
        self.transformer = TransformerBlock(slot_dim, num_heads, mlp_size)

    def forward(self, slots: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.transformer(slots)


class ActionConditionalTransformerPredictor(TransformerPredictor):
    def __init__(self, slot_dim: int, action_dim: int, num_head: int = 4, mlp_size: int = 256) -> None:
        super().__init__(slot_dim, action_dim, num_head, mlp_size)
        self.action_embedding = nn.Linear(action_dim, slot_dim)

    def forward(self, slots: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_tokens = self.action_embedding(actions).unsqueeze(1)
        tokens = torch.cat((slots, action_tokens), dim=1)
        predicted_slots = self.transformer(tokens)[:, :-1]
        return predicted_slots
