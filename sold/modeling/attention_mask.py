"""Attention masks for transformers operating on sequences of slots.

The transformer models used in this work operate on inputs of shape (batch_size, sequence_length, num_slots, slot_dim),
which are reshaped to (batch_size, sequence_length * num_slots, slot_dim) before being fed into the transformer.
The attention masks in this module abstract the logic of masking out future tokens in the sequence.
"""

from abc import ABC
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


class AttentionMask(ABC, nn.Module):
    def __init__(self, max_sequence_length: int, num_heads: int, num_slots: int, num_register_tokens: int = 0) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.num_register_tokens = num_register_tokens
        self.step_size = num_slots + 1 + num_register_tokens  # Size of each time-step (slots + cls + register)
        self.max_num_tokens = max_sequence_length * self.step_size

    def forward(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        if sequence_length > self.max_sequence_length:
            raise ValueError(f"Expected sequence length <= {self.max_sequence_length}, but got {sequence_length}.")

        if self._full_mask.size() != (self.num_heads, self.max_num_tokens, self.max_num_tokens):
            raise ValueError(f"Expected mask of size ({self.num_heads}, {self.max_num_tokens}, {self.max_num_tokens}), "
                             f"but got {self.mask.size()}.")

        return self._full_mask[:, -sequence_length * self.step_size:, -sequence_length * self.step_size:].repeat(
            batch_size, 1, 1)

    def show(self, head: int = 0) -> None:
        fig, ax = plt.subplots()
        ax.matshow(self._full_mask[head].cpu().numpy(), cmap='plasma')
        ax.set_xticks(np.arange(self.max_num_tokens + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.max_num_tokens + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5)
        plt.title(f"{type(self).__name__} (head={0})")
        plt.show()


class CausalMask(AttentionMask):
    def __init__(self, max_sequence_length: int, num_heads: int, num_slots: int, num_register_tokens: int = 0) -> None:
        super().__init__(max_sequence_length, num_heads, num_slots, num_register_tokens)
        causal_mask = torch.triu(fill_with_neg_inf(torch.zeros([max_sequence_length, max_sequence_length])), 1)
        causal_mask = torch.repeat_interleave(causal_mask, num_slots + 1 + num_register_tokens, dim=1)
        causal_mask = torch.repeat_interleave(causal_mask, num_slots + 1 + num_register_tokens, dim=0)
        self.register_buffer("_full_mask", causal_mask.unsqueeze(0).repeat(num_heads, 1, 1))


class AlibiMask(CausalMask):
    def __init__(self, max_sequence_length: int, num_heads: int, num_slots: int, num_register_tokens: int = 0) -> None:
        super().__init__(max_sequence_length, num_heads, num_slots, num_register_tokens)

        mask = torch.triu(fill_with_neg_inf(torch.zeros([max_sequence_length, max_sequence_length])), 1)
        slopes = torch.Tensor(self.get_slopes(num_heads))
        slopes = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_sequence_length).unsqueeze(0).unsqueeze(0).expand(
            num_heads, -1, -1)
        mask = mask.unsqueeze(0) + slopes
        mask = torch.repeat_interleave(mask, num_slots + 1 + num_register_tokens, dim=2)
        mask = torch.repeat_interleave(mask, num_slots + 1 + num_register_tokens, dim=1)

        # mask out cls and register tokens from prior time steps
        i_indices = torch.arange(mask.shape[1])
        j_indices = torch.arange(mask.shape[2])
        i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')
        # select grid cells where we are in prior time steps
        is_prior = (i_grid // (num_slots + 1 + num_register_tokens)) > (j_grid // (num_slots + 1 + num_register_tokens))
        # select grid cells of cls and register tokens
        is_cls_or_register = (j_grid % (num_slots + 1 + num_register_tokens)) >= num_slots
        mask[:, is_prior & is_cls_or_register] = float("-inf")
        self.register_buffer("_full_mask", mask)

    def get_slopes(self, n: int):
        def get_slopes_power_of_2(n: int):
            start = (2 ** (-2 ** -(math.log2(n) - 2)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + self.get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
