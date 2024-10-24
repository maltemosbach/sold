"""
Ensures that models can attend to not just the current slots, but also slots from previous time-steps.
"""


import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import Module, Linear, Dropout, MultiheadAttention

from object_centric_control.models.utils import RMSNorm


def fill_with_neg_inf(t):
  """FP16-compatible function that fills a tensor with -inf."""
  return t.float().fill_(float("-inf")).type_as(t)


class SlotMemoryAttention(nn.Module):
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, token_dim: int, num_heads: int,
                 num_layers: int, hidden_dim: int, num_register_tokens: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_register_tokens = num_register_tokens
        self.token_dim = token_dim
        self.num_heads = num_heads

        self.transformer_encoder_blocks = nn.Sequential(
            *[TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=self.num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
            ) for _ in range(num_layers)]
        )
        # Add cls and register tokens
        self.output_token = nn.Parameter(torch.Tensor(1, 1, 1 + self.num_register_tokens, self.token_dim))
        nn.init.xavier_uniform_(self.output_token)
        # Alibi
        self.build_alibi_mask(max_episode_steps)
        # Projection from slot to token dim
        self.slot_projection = nn.Linear(slot_dim, self.token_dim)

    def build_alibi_mask(self, max_steps: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2 ** (-2 ** -(math.log2(n) - 2)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n)  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2 ** math.floor(
                    math.log2(n))  # when the number of heads is not a power of 2, we use this workaround.
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                                   :n - closest_power_of_2]

        slopes = torch.Tensor(get_slopes(self.num_heads))
        # In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper).
        # If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
        # This works because the softmax operation is invariant to translation, and our bias functions are always linear.
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_steps).unsqueeze(0).unsqueeze(0).expand(self.num_heads, -1, -1)
        self.alibi_mask = torch.triu(fill_with_neg_inf(torch.zeros([max_steps, max_steps])), 1)
        self.alibi_mask = self.alibi_mask.unsqueeze(0) + alibi
        self.alibi_mask = torch.repeat_interleave(self.alibi_mask, self.num_slots + 1 + self.num_register_tokens, dim=2)
        self.alibi_mask = torch.repeat_interleave(self.alibi_mask, self.num_slots + 1 + self.num_register_tokens, dim=1)

        # mask out cls and register tokens from prior time steps
        mask_shape = self.alibi_mask.shape[1:]
        i_indices = torch.arange(mask_shape[0])
        j_indices = torch.arange(mask_shape[1])
        i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')
        # select grid cells where we are in prior time steps
        condition1 = (i_grid // (self.num_slots + 1 + self.num_register_tokens)) > (j_grid // (self.num_slots + 1 + self.num_register_tokens))
        # select grid cells of cls and register tokens
        condition2 = (j_grid % (self.num_slots + 1 + self.num_register_tokens)) >= self.num_slots

        self.alibi_mask[:, condition1 & condition2] = float("-inf")

    def generate_mask(self, batch_size, num_imgs, device):
        if device != self.alibi_mask.device:
            self.alibi_mask = self.alibi_mask.to(device)
        mask = self.alibi_mask[:, -num_imgs * (self.num_slots + 1 + self.num_register_tokens):, -num_imgs * (self.num_slots + 1 + self.num_register_tokens):]
        batch_mask = mask.repeat(batch_size, 1, 1)
        return batch_mask

    def forward(self, slots, start=0):
        batch_size, sequence_length, num_slots, slot_dim = slots.shape
        # project slots to token dim
        slots = self.slot_projection(slots)

        # expand the reward token to the full batch and concatenate to slots
        output = self.output_token.repeat(batch_size, sequence_length, 1, 1)
        output = torch.cat((slots, output), dim=2)

        # generate mask
        mask = self.generate_mask(batch_size, sequence_length, slots.device)

        # feeding through transformer blocks
        output = output.reshape(batch_size, -1, self.token_dim)
        for transformer_encoder_block in self.transformer_encoder_blocks:
            output = transformer_encoder_block(output, mask)

        output = output.reshape(batch_size, sequence_length, -1, self.token_dim)[:, -(sequence_length-start):, -1]
        return output


class TransformerEncoderLayer(Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = nn.SiLU()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
