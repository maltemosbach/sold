"""
Ensures that models can attend to not just the current slots, but also slots from previous time-steps.
"""


from sold.modeling.positional_encoding import build_alibi_mask
import torch
from torch import nn, Tensor
from torch.nn import Module, Linear, Dropout, MultiheadAttention, RMSNorm
from typing import Optional


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
        self.alibi_mask = build_alibi_mask(max_episode_steps, num_heads, num_slots, num_register_tokens)
        # Projection from slot to token dim
        self.slot_projection = nn.Linear(slot_dim, self.token_dim)

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
