from modeling.attention_mask import AttentionMask
from modeling.positional_encoding import PositionalEncoding
import torch
from torch import nn, Tensor
from torch.nn import Module, Linear, Dropout, MultiheadAttention, RMSNorm
from typing import Optional


class SlotAggregationTransformer(nn.Module):
    def __init__(self, attention_mask: AttentionMask, positional_encoding: PositionalEncoding, max_episode_steps: int,
                 num_slots: int, slot_dim: int, token_dim: int, num_heads: int, num_layers: int, hidden_dim: int,
                 num_register_tokens: int = 0, dropout: float = 0.0) -> None:
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
        self.out_and_register_tokens = nn.Parameter(torch.Tensor(1, 1, 1 + self.num_register_tokens, self.token_dim))
        nn.init.xavier_uniform_(self.out_and_register_tokens)

        self.attention_mask = attention_mask
        self.positional_encoding = positional_encoding
        self.slot_projection = nn.Linear(slot_dim, self.token_dim)

    def forward(self, slots: torch.Tensor, start: int = 0) -> torch.Tensor:
        batch_size, sequence_length, num_slots, slot_dim = slots.shape
        slot_tokens = self.slot_projection(slots)  # Project slots to token_dim.

        # Add the output and register tokens to the slots (at every batch and time step).
        out_and_register_tokens = self.out_and_register_tokens.repeat(batch_size, sequence_length, 1, 1)
        tokens = torch.cat((slot_tokens, out_and_register_tokens), dim=2)

        time_encoded_tokens = self.positional_encoding(tokens)

        output_tokens = time_encoded_tokens.reshape(batch_size, -1, self.token_dim)
        for transformer_encoder_block in self.transformer_encoder_blocks:
            output_tokens = transformer_encoder_block(output_tokens, self.attention_mask(batch_size, sequence_length))

        output_tokens = output_tokens.reshape(batch_size, sequence_length, -1, self.token_dim)[:, -(sequence_length-start):, -1]
        return output_tokens


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
