"""Positional encodings for transformers operating on sequences of slots.

The transformer models used in this work operate on inputs of shape (batch_size, sequence_length, num_slots, slot_dim).
This module provides positional encodings designed for this input type. They generate encodings of shape
(max_sequence_length, token_dim), which are expanded for the batch dimension and the number of slots
(+ out/register tokens) to retain permutation equivariance of the slots in each time-step.
"""

from abc import ABC
import math
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module, ABC):
    def __init__(self, max_sequence_length: int, token_dim: int) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.token_dim = token_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, num_slots, token_dim = tokens.size()

        if sequence_length > self.max_sequence_length:
            raise ValueError(f"Expected sequence length <= {self.max_sequence_length}, but got {sequence_length}.")

        if token_dim != self.token_dim:
            raise ValueError(f"Expected token_dim to be {self.token_dim}, but got {token_dim}.")

        if self.positional_encoding.size() != (self.max_sequence_length, token_dim):
            raise ValueError(f"Expected positional encoding of shape ({self.max_sequence_length}, {token_dim}), "
                             f"but got {self.positional_encoding.size()}.")

        positional_encoding = self.positional_encoding[:sequence_length].unsqueeze(0).unsqueeze(2)
        positional_encoding = positional_encoding.expand(batch_size, -1, num_slots, -1)
        return tokens + positional_encoding


class NoPositionalEncoding(PositionalEncoding):
    def __init__(self, max_sequence_length: int, token_dim: int) -> None:
        super().__init__(max_sequence_length, token_dim)
        self.register_buffer("positional_encoding", torch.zeros(self.max_sequence_length, self.token_dim))


class LearnedEncoding(PositionalEncoding):
    def __init__(self, max_sequence_length: int, token_dim: int) -> None:
        super().__init__(max_sequence_length, token_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(max_sequence_length, token_dim))
        nn.init.normal_(self.positional_encoding, std=0.02)


class SinusoidalEncoding(PositionalEncoding):
    def __init__(self, max_sequence_length: int, token_dim: int, dropout: float = 0.0) -> None:
        super().__init__(max_sequence_length, token_dim)
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, token_dim, 2) * (-math.log(10000.0) / token_dim))
        sinusoidal_encoding = torch.zeros(max_sequence_length, token_dim)
        sinusoidal_encoding[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", sinusoidal_encoding)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(super().forward(tokens))


class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional encoding to be added to the input tokens of the transformer predictor.

    Our positional encoding only informs about the time-step, i.e., all slots extracted
    from the same input frame share the same positional embedding. This allows our predictor
    model to maintain the permutation equivariance properties.

    Args:
    -----
    batch_size: int
        Number of elements in the batch.
    num_slots: int
        Number of slots extracted per frame. Positional encoding will be repeat for each of these.
    d_model: int
        Dimensionality of the slots/tokens
    dropout: float
        Percentage of dropout to apply after adding the poisitional encoding. Default is 0.1
    max_len: int
        Length of the sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=50):
        """
        Initializing the positional encoding
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # initializing sinusoidal positional embedding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, 1, d_model)
        self.pe = pe
        return

    def forward(self, x, batch_size, num_slots):
        """
        Adding the positional encoding to the input tokens of the transformer

        Args:
        -----
        x: torch Tensor
            Tokens to enhance with positional encoding. Shape is (B, Seq_len, Num_Slots, Token_Dim)
        batch_size: int
            Given batch size to repeat the positional encoding for
        num_slots: int
            Number of slots to repear the positional encoder for
        """

        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        cur_seq_len = x.shape[1]
        cur_pe = self.pe.repeat(batch_size, 1, num_slots, 1)[:, :cur_seq_len]
        x = x + cur_pe
        y = self.dropout(x)
        return y


class TokenWiseSinusoidalPositionalEncoding(SinusoidalPositionalEncoding):
    def forward(self, x, batch_size, num_slots):
        """
        Adding the positional encoding to the input tokens of the transformer

        Args:
        -----
        x: torch Tensor
            Tokens to enhance with positional encoding. Shape is (B, any, Token_Dim)
        batch_size: int
            Given batch size to repeat the positional encoding for
        num_slots: int
            Number of slots to repear the positional encoder for
        """

        visible_slots = x.shape[1]

        # one-liner for missing at current time step.
        missing_at_current_time_step = num_slots - (x.shape[1] % num_slots) if x.shape[1] % num_slots != 0 else 0


        x = torch.cat([x, torch.zeros(x.shape[0], missing_at_current_time_step, x.shape[2]).to(x.device)], dim=1)
        x = x.view(x.shape[0], -1, num_slots, x.shape[2])


        y = super().forward(x, batch_size, num_slots)

        y = y.view(y.shape[0], -1, y.shape[3])
        y = y[:, :visible_slots]
        return y



