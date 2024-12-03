import math
import numpy as np
import torch
import torch.nn as nn


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



def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def build_alibi_mask(max_steps: int, num_heads: int, num_slots: int, num_register_tokens: int) -> torch.Tensor:
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

    slopes = torch.Tensor(get_slopes(num_heads))
    # In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper).
    # If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
    # This works because the softmax operation is invariant to translation, and our bias functions are always linear.
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_steps).unsqueeze(0).unsqueeze(0).expand(num_heads, -1, -1)
    alibi_mask = torch.triu(fill_with_neg_inf(torch.zeros([max_steps, max_steps])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    alibi_mask = torch.repeat_interleave(alibi_mask, num_slots + 1 + num_register_tokens, dim=2)
    alibi_mask = torch.repeat_interleave(alibi_mask, num_slots + 1 + num_register_tokens, dim=1)

    # mask out cls and register tokens from prior time steps
    mask_shape = alibi_mask.shape[1:]
    i_indices = torch.arange(mask_shape[0])
    j_indices = torch.arange(mask_shape[1])
    i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')
    # select grid cells where we are in prior time steps
    condition1 = (i_grid // (num_slots + 1 + num_register_tokens)) > (j_grid // (num_slots + 1 + num_register_tokens))
    # select grid cells of cls and register tokens
    condition2 = (j_grid % (num_slots + 1 + num_register_tokens)) >= num_slots

    alibi_mask[:, condition1 & condition2] = float("-inf")
    return alibi_mask


class SoftPositionEmbed(nn.Module):
    """
    Soft positional embedding with learnable linear projection.
        1. The positional encoding corresponds to a 4-channel grid with coords [-1, ..., 1] and
           [1, ..., -1] in the vertical and horizontal directions
        2. The 4 channels are projected into a hidden_dimension via a linear layer (or Conv-1D)


    Args:
    -----
    hidden_size: int
        Number of output channels
    resolution: list/tuple of integers
        Number of elements in the positional embedding. Corresponds to a spatial size
    vmin, vmax: int
        Minimum and maximum values in the grids. By default vmin=-1 and vmax=1
    """

    def __init__(self, hidden_size, resolution, vmin=-1., vmax=1.):
        """
        Soft positional encoding
        """
        super().__init__()
        self.projection = nn.Conv2d(4, hidden_size, kernel_size=1)
        self.grid = build_grid(resolution, vmin=-1., vmax=1.).permute(0, 3, 1, 2)
        return

    def forward(self, inputs, channels_last=True):
        """
        Projecting grid and adding to inputs
        """
        b_size = inputs.shape[0]
        if self.grid.device != inputs.device:
            self.grid = self.grid.to(inputs.device)
        grid = self.grid.repeat(b_size, 1, 1, 1)
        emb_proj = self.projection(grid)
        if channels_last:
            emb_proj = emb_proj.permute(0, 2, 3, 1)
        return inputs + emb_proj


def build_grid(resolution, vmin=-1., vmax=1., device=None):
    """
    Building four grids with gradients [0,1] in directios (x,-x,y,-y)
    This can be used as a positional encoding.

    Args:
    -----
    resolution: list/tuple of integers
        number of elements in each of the gradients

    Returns:
    -------
    torch_grid: torch Tensor
        Grid gradients in 4 directions. Shape is [R, R, 4]
    """
    ranges = [np.linspace(vmin, vmax, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    torch_grid = torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)
    return torch_grid
