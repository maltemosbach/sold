import torch
import torch.distributions as D
import torch.nn as nn
from .slot_memory_attention import SlotMemoryAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Predictor(nn.Module):
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, token_dim: int, num_heads: int,
                 num_layers: int, hidden_dim: int, output_dim: int, num_register_tokens: int = 0, num_mlp_layers: int = 1,) -> None:
        """Used to predict single quantities like rewards or actions from the set and history of slots."""
        super().__init__()
        self.slot_memory_attention = SlotMemoryAttention(max_episode_steps, num_slots, slot_dim, token_dim, num_heads,
                                                         num_layers, hidden_dim, num_register_tokens)
        self.mlp = []
        for layer_num in range(num_mlp_layers):
            if layer_num == 0:
                self.mlp.append(nn.Linear(token_dim, hidden_dim))
            else:
                self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(RMSNorm(hidden_dim))
            self.mlp.append(nn.SiLU())
        self.mlp.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, slots: torch.Tensor, start: int = 0) -> torch.Tensor:
        features = self.slot_memory_attention(slots, start=start)
        return self.mlp(features)


class GaussianPredictor(Predictor):
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, token_dim: int, num_heads: int,
                 num_layers: int, hidden_dim: int, output_dim: int, num_register_tokens: int = 0, num_mlp_layers: int = 1,
                 lower_bound = None, upper_bound = None) -> None:
        super().__init__(max_episode_steps, num_slots, slot_dim, token_dim, num_heads, num_layers, hidden_dim,
                         output_dim=2*output_dim, num_register_tokens=num_register_tokens, num_mlp_layers=num_mlp_layers)
        self.max_std, self.min_std, self.init_std = 1.0, 0.1, 2.0
        self.lower_bound = torch.tensor(lower_bound)
        self.upper_bound = torch.tensor(upper_bound)

    def forward(self, slots: torch.Tensor, start: int = 0) -> D.Distribution:
        x = super().forward(slots, start=start)
        mean_, std_ = x.chunk(2, -1)
        mean_ = torch.clamp(mean_, self.lower_bound.to(mean_.device), self.upper_bound.to(mean_.device))
        std = (self.max_std - self.min_std) * torch.sigmoid(std_ + self.init_std) + self.min_std
        dist = D.Normal(torch.tanh(mean_), std, validate_args=False)
        return D.Independent(dist, 1, validate_args=False)


class TwoHotPredictor(Predictor):
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, token_dim: int, num_heads: int,
                 num_layers: int, hidden_dim: int, num_register_tokens: int = 0, num_mlp_layers: int = 1) -> None:
        """Predict over 255 exponentially-spaced bins to represent scalar values like rewards."""
        super().__init__(max_episode_steps, num_slots, slot_dim, token_dim, num_heads, num_layers, hidden_dim,
                         output_dim=255, num_register_tokens=num_register_tokens, num_mlp_layers=num_mlp_layers)
