import torch
import torch.nn as nn


class Corrector(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int, feature_dim: int, num_iterations: int,
                 num_initial_iterations: int, hidden_dim: int = 128, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.num_initial_iterations = num_initial_iterations
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.scale = feature_dim ** -0.5

        # Normalization layers
        self.norm_input = nn.LayerNorm(feature_dim, eps=0.001)
        self.norm_slot = nn.LayerNorm(slot_dim, eps=0.001)
        self.norm_mlp = nn.LayerNorm(slot_dim, eps=0.001)

        # Embedding layers.
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(slot_dim, slot_dim)
        self.to_v = nn.Linear(slot_dim, slot_dim)

        # Slot update functions.
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        return

    def forward(self, image_features: torch.Tensor, predicted_slots: torch.Tensor, step=0, **kwargs):
        """Apply slot attention on image features.

        Args:
            image_features (torch.Tensor): Image feature vectors extracted by the encoder.
            predicted_slots (torch.Tensor): Predicted slots from the predictor or initializer.

        Returns:
            torch.Tensor: Slot representation after applying the attention mechanism.
        """
        batch_size = image_features.shape[0]
        self.attention_masks = None

        image_features = self.norm_input(image_features)
        k, v = self.to_k(image_features), self.to_v(image_features)

        # iterative refinement of the slot representation
        num_iters = self.num_initial_iterations if step == 0 else self.num_iterations
        for _ in range(num_iters):
            slots_prev = predicted_slots
            slots = self.norm_slot(predicted_slots)
            q = self.to_q(slots)

            # q ~ (B, N_Slots, Slot_dim)
            # k, v ~ (B, N_locs, Slot_dim)
            # attention equation [softmax(Q K^T) V]
            dots = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale  # dots ~ (B, N_slots, N_locs)
            attn = dots.softmax(dim=1) + self.epsilon  # enforcing competition between slots
            attn = attn / attn.sum(dim=-1, keepdim=True)  # attn ~ (B, N_slots, N_locs)
            self.attention_masks = attn
            updates = torch.einsum('b i d , b d j -> b i j', attn, v)  # updates ~ (B, N_slots, slot_dim)
            # further refinement
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            )
            slots = slots.reshape(batch_size, -1, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

    def get_attention_masks(self):
        """Fetches last computed attention masks."""
        B, N_slots, N_locs = self.attention_masks.shape
        masks = self.attention_masks
        return masks
