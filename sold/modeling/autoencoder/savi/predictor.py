from abc import ABC, abstractmethod
import torch
import torch.nn as nn


@torch.no_grad()
def init_xavier_(model: nn.Module):
    """
    Initializes (in-place) a model's weights with xavier uniform, and its biases to zero.
    All parameters with name containing "bias" are initialized to zero.
    All other parameters are initialized with xavier uniform with default parameters,
    unless they have dimensionality <= 1.
    """
    for name, tensor in model.named_parameters():
        if name.endswith(".bias"):
            tensor.zero_()
        elif len(tensor.shape) <= 1:
            pass  # silent
        else:
            torch.nn.init.xavier_uniform_(tensor)


class MetaAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=1, dropout=0., out_dim=None, **kwargs):
        assert num_heads >= 1
        assert emb_dim % num_heads == 0, "Embedding dim. must be divisible by number of heads..."
        super().__init__()

        out_dim = out_dim if out_dim is not None else emb_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

        self.out_projection = nn.Sequential(
                nn.Linear(emb_dim, out_dim, bias=False)
            )
        self.attention_masks = None
        return

    def attention(self, query, key, value, dim_head):
        scale = dim_head ** -0.5  # 1/sqrt(d_k)
        dots = torch.einsum('b i d , b j d -> b i j', query, key) * scale  # Q * K.T / sqrt(d_k)
        attention = dots.softmax(dim=-1)
        self.attention_masks = attention
        attention = self.drop(attention)
        vect = torch.einsum('b i d , b d j -> b i j', attention, value)  # Att * V
        return vect

    def get_attention_masks(self, reshape=None):
        assert self.attention_masks is not None, "Attention masks have not yet been computed..."
        masks = self.attention_masks
        return masks


class MultiHeadSelfAttention(MetaAttention):
    def __init__(self, emb_dim, num_heads=8, dropout=0.):
        super().__init__(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        return

    def forward(self, x, **kwargs):
        batch_size, num_tokens, token_dim = x.size()
        dim_head = token_dim // self.num_heads

        # linear projections
        q, k, v = self.q(x), self.k(x), self.v(x)

        # split into heads and move to batch-size side:
        # (Batch, Token, Dims) --> (Batch, Heads, Token, HeadDim) --> (Batch* Heads, Token, HeadDim)
        q = q.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        q = q.reshape(batch_size * self.num_heads, num_tokens, dim_head)
        k = k.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        k = k.reshape(batch_size * self.num_heads, num_tokens, dim_head)
        v = v.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        v = v.reshape(batch_size * self.num_heads, num_tokens, dim_head)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v, dim_head=dim_head)
        # rearranging heads and recovering original shape
        vect = vect.reshape(batch_size, self.num_heads, num_tokens, dim_head).transpose(1, 2)
        vect = vect.reshape(batch_size * num_tokens, self.num_heads * dim_head)

        y = self.out_projection(vect)
        y = y.reshape(batch_size, num_tokens, self.num_heads * dim_head)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_size, pre_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        assert num_heads >= 1

        self.attn = MultiHeadSelfAttention(
            emb_dim=embed_dim,
            num_heads=num_heads,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_dim),
        )
        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_model()

    @torch.no_grad()
    def _init_model(self):
        init_xavier_(self)

    def forward(self, inputs):
        assert inputs.ndim == 3
        B, L, _ = inputs.shape

        if self.pre_norm:
            x = self.layernorm_query(inputs)
            x = self.attn(x)
            x = x + inputs

            y = x

            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            z = z + y
        else:
            x = self.attn(inputs)
            x = x + inputs
            x = self.layernorm_query(x)

            y = x

            z = self.mlp(y)
            z = z + y
            z = self.layernorm_mlp(z)
        return z


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
