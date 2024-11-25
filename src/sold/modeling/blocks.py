import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Simple convolutional block for conv. encoders

    Args:
    -----
    in_channels: int
        Number of channels in the input feature maps.
    out_channels: int
        Number of convolutional kernels in the conv layer
    kernel_size: int
        Size of the kernel for the conv layer
    stride: int
        Amount of strid applied in the convolution
    padding: int/None
        Whether to pad the input feature maps, and how much padding to use.
    batch_norm: bool
        If True, Batch Norm is applied after the convolutional layer
    max_pool: int/tuple/None
        If not None, output feature maps are downsampled by this amount via max pooling
    activation: bool
        If True, output feature maps are activated via a ReLU nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, max_pool=None, activation=True):
        """
        Module initializer
        """
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        # adding conv-(bn)-(pool)-act layer
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if max_pool:
            assert isinstance(max_pool, (int, tuple, list))
            layers.append(nn.MaxPool2d(kernel_size=max_pool, stride=max_pool))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """
        Forward pass
        """
        y = self.block(x)
        return y


class ConvTransposeBlock(nn.Module):
    """
    Simple transposed-convolutional block for conv. decoders

    Args:
    -----
    in_channels: int
        Number of channels in the input feature maps.
    out_channels: int
        Number of convolutional kernels in the conv layer
    kernel_size: int
        Size of the kernel for the conv layer
    stride: int
        Amount of strid applied in the convolution
    padding: int/None
        Whether to pad the input feature maps, and how much padding to use.
    batch_norm: bool
        If True, Batch Norm is applied after the convolutional layer
    upsample: int/tuple/None
        If not None, output feature maps are upsampled by this amount via (nn.) Upsampling
    activation: bool
        If True, output feature maps are activated via a ReLU nonlinearity.
    conv_transpose_2d: bool
        If True, Transposed convolutional layers are used.
        Otherwise, standard convolutions (combined with Upsampling) are applied.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, upsample=None, activation=True, conv_transpose_2d=True):
        """ Module initializer """
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        # adding conv-(bn)-(pool)-act layer
        layers = []
        if conv_transpose_2d:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
            )
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if upsample:
            assert isinstance(upsample, (int, tuple, list))
            layers.append(nn.Upsample(scale_factor=upsample))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """
        Forward pass
        """
        y = self.block(x)
        return y


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
    """
    MetaClass for (Multi-Head) Key-Value Attention Mechanisms

    Args:
    -----
    emb_dim: integer
        Dimensionality of the token embeddings.
    num_heads: integer
        Number of heads accross which we compute attention.
        Head_dim = Emb_dim / Num_Heads. Division must be exact!
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    out_dim: int/None
        Dimensionality of the output embeddings. If not given, it is set to 'emb_dim'
    """

    def __init__(self, emb_dim, num_heads=1, dropout=0., out_dim=None, **kwargs):
        """
        Initializer of the attention block
        """
        assert num_heads >= 1
        assert emb_dim % num_heads == 0, "Embedding dim. must be divisible by number of heads..."
        super().__init__()

        out_dim = out_dim if out_dim is not None else emb_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        # computing query, key, value for all embedding heads
        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

        # output projection
        self.out_projection = nn.Sequential(
                nn.Linear(emb_dim, out_dim, bias=False)
            )
        self.attention_masks = None
        return

    def forward(self, x):
        """ """
        raise NotImplementedError("Base-Class does not implement a 'forward' method...")

    def attention(self, query, key, value, dim_head):
        """
        Implementation of the standard normalized key-value attention equation
        """
        scale = dim_head ** -0.5  # 1/sqrt(d_k)
        dots = torch.einsum('b i d , b j d -> b i j', query, key) * scale  # Q * K.T / sqrt(d_k)
        attention = dots.softmax(dim=-1)
        self.attention_masks = attention
        attention = self.drop(attention)
        vect = torch.einsum('b i d , b d j -> b i j', attention, value)  # Att * V
        return vect

    def get_attention_masks(self, reshape=None):
        """
        Fetching last computer attention masks
        """
        assert self.attention_masks is not None, "Attention masks have not yet been computed..."
        masks = self.attention_masks
        return masks


class MultiHeadSelfAttention(MetaAttention):
    """
    Vanilla Multi-Head dot-product attention mechanism.

    Args:
    -----
    emb_dim: integer
        Dimensionality of the token embeddings.
    num_heads: integer
        Number of heads accross which we compute attention.
        Head_dim = Emb_dim / Num_Heads. Division must be exact!
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    """

    def __init__(self, emb_dim, num_heads=8, dropout=0.):
        """
        Initializer of the attention block
        """
        super().__init__(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        return

    def forward(self, x, **kwargs):
        """
        Forward pass through multi-head attention
        """
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
        # applying attention equation
        vect = self.attention(query=q, key=k, value=v, dim_head=dim_head)
        # rearranging heads and recovering original shape
        vect = vect.reshape(batch_size, self.num_heads, num_tokens, dim_head).transpose(1, 2)
        vect = vect.reshape(batch_size * num_tokens, self.num_heads * dim_head)

        # output projection
        y = self.out_projection(vect)
        y = y.reshape(batch_size, num_tokens, self.num_heads * dim_head)
        return y


class TransformerBlock(nn.Module):
    """
    Tranformer encoder block.
    This is used as predictor module in SAVi.

    Args:
    -----
    embed_dim: int
        Dimensionality of the input embeddings
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    pre_norm: bool
        If True, transformer computes the LayerNorm before attention and MLP.
        Otherwise, LayerNorm is used after the aforementaitoned layers
    """

    def __init__(self, embed_dim, num_heads, mlp_size, pre_norm=False):
        """
        Module initializer
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        assert num_heads >= 1

        # MHA
        self.attn = MultiHeadSelfAttention(
            emb_dim=embed_dim,
            num_heads=num_heads,
        )
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_dim),
        )
        # LayerNorms
        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_model()

    @torch.no_grad()
    def _init_model(self):
        """ Parameter initialization """
        init_xavier_(self)

    def forward(self, inputs):
        """
        Forward pass through transformer encoder block
        """
        assert inputs.ndim == 3
        B, L, _ = inputs.shape

        if self.pre_norm:
            # Self-attention.
            x = self.layernorm_query(inputs)
            x = self.attn(x)
            x = x + inputs

            y = x

            # MLP
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            z = z + y
        else:
            # Self-attention on queries.
            x = self.attn(inputs)
            x = x + inputs
            x = self.layernorm_query(x)

            y = x

            # MLP
            z = self.mlp(y)
            z = z + y
            z = self.layernorm_mlp(z)
        return z

