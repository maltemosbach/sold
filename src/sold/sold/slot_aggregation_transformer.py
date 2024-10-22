

class SlotAggregationTransformer:
    def __init__(self, token_dim: int, num_heads: int, num_layers: int, hidden_dim: int) -> None:
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim