from sold.sold.slot_aggregation_transformer import SlotAggregationTransformer


class Actor(SlotAggregationTransformer):
    def __init__(self, action_dim: int, token_dim: int, num_heads: int, num_layers: int, hidden_dim: int) -> None:
        super().__init__(token_dim, num_heads, num_layers, hidden_dim)
        self.action_dim = action_dim