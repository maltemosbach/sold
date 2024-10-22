from sold.sold.slot_aggregation_transformer import SlotAggregationTransformer


class RewardPredictor(SlotAggregationTransformer):
    def __init__(self, token_dim: int, num_heads: int, num_layers: int, hidden_dim: int) -> None:
        super().__init__(token_dim, num_heads, num_layers, hidden_dim)