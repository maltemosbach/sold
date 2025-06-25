from datasets.ring_buffer import RingBufferDataset
from torch.utils.data import IterableDataset
from typing import Iterator
import warnings


class NumUpdatesWrapper(IterableDataset):
    def __init__(self, dataset: RingBufferDataset, num_updates: int) -> None:
        self.dataset = dataset
        self.num_updates = num_updates

    def __iter__(self) -> Iterator:
        if self.dataset.is_empty:
            #warnings.warn("Replay buffer is empty. Skipping update.")
            return iter([])

        for _ in range(self.num_updates):
            yield self.dataset.sample()
