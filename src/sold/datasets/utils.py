from sold.datasets.ring_buffer import RingBufferDataset
from torch.utils.data import IterableDataset
from typing import Iterator
import warnings


class NumUpdatesWrapper(IterableDataset):
    def __init__(self, dataset: RingBufferDataset, num_updates: int) -> None:
        self.dataset = dataset
        self.num_updates = num_updates

    def __iter__(self) -> Iterator:
        if self.dataset.is_empty:
            warnings.warn("Replay buffer is empty. Skipping update.")
            return iter([])

        for _ in range(self.num_updates):
            sequence_batch = self.dataset.sample()

            def return_batch(i):
                return {k: v[i] for k, v in sequence_batch.items()}

            for i in range(len(next(iter(sequence_batch.values())))):
                yield return_batch(i)
