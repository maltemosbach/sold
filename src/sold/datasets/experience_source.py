import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Callable, Iterator


class DummyValidationDataset(Dataset):
    def __init__(self, num_validation_episodes: int) -> None:
        self.num_validation_episodes = num_validation_episodes

    def __len__(self):
        return self.num_validation_episodes

    def __getitem__(self, idx):
        return torch.randn(10)


class ExperienceSourceDataset(IterableDataset):
    def __init__(self, sample_batch: Callable, collect_interval: int) -> None:
        self._sample_batch = sample_batch
        self.collect_interval = collect_interval  # Number of times to sample a batch before collecting new data.

    def __iter__(self) -> Iterator:
        for _ in range(self.collect_interval):
            images, actions, rewards, is_firsts = self._sample_batch()

            def return_batch(i):
                return images[i], actions[i], rewards[i], is_firsts[i]

            for i in range(len(images)):
                yield return_batch(i)
