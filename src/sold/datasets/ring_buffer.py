from abc import ABC, abstractmethod
from collections import defaultdict, deque
import numpy as np
import os
from sold.datasets.utils import EpisodeDatasetInfo, Fields
import torch
from typing import Any, Dict, List, Optional
import warnings


class EpisodeDatasetInfoMixin:
    def __init__(self, info: EpisodeDatasetInfo) -> None:
        self.info = info

    def _update_stats_on_store(self, episode: Dict[str, List]) -> None:
        self.info.num_episodes += 1
        self.info.num_timesteps += len(next(iter(episode.values())))
        self.info.episode_lengths.append(len(next(iter(episode.values()))))

    def _update_stats_on_remove(self) -> None:
        self.info.num_episodes -= 1
        self.info.num_timesteps -= self.info.episode_lengths.pop(0)
        self.info.is_full = True  # is_full is set to True once episodes start being removed.

    @property
    def num_episodes(self) -> int:
        return self.info.num_episodes

    @property
    def num_timesteps(self) -> int:
        return self.info.num_timesteps

    @property
    def episode_lengths(self) -> List[int]:
        return self.info.episode_lengths

    @property
    def is_empty(self) -> bool:
        return self.info.is_empty

    @property
    def is_full(self) -> bool:
        return self.info.is_full

    @property
    def average_episode_length(self) -> float:
        return sum(self.episode_lengths) / self.num_episodes if self.num_episodes > 0 else 0

    def __len__(self) -> int:
        return self.num_timesteps

    @property
    def fields(self) -> Fields:
        return self.info.fields


class EpisodeDataset(EpisodeDatasetInfoMixin, ABC):
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        sequence_length: int = 1,
        info: Optional[EpisodeDatasetInfo] = None,
    ) -> None:
        info = info or EpisodeDatasetInfo()
        EpisodeDatasetInfoMixin.__init__(self, info=info)
        self.capacity = capacity
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.current_episode = defaultdict(list)

        if self.info.fields:
            self._initialize_from_fields(self.info.fields)

    def __repr__(self) -> str:
        summary = (f"{self.__class__.__name__}("
                   f"\n    num_episodes: {self.num_episodes}"
                   f"\n    num_timesteps: {self.num_timesteps}")
        summary += f"\n    fields: {list(self.fields.keys())} \n)" if self.fields else "\n    fields: None\n)"
        return summary

    def _initialize_from_fields(self, fields: Fields) -> None:
        """Used to initialize the episode dataset's storage from fields."""
        pass

    def add_step(self, step_data: Dict[str, Any], done: bool = False) -> None:
        """Add a batch of step data to the buffer.

        Args:
            step_data (dict): Batch of step data to add.
            done (bool): Whether each episode is done.
        """

        for key, value in step_data.items():
            self.current_episode[key].append(torch.tensor(value))

        if done:
            self.add_episode(self.current_episode)
            self.current_episode = defaultdict(list)

    def add_episode(self, episode: Dict[str, List]) -> None:
        # If dataset fields are not defined, initialize them from the first episode.
        if not self.info.fields:
            self.info.fields = Fields.from_episode(episode)
            self._initialize_from_fields(self.info.fields)

        # Remove episodes until there is enough space to store the new episode.
        while len(self) + len(next(iter(episode.values()))) > self.capacity:
            self.remove_episode()
            self._update_stats_on_remove()

        # Add new episode to storage.
        self.store_episode(episode)
        self._update_stats_on_store(episode)

    @abstractmethod
    def store_episode(self, episode: Dict[str, List]) -> None:
        """Add an episode to the storage.

        Args:
            episode (dict): Episode to append.
        """

        raise NotImplementedError

    @abstractmethod
    def remove_episode(self) -> None:
        """Removes the oldest episode from the storage."""

        raise NotImplementedError

    @abstractmethod
    def sample(self) -> Dict[str, torch.Tensor]:
        """Sample a batch of sequences from the dataset.

        Returns:
            dict: Batch of sequences.
        """

        raise NotImplementedError


class RingBufferDataset(EpisodeDataset):
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        sequence_length: int = 1,
        info: Optional[EpisodeDatasetInfo] = None,
        save_path: Optional[str] = None,
    ) -> None:
        super().__init__(capacity, batch_size, sequence_length, info)
        self.save_path = save_path

        # Indices used to keep track of episodes stored in contiguous memory.
        self.episode_boundaries = deque()
        self.head = 0
        self.tail = 0

    def store_episode(self, episode: Dict[str, List]) -> None:
        # Determine place for the new episode.
        start = self.tail
        end = self.tail + len(next(iter(episode.values())))
        self.episode_boundaries.append((start, end))

        # The episode fits in the buffer without wrapping.
        if end <= self.capacity:
            for key, value in episode.items():
                self.ring_buffer[key][start:end] = torch.stack(value).cpu().numpy()

        # The episode wraps around the buffer.
        else:
            split = self.capacity - start
            end %= self.capacity  # Wrap remainder of episode around to the start of the buffer.
            for key, value in episode.items():
                self.ring_buffer[key][start:] = torch.stack(value[:split]).cpu().numpy()
                self.ring_buffer[key][:end] = torch.stack(value[split:]).cpu().numpy()

        # Update tail index.
        self.tail = end % self.capacity

    def remove_episode(self) -> None:
        # Set head index to the start of the second-oldest episode.
        self.head = self.episode_boundaries[1][0] if len(self.episode_boundaries) > 1 else self.tail

        # Remove the oldest episode from the episode boundaries.
        self.episode_boundaries.popleft()

    def sample(self) -> Dict[str, torch.Tensor]:
        # Get indices of viable episodes.
        if self.sequence_length is None:
            viable_episode_indices = torch.arange(self.num_episodes)
        else:
            viable_episode_indices = torch.tensor([i for i in range(self.num_episodes) if
                                                   self.episode_lengths[i] >= self.sequence_length])

        print("self.num_episodes: ", self.num_episodes)
        print("viable_episode_indices: ", viable_episode_indices)

        # Sample random episodes.
        episode_indices = viable_episode_indices[torch.randint(len(viable_episode_indices), (self.batch_size,))]

        # Get tensor of episode boundaries.
        selected_boundaries = torch.tensor([self.episode_boundaries[i] for i in episode_indices])

        # Compute maximum offset that can be used for each sequence.
        max_offsets = torch.remainder(
            selected_boundaries[:, 1] - selected_boundaries[:, 0], self.capacity) - self.sequence_length
        offsets = (max_offsets * torch.rand(self.batch_size)).long()

        # Sample start indices for each sequence.
        start_indices = selected_boundaries[:, 0] + offsets
        linear_sequence_indices = start_indices.unsqueeze(1) + torch.arange(self.sequence_length)

        # Adjust indices to account for circular buffer.
        sequence_indices = torch.remainder(linear_sequence_indices, self.capacity).numpy()

        return {key: torch.from_numpy(self.ring_buffer[key][sequence_indices]) for key in self.ring_buffer.keys()}

    @property
    def is_empty(self) -> bool:
        if self.num_episodes > 0:
            viable_episode_indices = [i for i in range(self.num_episodes) if
                                      self.episode_lengths[i] >= self.sequence_length]
            if len(viable_episode_indices) > 0:
                return False
            else:
                warnings.warn(f"Episodes have been stored to the buffer, but none are of sufficient length for sampling "
                              f"sequence length {self.sequence_length}.")
        return True

    def _initialize_from_fields(self, fields: Fields) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        self.ring_buffer = {}
        for key, value in fields.items():
            filename = os.path.join(self.save_path, f"{key}.dat")

            if value.dtype == torch.float32:
                dtype = 'float32'
            elif value.dtype == torch.bool:
                dtype = 'bool'
            else:
                assert False

            self.ring_buffer[key] = np.memmap(filename, dtype=dtype, mode='w+', shape=(self.capacity, *value.shape))
