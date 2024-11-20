from collections import defaultdict
import os
import time
import numpy as np
import torch


class ExperienceReplay:
    def __init__(self, file_path, size, ep_length, observation_size, action_size):
        self.file_path = file_path
        self.size = size
        self.ep_length = ep_length
        self.action_size = action_size
        self.observation_size = observation_size
        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        self.episodes = 0  # Tracks how much experience has been used in total

        # File names for memory-mapped arrays
        os.makedirs(file_path, exist_ok=True)
        timestamp = int(time.time())
        self.obs_filename = os.path.join(file_path, f'observations_{timestamp}.dat')
        self.act_filename = os.path.join(file_path, f'actions_{timestamp}.dat')
        self.rew_filename = os.path.join(file_path, f'rewards_{timestamp}.dat')
        self.first_filename = os.path.join(file_path, f'is_first_{timestamp}.dat')

        # Create memory-mapped arrays for observations, actions, and rewards
        self.observations = np.memmap(self.obs_filename, dtype='float32', mode='w+', shape=(size, ep_length+1, 3, *observation_size))
        self.actions = np.memmap(self.act_filename, dtype='float32', mode='w+', shape=(size, ep_length+1, action_size))
        self.rewards = np.memmap(self.rew_filename, dtype='float32', mode='w+', shape=(size, ep_length+1))
        self.is_first = np.memmap(self.first_filename, dtype='bool', mode='w+', shape=(size, ep_length+1))

    def append(self, observations, actions, rewards, is_first):
        self.observations[self.idx] = torch.cat(observations).numpy()
        self.actions[self.idx] = torch.stack(actions).numpy()
        self.rewards[self.idx] = np.array(rewards)
        self.is_first[self.idx] = np.array(is_first)
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.episodes += 1

    def _sample_ep_idx(self, n):
        max_idx = self.size if self.full else self.episodes
        return np.random.randint(0, max_idx, size=n)

    def _sample_idx_in_ep(self, n, length):
        return np.random.randint(0, self.ep_length - length, size=n) if self.ep_length - length > 0 else np.zeros(n, dtype=int)

    def _retrieve_batch(self, ep_indx, start_indx, n, length):
        # Create an empty array for the batch
        observations_batch = np.empty((n, length, 3, *self.observation_size), dtype=np.float32)
        actions_batch = np.empty((n, length, self.action_size), dtype=np.float32)
        rewards_batch = np.empty((n, length), dtype=np.float32)
        is_first_batch = np.empty((n, length), dtype=bool)

        # Retrieve data for each sample in the batch
        for i in range(n):
            observations_batch[i] = self.observations[ep_indx[i], start_indx[i]:start_indx[i] + length]
            actions_batch[i] = self.actions[ep_indx[i], start_indx[i]:start_indx[i] + length]
            rewards_batch[i] = self.rewards[ep_indx[i], start_indx[i]:start_indx[i] + length]
            is_first_batch[i] = self.is_first[ep_indx[i], start_indx[i]:start_indx[i] + length]

        return (
            torch.tensor(observations_batch),
            torch.tensor(actions_batch),
            torch.tensor(rewards_batch),
            torch.tensor(is_first_batch)
        )

    def sample(self, n, length):
        ep_indx = self._sample_ep_idx(n)
        start_indx = self._sample_idx_in_ep(n, length)
        return self._retrieve_batch(ep_indx, start_indx, n, length)

    def cleanup(self):
        # Remove memory-mapped files
        os.remove(self.obs_filename)
        os.remove(self.act_filename)
        os.remove(self.rew_filename)
        os.remove(self.first_filename)

    def __del__(self):
        del self.observations
        del self.actions
        del self.rewards
        del self.is_first
