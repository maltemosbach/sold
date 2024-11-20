from abc import ABC, abstractmethod
from collections import defaultdict
import gym
from lightning.pytorch import LightningModule
import numpy as np
from sold.datasets.episode_dataset import RingBufferDataset
from sold.dataset.utils import NumUpdatesWrapper
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, List


class OnlineModule(LightningModule, ABC):
    def __init__(self, env: gym.Env, seed_steps: int = 0, update_freq: int = 1, num_updates: int = 1,
                 eval_freq: int = 1000, num_eval_episodes: int = 10, batch_size: int = 16, sequence_length: int = 1,
                 buffer_capacity: int = 1e6) -> None:
        """Integrates online experience collection with the PyTorch Lightning training loop.

        Args:
            env (gym.Env): The environment to interact with.
            seed_steps (int): Number of steps to collect before training.
            update_freq (int): Update the agent every 'update_freq' environment steps.
            num_updates (int): Number of updates to perform whenever the agent is being updated.
            eval_freq (int): Evaluate the agent every 'eval_freq' environment steps.
            num_eval_episodes (int): Number of episodes to collect when evaluating the agent.
            batch_size (int): Batch size of experience sampled from the replay buffer.
            sequence_length (int): Length of sequences sampled from the replay buffer.
            buffer_capacity (int): Maximum number of steps to store in the replay buffer.
        """
        super().__init__()
        self.env = env
        self.seed_steps = seed_steps
        self.update_freq = update_freq
        self.num_updates = num_updates
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.buffer_capacity = buffer_capacity

        self.eval_next = False
        self.obs = None
        self.done = True

    @property
    def current_step(self) -> int:
        return self.current_epoch

    def get_num_updates(self) -> int:
        if self.current_step > self.seed_steps and self.current_step % self.update_every_n_steps == 0:
            return self.num_updates
        return 0

    def train_dataloader(self) -> DataLoader:
        self.replay_buffer = RingBufferDataset(self.buffer_capacity, self.batch_size, self.sequence_length,
                                               save_path=self.logger.log_dir + "/replay_buffer")
        dataset = NumUpdatesWrapper(self.replay_buffer.sample, self.get_num_updates)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, num_workers=1)
        return dataloader

    @abstractmethod
    def select_action(self, obs: torch.Tensor, is_first: bool = False, sample: bool = False) -> np.ndarray:
        pass

    def to_time_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in step_data:
            step_data["action"] = torch.full_like(self.env.action_space.sample(), float('nan'))
        if "reward" not in step_data:
            step_data["reward"] = torch.tensor(float('nan'))
        return step_data

    def on_train_epoch_start(self) -> None:
        """Collect one step of environment experience."""

        if self.current_step % self.eval_freq == 0:
            self.eval_next = True

        if self.done:
            if self.eval_next:
                self.run_evaluation()
                self.eval_next = False

            # Reset environment and store initial observation.
            if self.current_step > 0:
                self.log("train/episode_return", self.current_episode_return, prog_bar=True)
            self.obs = self.env.reset()
            self.replay_buffer.add_step(self.to_time_step({"obs": self.obs, "is_first": True}))

        # Select action, perform environment step, and store resulting experience.
        if self.steps <= self.seed_steps:
            action = self.env.action_space.sample()
        else:
            action = self.select_action(self.obs, is_first=self.done, sample=True)
        self.obs, reward, self.done, info = self.env.step(action)
        self.replay_buffer.add_step(self.to_time_step({"obs": self.obs, "action": action, "reward": reward, "is_first": False}), done=self.done)

    def run_evaluation(self) -> None:
        episode_returns, successes = [], []
        for _ in range(self.num_eval_episodes):
            episode = self.collect_eval_episode()
            episode_returns.append(sum(episode["reward"]))
            if "success" in episode:
                successes.append(episode["success"])

        self.log("eval/episode_return", np.mean(episode_returns), prog_bar=True)
        if successes:
            self.log("eval/success_rate", np.mean(successes), prog_bar=True)

    def collect_eval_episode(self) -> Dict[str, Any]:
        if not self.done:
            raise RuntimeError("Current training episode must have terminated before collecting a validation episode.")

        self.obs, self.done = self.env.reset(), False
        episode = defaultdict(list)
        episode["obs"].append(self.obs)
        while not self.done:
            action = self.select_action(self.obs, is_first=len(episode["obs"]) == 1, sample=False)
            self.obs, reward, self.done, info = self.env.step(action)
            episode["obs"].append(self.obs)
            episode["reward"].append(reward)

        if "success" in info:
            episode["success"] = info["success"]
        return episode
