from abc import ABC, abstractmethod
from collections import defaultdict
import gym
from lightning.pytorch import LightningModule
import numpy as np
from lightning.pytorch.utilities.types import STEP_OUTPUT
import os
from sold.datasets.ring_buffer import RingBufferDataset
from sold.datasets.utils import NumUpdatesWrapper
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, Union, Callable
import warnings


class OnlineModule(LightningModule, ABC):
    def __init__(self, env: gym.Env, seed_steps: int = 100, update_freq: int = 1, num_updates: int = 1,
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
        self.after_eval = False
        self.obs = None
        self.done = True

    def on_fit_start(self) -> None:
        self.logger._model = self

    @property
    def current_step(self) -> int:
        return self.current_epoch

    def get_num_updates(self) -> int:
        if self.current_step >= self.seed_steps and self.current_step % self.update_freq == 0:
            if self.replay_buffer.is_empty:
                warnings.warn("Replay buffer is empty. Skipping update.")
                return 0
            return self.num_updates
        return 0

    def train_dataloader(self) -> DataLoader:
        self.replay_buffer = RingBufferDataset(self.buffer_capacity, self.batch_size, self.sequence_length,
                                               save_path=self.logger.log_dir + "/replay_buffer")
        dataset = NumUpdatesWrapper(self.replay_buffer.sample, self.get_num_updates)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, num_workers=1)
        return dataloader

    @abstractmethod
    def select_action(self, obs: torch.Tensor, is_first: bool = False, sample: bool = False) -> torch.Tensor:
        pass

    def to_time_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in step_data:
            step_data["action"] = torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)), float('nan'))
        if "reward" not in step_data:
            step_data["reward"] = torch.tensor(float('nan'))
        return step_data

    @torch.no_grad()
    def on_train_epoch_start(self) -> None:
        """Collect one step of environment experience."""
        self.after_eval = False

        if self.current_step % self.eval_freq == 0:
            self.eval_next = True

        if self.done:
            if self.eval_next:
                self.run_evaluation()

            # Reset environment and store initial observation.
            self.log("train/buffer_size", self.replay_buffer.num_timesteps)
            if self.current_step > 0:
                self.log("train/episode_return", self.replay_buffer.last_episode_return, prog_bar=True)
            self.obs = self.env.reset()
            self.replay_buffer.add_step(self.to_time_step({"obs": self.obs, "is_first": True}))

        # Select action, perform environment step, and store resulting experience.
        if self.current_step <= self.seed_steps:
            action = torch.from_numpy(self.env.action_space.sample().astype(np.float32))
        else:
            action = self.select_action(self.obs.to(self.device), is_first=self.done, sample=True).cpu()
        self.obs, reward, self.done, info = self.env.step(action)
        self.replay_buffer.add_step(self.to_time_step({"obs": self.obs, "action": action, "reward": reward, "is_first": False}), done=self.done)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.after_eval = False  # Only true for the first batch of the epoch.

    @torch.no_grad()
    def run_evaluation(self) -> None:
        episode_returns, successes = [], []
        for episode_index in range(self.num_eval_episodes):
            episode = self.collect_eval_episode()
            self.logger.log_video(f"eval/episode_{episode_index}", torch.stack(episode["obs"]))
            episode_returns.append(sum(episode["reward"]))
            if "success" in episode:
                successes.append(episode["success"])
        self.log("eval/episode_return", np.mean(episode_returns), prog_bar=True)
        if successes:
            self.log("eval/success_rate", np.mean(successes))

        # Save model checkpoint.
        save_dir = os.path.join(self.logger.log_dir, "checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.trainer.save_checkpoint(os.path.join(save_dir, f"sold-steps={self.current_step}-episodes={self.replay_buffer.num_episodes}-eval_episode_return={np.mean(episode_returns)}.ckpt"))

        self.eval_next = False
        self.after_eval = True

    @torch.no_grad()
    def collect_eval_episode(self) -> Dict[str, Any]:
        if not self.done:
            raise RuntimeError("Current training episode must have terminated before collecting a validation episode.")

        self.obs, self.done = self.env.reset(), False
        episode = defaultdict(list)
        episode["obs"].append(self.obs)
        while not self.done:
            action = self.select_action(self.obs.to(self.device), is_first=len(episode["obs"]) == 1, sample=False)
            self.obs, reward, self.done, info = self.env.step(action)
            episode["obs"].append(self.obs.cpu())
            episode["reward"].append(reward)

        if "success" in info:
            episode["success"] = info["success"]
        return episode
