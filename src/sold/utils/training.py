from abc import ABC, abstractmethod
from collections import defaultdict
import gym
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
import os
import random
from sold.datasets.ring_buffer import RingBufferDataset
from sold.datasets.utils import NumUpdatesWrapper
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict
import warnings


def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class OnlineModule(LightningModule, ABC):
    def __init__(self, env: gym.Env, train_after: int = 0, update_freq: int = 1, num_updates: int = 1,
                 eval_freq: int = 1000, num_eval_episodes: int = 10, batch_size: int = 16, sequence_length: int = 1,
                 buffer_capacity: int = 1e6, interval: str = "episode") -> None:
        """Integrates online experience collection with the PyTorch Lightning training loop.

        Args:
            env (gym.Env): The environment to interact with.
            train_after (int): Number of intervals to wait before starting training. (e.g. 2 with interval='episode' means training starts after 2 episodes.)
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
        self.train_after = train_after
        self.update_freq = update_freq
        self.num_updates = num_updates
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.buffer_capacity = buffer_capacity
        self.interval = interval
        assert interval in ["time_step", "episode"]

        self.eval_next = False
        self.after_eval = False
        self.obs = None
        self.done = True
        self.last_action = torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)), float('nan')).to(self.device)

        self.current_time_step = 0
        self.current_episode = 0

    def on_fit_start(self) -> None:
        self.logger.pl_module = self

    @property
    def current_interval_step(self) -> int:
        return self.current_epoch

    def get_num_updates(self) -> int:
        if self.current_interval_step >= self.train_after and self.current_interval_step % self.update_freq == 0:
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
    def select_action(self, obs: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        pass

    def to_time_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in step_data:
            step_data["action"] = torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)), float('nan'))
        if "reward" not in step_data:
            step_data["reward"] = torch.tensor(float('nan'))
        return step_data

    @torch.no_grad()
    def on_train_epoch_start(self) -> None:
        if self.interval == "time_step":
            self.collect_step()
        elif self.interval == "episode":
            self.collect_episode()

    @torch.no_grad()
    def collect_step(self) -> None:
        """Collect one step of environment experience."""
        if self.current_interval_step % self.eval_freq == 0:
            self.eval_next = True

        if self.done:
            self.env.close()
            if self.eval_next:
                self.run_evaluation()

            # Reset environment and store initial observation.
            self.log("train/buffer_size", self.replay_buffer.num_timesteps)
            if self.current_time_step > 0:
                self.log("train/episode_return", self.replay_buffer.last_episode_return, prog_bar=True)
            self.obs = self.env.reset()
            self.replay_buffer.add_step(self.to_time_step({"obs": self.obs}))

        # Select action, perform environment step, and store resulting experience.
        mode = "train" if self.current_time_step >= self.train_after else "random"
        self.last_action[:] = self.select_action(self.obs.to(self.device), is_first=self.done, mode=mode).cpu()
        self.obs, reward, self.done, info = self.env.step(self.last_action)

        self.replay_buffer.add_step(
            self.to_time_step({"obs": self.obs, "action": self.last_action, "reward": reward}), done=self.done)
        self.current_time_step += 1

    def collect_episode(self) -> None:
        """Collect one episode of environment experience."""
        self.current_episode += 1
        self.eval_next = False
        if not self.done:
            raise RuntimeError("Previous episode should be done at the start of 'collect_episode'.")

        self.collect_step()
        while not self.done:
            self.collect_step()

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
        self.trainer.save_checkpoint(os.path.join(save_dir, f"sold-steps={self.current_time_step}-episode={self.current_episode}-eval_episode_return={np.mean(episode_returns)}.ckpt"))

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
            self.last_action = self.select_action(self.obs.to(self.device), is_first=len(episode["obs"]) == 1, mode="eval").cpu()
            self.obs, reward, self.done, info = self.env.step(self.last_action)
            episode["obs"].append(self.obs.cpu())
            episode["reward"].append(reward)

        if "success" in info:
            episode["success"] = info["success"]
        return episode
