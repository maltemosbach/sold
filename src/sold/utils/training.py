from abc import ABC, abstractmethod
from collections import defaultdict
import gym
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
import os
import random
from sold.datasets.ring_buffer import RingBufferDataset
from sold.datasets.utils import NumUpdatesWrapper
from sold.utils.logging import ExtendedLoggingModule
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


class OnlineModule(ExtendedLoggingModule, ABC):
    def __init__(self, env: gym.Env, seed_steps: int = 0, update_freq: int = 1, num_updates: int = 1,
                 eval_freq: int = 1000, num_eval_episodes: int = 10, batch_size: int = 16, sequence_length: int = 1,
                 buffer_capacity: int = 1e6) -> None:
        """Integrates online experience collection with the PyTorch Lightning training loop.

        Args:
            env (gym.Env): The environment to interact with.
            seed_steps (int): Number of steps to collect before starting training.
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

        # Keep track of the current state of the train-eval loop and MDP.
        self.eval_next = False
        self.after_eval = False
        self.obs = None
        self.done = True
        self.last_action = torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)), float('nan')).to(self.device)

    @abstractmethod
    def select_action(self, obs: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        pass

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

    def _complete_first_timestep(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in step_data:
            step_data["action"] = torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)), float('nan'))
        if "reward" not in step_data:
            step_data["reward"] = torch.tensor(float('nan'))
        return step_data

    @torch.no_grad()
    def on_train_epoch_start(self) -> None:
        self.collect_step()

    @torch.no_grad()
    def collect_step(self) -> None:
        """Collect one step of environment experience."""
        if self.current_step % self.eval_freq == 0:
            self.eval_next = True

        if self.done:
            if self.eval_next:
                self.run_evaluation()

            # Reset environment and store initial observation.
            self.log("train/buffer_size", self.replay_buffer.num_timesteps)
            if self.replay_buffer.last_episode_return is not None:
                self.log("train/episode_return", self.replay_buffer.last_episode_return, prog_bar=True)
            self.obs = self.env.reset()
            self.replay_buffer.add_step(self._complete_first_timestep({"obs": self.obs}))

        # Select action, perform environment step, and store resulting experience.
        mode = "train" if self.current_step >= self.seed_steps else "random"
        self.last_action[:] = self.select_action(self.obs.to(self.device), is_first=self.done, mode=mode).cpu()
        self.obs, reward, self.done, info = self.env.step(self.last_action)
        self.replay_buffer.add_step({"obs": self.obs, "action": self.last_action, "reward": reward}, done=self.done)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.after_eval = False  # Reset 'after_eval' to False after the first training batch.

    # def on_train_epoch_end(self) -> None:
    #     self.after_eval = False  # Reset 'after_eval' to False after the training epoch.

    @torch.no_grad()
    def run_evaluation(self) -> None:
        episode_returns, successes = [], []
        for episode_index in range(self.num_eval_episodes):
            episode = self.play_episode(mode="eval")
            self.log(f"eval/episode_{episode_index}", torch.stack(episode["obs"]))
            episode_returns.append(sum(episode["reward"]))
            if "success" in episode:
                successes.append(episode["success"])
        self.log("eval/episode_return", np.mean(episode_returns), prog_bar=True)
        if successes:
            self.log("eval/success_rate", np.mean(successes))

        for episode_index in range(3):
            episode = self.play_episode(mode="train")
            self.log(f"train/episode_{episode_index}", torch.stack(episode["obs"]))

        # Save model checkpoint.
        save_dir = os.path.join(self.logger.log_dir, "checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.trainer.save_checkpoint(os.path.join(save_dir, f"sold-steps={self.current_step}-episode={self.replay_buffer.num_episodes}-eval_episode_return={np.mean(episode_returns)}.ckpt"))

        self.eval_next = False
        self.after_eval = True

    @torch.no_grad()
    def play_episode(self, mode: str = "eval") -> Dict[str, Any]:
        """Run current policy for an episode to evaluate it without storing any experiences."""
        if not self.done:
            raise RuntimeError("Current training episode must have terminated before playing an episode.")

        self.obs, self.done = self.env.reset(), False
        episode = defaultdict(list)
        episode["obs"].append(self.obs)
        while not self.done:
            self.last_action = self.select_action(self.obs.to(self.device), is_first=len(episode["obs"]) == 1, mode=mode).cpu()
            self.obs, reward, self.done, info = self.env.step(self.last_action)
            episode["obs"].append(self.obs.cpu())
            episode["reward"].append(reward)

        if "success" in info:
            episode["success"] = info["success"]
        return episode
