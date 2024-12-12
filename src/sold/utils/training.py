from abc import ABC, abstractmethod
from collections import defaultdict
import gym
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
import os
import random
from sold.datasets.ring_buffer import RingBufferDataset
from sold.datasets.utils import NumUpdatesWrapper
from lightning import LightningModule
from sold.utils.logging import LoggingStepMixin
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict


def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class OnlineModule(LoggingStepMixin, LightningModule, ABC):
    def __init__(self, env: gym.Env, max_steps: int, num_seed: int, update_freq: int, num_updates: int,
                 eval_freq: int, num_eval_episodes: int, batch_size: int, sequence_length: int, buffer_capacity: int
                 ) -> None:
        """Integrates online experience collection with PyTorch Lightning's training loop.

        Args:
            env (gym.Env): The environment to interact with.
            max_steps (int): Maximum number of environment steps to collect before stopping training.
            num_seed (int): Number of seed steps/episodes to collect before training.
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
        self.max_steps = max_steps
        self.num_seed = num_seed
        self.update_freq = update_freq
        self.num_updates = num_updates
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.buffer_capacity = buffer_capacity

        self.num_steps = 0
        self.num_episodes = -1  # Start at -1 to account for the initial reset.

    @abstractmethod
    def select_action(self, obs: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        pass

    def on_fit_start(self) -> None:
        self.eval_next = False
        self.after_eval = False
        self.obs = None
        self.done = True
        self.last_action = torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)),
                                           float('nan')).to(self.device)

    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, "replay_buffer"):
            self.replay_buffer = RingBufferDataset(self.buffer_capacity, self.batch_size, self.sequence_length,
                                                   save_path=self.logger.log_dir + "/replay_buffer")
        dataset = NumUpdatesWrapper(self.replay_buffer, self.num_updates)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, num_workers=1)
        return dataloader

    @torch.no_grad()
    def on_train_epoch_start(self) -> None:
        num_collect_steps = self.num_seed if self.num_steps == 0 and self.num_seed > 0 else self.update_freq
        for _ in range(num_collect_steps):
            self.collect_step()

    @torch.no_grad()
    def collect_step(self) -> None:
        if self.num_steps % self.eval_freq == 0:
            self.eval_next = True

        if self.done:
            self.num_episodes += 1
            if self.eval_next:
                self.eval_loop()

            # Reset environment and store initial observation.
            self.log("train/buffer_size", self.replay_buffer.num_timesteps)
            if self.replay_buffer.last_episode_return is not None:
                self.log("train/episode_return", self.replay_buffer.last_episode_return.item(), prog_bar=True)
            self.obs = self.env.reset()
            self.replay_buffer.add_step(self._complete_first_timestep({"obs": self.obs}))
            self.log("train/num_episodes", self.num_episodes)
            self.log("train/num_steps", self.num_steps)

        # Select action, perform environment step, and store resulting experience.
        mode = "train" if self.num_steps >= self.num_seed else "random"
        self.last_action[:] = self.select_action(self.obs.to(self.device), is_first=self.done, mode=mode)
        self.obs, reward, self.done, info = self.env.step(self.last_action.cpu())
        self.replay_buffer.add_step({"obs": self.obs, "action": self.last_action.cpu(), "reward": reward}, done=self.done)
        self.num_steps += 1

    def _complete_first_timestep(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in step_data:
            step_data["action"] = torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)),
                                                  float('nan'))
        if "reward" not in step_data:
            step_data["reward"] = torch.tensor(float('nan'))
        return step_data

    @torch.no_grad()
    def eval_loop(self) -> None:
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
        self.trainer.save_checkpoint(
            os.path.join(save_dir, f"sold-steps={self.num_steps}-episode={self.num_episodes}-eval_episode_return="
                                   f"{np.mean(episode_returns)}.ckpt"))

        self.eval_next = False
        self.after_eval = True

    @torch.no_grad()
    def play_episode(self, mode: str = "eval") -> Dict[str, Any]:
        """Run current policy for an episode to evaluate it without storing any experiences."""
        if not self.done:
            raise RuntimeError("Current training episode must have terminated before playing an episode.")

        self.obs, self.done, info = self.env.reset(), False, {}
        episode = defaultdict(list)
        episode["obs"].append(self.obs)
        while not self.done:
            self.last_action[:] = self.select_action(self.obs.to(self.device), is_first=len(episode["obs"]) == 1,
                                                  mode=mode)
            self.obs, reward, self.done, info = self.env.step(self.last_action.cpu())
            episode["obs"].append(self.obs.cpu())
            episode["reward"].append(reward)

        if "success" in info:
            episode["success"] = info["success"]
        return episode

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.after_eval = False  # Reset 'after_eval' to False after the first training batch.

    def on_train_epoch_end(self) -> None:
        if self.num_steps >= self.max_steps:
            self.trainer.should_stop = True

    @property
    def logging_step(self) -> int:
        return self.num_steps

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["num_steps"] = self.num_steps
        checkpoint["num_episodes"] = self.num_episodes
        checkpoint["replay_buffer_capacity"] = self.replay_buffer.capacity
        checkpoint["replay_buffer_path"] = os.path.abspath(self.replay_buffer.save_path)
        checkpoint["replay_buffer_info"] = self.replay_buffer.info
        checkpoint["replay_buffer_head"] = self.replay_buffer.head
        checkpoint["replay_buffer_tail"] = self.replay_buffer.tail
        checkpoint["replay_buffer_episode_boundaries"] = self.replay_buffer.episode_boundaries

    def on_load_checkpoint(self, checkpoint: Dict[str, Any], load_replay_buffer: bool = False) -> None:
        self.num_steps = checkpoint["num_steps"]
        self.num_episodes = checkpoint["num_episodes"]

        # Load latest replay buffer state and ensure that it is consistent with this checkpoint by setting the pointers.
        # Data that has been added later will simply be ignored and overwritten. Only works while replay buffer is not
        # full.
        if load_replay_buffer:
            if checkpoint["replay_buffer_info"].is_full:
                raise ValueError("Cannot load replay buffer checkpoint that is already full. Since we only save the "
                                 "latest state of the replay buffer and not the entire history, we cannot guarantee "
                                 "that the view on the data generated by loading 'head', 'tail' and "
                                 "'episode_boundaries' is correct, since episodes have already been overwritten.")

            self.train_dataloader()  # Initializes the replay buffer.
            if checkpoint["replay_buffer_info"].fields is None:
                raise ValueError("Cannot load replay buffer without fields.")
            self.replay_buffer.load_from_files(checkpoint["replay_buffer_path"], checkpoint["replay_buffer_capacity"],
                                               checkpoint["replay_buffer_info"], checkpoint["replay_buffer_head"],
                                               checkpoint["replay_buffer_tail"],
                                               checkpoint["replay_buffer_episode_boundaries"])
