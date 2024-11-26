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




import os
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any, Literal, Optional, Union, cast

import lightning as L
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from tqdm import tqdm


class MyCustomTrainer:
    def __init__(
        self,
        env: gym.Env, seed_steps: int = 0, update_freq: int = 1, num_updates: int = 1,
        eval_freq: int = 1000, num_eval_episodes: int = 10, batch_size: int = 16, sequence_length: int = 1,
        buffer_capacity: int = 1e6,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[list[Any], Any]] = None,
        loggers: Optional[Union[Logger, list[Logger]]] = None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
    ) -> None:
        """Exemplary Trainer with Fabric. This is a very simple trainer focused on readablity but with reduced
        featureset. As a trainer with more included features, we recommend using the
        :class:`lightning.pytorch.Trainer`.

        Args:
            accelerator: The hardware to run on. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
            strategy: Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            devices: Number of devices to train on (``int``),
                which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                The value applies per node.
            precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
                or bfloat16 precision AMP (``"bf16-mixed"``).
            plugins: One or several custom plugins
            callbacks: A single callback or a list of callbacks. The following hooks are supported:
                - on_train_epoch_start
                - on train_epoch_end
                - on_train_batch_start
                - on_train_batch_end
                - on_before_backward
                - on_after_backward
                - on_before_zero_grad
                - on_before_optimizer_step
                - on_validation_model_eval
                - on_validation_model_train
                - on_validation_epoch_start
                - on_validation_epoch_end
                - on_validation_batch_start
                - on_validation_batch_end

            loggers: A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
                information.

            max_epochs: The maximum number of epochs to train
            max_steps: The maximum number of (optimizer) steps to train
            grad_accum_steps: How many batches to process before each optimizer step
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches: Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            validation_frequency: How many epochs to run before each validation epoch.
            use_distributed_sampler: Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            checkpoint_dir: Directory to store checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.

        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won't work!

        """

        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.global_step = 0
        self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

        self.env = env
        self.seed_steps = seed_steps
        self.update_freq = update_freq
        self.num_updates = num_updates
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.buffer_capacity = buffer_capacity

    def fit(
        self,
        model: L.LightningModule,
        ckpt_path: Optional[str] = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.

        """
        self.fabric.launch()

        # setup model and optimizer
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            raise NotImplementedError("OnlineTrainer currently does not support FSDP")

        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(model.configure_optimizers())
        assert optimizer is not None
        model, optimizer = self.fabric.setup(model, optimizer)

        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}

        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)

                # check if we even need to train here
                if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                    self.should_stop = True

        # Create replay buffer.
        self.replay_buffer = RingBufferDataset(self.buffer_capacity, self.batch_size, self.sequence_length,
                                               save_path=self.logger.log_dir + "/replay_buffer")


        prog_bar = tqdm(range(self.max_steps), desc="Training", disable=self.fabric.is_global_zero)

        for self.current_step in prog_bar:
            prog_bar.set_description(f"Steps {self.current_step}, Episodes {self.replay_buffer.num_episodes}")

            # Queue up the next evaluation if it's time.
            if self.current_step % self.eval_freq == 0:
                self.eval_next = True

            if self.done:
                if self.eval_next:
                    self.eval_loop()

                # Reset environment and store initial observation.
                self.log("train/buffer_size", self.replay_buffer.num_timesteps)
                if self.current_time_step > 0:
                    self.log("train/episode_return", self.replay_buffer.last_episode_return, prog_bar=True)
                self.obs = self.env.reset()
                self.replay_buffer.add_step(self.to_time_step({"obs": self.obs, "is_first": True}))

            # Select action, perform environment step, and store resulting experience.
            if self.current_step <= self.seed_steps:
                action = torch.from_numpy(self.env.action_space.sample().astype(np.float32))
            else:
                action = self.select_action(self.obs.to(self.device), is_first=self.done, sample=True).cpu()
            self.obs, reward, self.done, info = self.env.step(action)
            self.replay_buffer.add_step(
                self.to_time_step({"obs": self.obs, "action": action, "reward": reward, "is_first": False}),
                done=self.done)

            if self.current_step >= self.seed_steps and not self.replay_buffer.is_empty:
                for batch_index in range(self.num_updates):
                    batch = self.replay_buffer.sample()
                    model.training_step(batch, batch_index)

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch)

        self.fabric.save(os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"), state)

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {"interval": "epoch", "frequency": 1, "monitor": "val_loss"}

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(isinstance(_opt_cand, L.fabric.utilities.types.Optimizable) for _opt_cand in configure_optim_output):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(configure_optim_output[0])[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    @staticmethod
    def _format_iterable(
        prog_bar, candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], prefix: str
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)







class OnlineTrainer(ABC):
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
        self.after_eval = False

        if self.interval == "time_step":
            self.collect_step()
        elif self.interval == "episode":
            self.collect_episode()

    def collect_step(self) -> None:
        """Collect one step of environment experience."""
        if self.current_interval_step % self.eval_freq == 0:
            self.eval_next = True

        if self.done:
            if self.eval_next:
                self.run_evaluation()

            # Reset environment and store initial observation.
            self.log("train/buffer_size", self.replay_buffer.num_timesteps)
            if self.current_time_step > 0:
                self.log("train/episode_return", self.replay_buffer.last_episode_return, prog_bar=True)
            self.obs = self.env.reset()
            self.replay_buffer.add_step(self.to_time_step({"obs": self.obs, "is_first": True}))

        # Select action, perform environment step, and store resulting experience.
        if self.current_interval_step <= self.train_after:
            action = torch.from_numpy(self.env.action_space.sample().astype(np.float32))
        else:
            action = self.select_action(self.obs.to(self.device), is_first=self.done, sample=True).cpu()
        self.obs, reward, self.done, info = self.env.step(action)
        self.replay_buffer.add_step(
            self.to_time_step({"obs": self.obs, "action": action, "reward": reward, "is_first": False}), done=self.done)
        self.current_time_step += 1

    def collect_episode(self) -> None:
        """Collect one episode of environment experience."""
        self.current_episode += 1
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
            action = self.select_action(self.obs.to(self.device), is_first=len(episode["obs"]) == 1, sample=False)
            self.obs, reward, self.done, info = self.env.step(action)
            episode["obs"].append(self.obs.cpu())
            episode["reward"].append(reward)

        if "success" in info:
            episode["success"] = info["success"]
        return episode



