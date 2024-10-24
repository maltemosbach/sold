from functools import partial
from typing import Any

from lightning import LightningModule
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import Optimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT, TRAIN_DATALOADERS
from typing import Callable, Iterator
import sold
from sold.savi.model import SAVi
from sold.sold.replay_buffer import ExperienceReplay
from sold.modeling.prediction import GaussianPredictor, TwoHotPredictor
import numpy as np
from torchvision import transforms


class ExperienceSourceDataset(IterableDataset):
    def __init__(self, generate_batch: Callable) -> None:
        self._generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self._generate_batch()
        return iterator


class SOLD(LightningModule):
    def __init__(self, env: sold.Env, savi: SAVi, actor: partial[GaussianPredictor], critic: partial[TwoHotPredictor],
                 reward_predictor: partial[TwoHotPredictor], learning_rate: float) -> None:
        super().__init__()
        self.env = env
        self.savi = savi
        regression_infos = {"max_episode_steps": env.max_episode_steps,  "num_slots": savi.corrector.num_slots,
                            "slot_dim": savi.corrector.slot_dim}
        self.actor = actor(**regression_infos, output_dim=env.action_space.shape[0])
        self.critic = critic(**regression_infos)
        self.reward_predictor = reward_predictor(**regression_infos)

        self.learning_rate = learning_rate

        self.savi_finetuning_optimizer = torch.optim.Adam(self.savi.parameters(), lr=0.0001)

        self.batch_size = 64
        self.sequence_length = 10
        self.update_every_n_episodes = 1
        self.num_updates = 10

    def on_fit_start(self) -> None:
        """Populate the replay buffer with random seed episodes."""
        self.replay_buffer = ExperienceReplay(self.logger.log_dir + "/replay_buffer", int(1e6),
                                              self.env.max_episode_steps // self.env.action_repeat, self.env.image_size,
                                              self.env.action_space.shape[0])
        num_seed_episodes = 10
        for episode_idx in range(num_seed_episodes):
            images, actions, rewards, is_first = [], [], [], []
            image, done, time_step = self.env.reset(), False, 0

            images.append(transforms.ToTensor()(image.copy()).unsqueeze(dim=0))
            actions.append(torch.zeros(self.env.action_space.shape[0]))
            rewards.append(0)
            is_first.append(True)

            while not done:
                action = self.env.action_space.sample()
                image, reward, done = self.env.step(action)
                images.append(transforms.ToTensor()(image.copy()).unsqueeze(dim=0))
                actions.append(torch.from_numpy(action).float())
                rewards.append(reward)
                is_first.append(False)
                time_step += 1

            self.replay_buffer.append(images, actions, rewards, is_first)
            self.env.close()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = ExperienceSourceDataset(self._sample_batch)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, num_workers=1)
        return dataloader

    def _sample_batch(self):
        for _ in range(self.num_updates):
            videos, actions, rewards, is_firsts = self.replay_buffer.sample(self.batch_size, self.sequence_length)
            def return_batch(i):
                return videos[i], actions[i], rewards[i], is_firsts[i]
            for i in range(self.batch_size):
                yield return_batch(i)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        actor_weights = list(self.actor.parameters())
        critic_weights = list(self.critic.parameters())
        reward_predictor_weights = list(self.reward_predictor.parameters())
        return torch.optim.Adam([
            {'params': actor_weights, 'lr': self.learning_rate},
            {'params': critic_weights, 'lr': self.learning_rate},
            {'params': reward_predictor_weights, 'lr': self.learning_rate}
        ])

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        images, actions,rewards, is_firsts = batch

        print("batch_idx:", batch_idx)

        print("images.shape:", images.shape)
        print("actions.shape:", actions.shape)
        print("rewards.shape:", rewards.shape)
        print("is_firsts.shape:", is_firsts.shape)
        input()

        return None

        # if self.savi.finetune and episode_idx % self.savi.finetune_every_n_episodes == 0:
        #     self._finetune_savi()
        #
        # self.predictor_train_every_n_episodes = 10
        # if episode_idx % self.predictor_train_every_n_episodes:
        #     self._train_predictor()

    def on_train_epoch_end(self) -> None:
        """Collect data samples after every epoch."""
        print("Collecting new episode.")

        input()

        images = []
        actions = []
        rewards = []
        is_first = []

        image, total_reward = self.env.reset(), 0
        images.append(image)
        actions.append(torch.zeros(self.env.action_space.shape[0]))
        rewards.append(0.0)
        is_first.append(True)

        slot_history = None
        for time_step in range(self.env.max_episode_steps):
            # Encode image into slots and append to the slot history.
            last_slots = slot_history[:, -1] if slot_history is not None else None
            step_offset = 1 if slot_history is not None else 0
            slots = self.savi(image, prior_slots=last_slots, step_offset=step_offset, reconstruct=False)
            slot_history = torch.cat([slot_history, slots], dim=1) if slot_history is not None else slots

            action = self.select_action(slot_history, sample=True)
            next_image, reward, done = self.env.step(action)


            print("reward:", reward)
            print("len(reward):", len(reward))
            input()

            for j in range(len(reward)):
                images.append(image[j].unsqueeze(0))
                # Ensure action maintains at least one dimension
                actions.append(action[:, j].view(-1).cpu())
                rewards.append(reward[j])
                is_first.append(False)
                total_reward += reward[j]

        self.replay_buffer.append(images, actions, rewards, is_first)
        self.env.close()

        self.writer.add_scalar(
            name="Rewards/train_rewards",
            val=np.sum(rewards),
            step=self.episode + 1
        )

    def select_action(self, slot_history: torch.Tensor, sample: bool = False) -> np.ndarray:
        action_dist = self.actor(slot_history, start=slot_history.shape[1] - 1)
        selected_action = action_dist.sample().squeeze() if sample else action_dist.mode.squeeze()
        return selected_action.clamp_(self.env.action_space.low[0], self.env.action_space.high[0]).cpu().numpy()

    def _finetune_savi(self) -> None:
        for iteration in range(10):
            videos, actions, _, _ = self.replay_buffer.sample(self.savi.batch_size, self.savi.sequence_length)
            self.savi_finetuning_optimizer.zero_grad()
            reconstruction_loss = self.savi.compute_reconstruction_loss((videos, actions))
            self.log("reconstruction_loss", reconstruction_loss.item())
            reconstruction_loss.backward()
            # Apply gradient clipping as before.
            torch.nn.utils.clip_grad_norm_(self.savi.parameters(), self.savi.trainer.gradient_clip_val)
            self.savi_finetuning_optimizer.step()
