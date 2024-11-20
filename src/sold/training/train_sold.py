import hydra
from omegaconf import DictConfig
from sold.utils.train import seed_everything, instantiate_trainer
from abc import ABC, abstractmethod

from functools import partial
from typing import Any, Callable, Iterable, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT, TRAIN_DATALOADERS
from lightning import LightningModule

import sold
from sold.models.savi.model import SAVi
from sold.utils.replay_buffer import ExperienceReplay
from sold.models.sold.prediction import GaussianPredictor, TwoHotPredictor
from sold.datasets.experience_source import ExperienceSourceDataset, DummyValidationDataset
import numpy as np
from torchvision import transforms
from sold.models.sold.dynamics import DynamicsModel, AutoregressiveWrapper, TokenWiseAutoregressiveWrapper
from sold.envs import load_env
from sold.envs.image_env import ImageEnv
from sold.training.train_savi import SAViTrainer
from sold.replay_buffer.datasets.ring_buffer import RingBufferDataset
from torch.optim import Optimizer
from sold.utils.distributions import TwoHotEncodingDistribution, Moments
import copy
from torch.distributions import Distribution


class OnlineTrainer(LightningModule, ABC):
    """"""
    def __init__(
            self,
            env: ImageEnv,
            batch_size: int = 16,
            sequence_length: int = 16,
            collect_interval: int = 10,
            num_seed_episodes: int = 10,
            num_validation_episodes: int = 10
    ) -> None:
        """ Creates an online RL trainer that handles the integration of training and experience collection.

        Args:
            collect_interval (int): Collect a new episode after this many training steps.
        """
        super().__init__()
        self.env = env
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.collect_interval = collect_interval
        self.num_seed_episodes = num_seed_episodes
        self.num_validation_episodes = num_validation_episodes

        self.validation_returns = []
        self.validation_successes = []

    @property
    def current_episode(self) -> int:
        return self.current_epoch + self.num_seed_episodes

    def train_dataloader(self) -> DataLoader:
        self.replay_buffer = RingBufferDataset(int(1e7), self.batch_size, self.sequence_length,
                                               save_path=self.logger.log_dir + "/replay_buffer")

        # self.replay_buffer = ExperienceReplay(
        #     self.logger.log_dir + "/replay_buffer", int(1e6), self.env.max_episode_steps // self.env.action_repeat,
        #     self.env.image_size, self.env.action_space.shape[0])

        for episode_index in range(self.num_seed_episodes):
            self.collect_episode(mode="random")

        dataset = ExperienceSourceDataset(
            self.replay_buffer.sample,
            collect_interval=self.collect_interval)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, num_workers=1)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return DataLoader(DummyValidationDataset(self.num_validation_episodes), batch_size=1)

    @abstractmethod
    def select_action(self, observation: torch.Tensor, is_first: bool = False, sample: bool = False) -> np.ndarray:
        pass

    @torch.no_grad()
    def collect_episode(self, mode: str = "random", store: bool = True):
        """
        Collect an episode by interacting with the environment.

        Args:
            mode (str): Mode for action selection. Can be "deterministic", "sample", or "random".
        """

        image, is_first = self.env.reset(), True
        image = transforms.ToTensor()(image.copy()).to(self.device)  # Expand batch dimension.

        images, rewards = [], []
        images.append(image.cpu().unsqueeze(0))
        rewards.append(0.0)

        if store:
            self.replay_buffer.add_step({"image": image, "action": torch.zeros(self.env.action_space.shape[0]), "reward": 0.0, "is_first": is_first})

        done = False
        while not done:
            if mode == "random":
                action = self.env.action_space.sample()
            else:
                action = self.select_action(image, is_first=is_first, sample=mode == "sample")

            image, reward, done = self.env.step(action)
            is_first = False
            image = transforms.ToTensor()(image.copy()).to(self.device)  # Expand batch dimension.
            images.append(image.cpu().unsqueeze(0))
            rewards.append(reward)

            if store:
                self.replay_buffer.add_step({"image": image, "action": torch.tensor(action), "reward": reward, "is_first": is_first}, done=done)

        success = self.env.success()
        self.env.close()
        return images, rewards, success

    def on_train_epoch_end(self) -> None:
        """Collect new episode after every epoch of training."""
        images, rewards, success = self.collect_episode(mode="sample")
        self.log("train/episode_return", np.sum(rewards), prog_bar=True)
        self.log("current_episode", self.current_episode)

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Collect validation episodes (Episodes that are not added to the replay buffer or used for training and where
        the actor does not explore)."""
        images, rewards, success = self.collect_episode(mode="deterministic", store=False)
        self.validation_returns.append(np.sum(rewards))
        self.validation_successes.append(success)
        return {"images": images, "rewards": rewards}

    def on_validation_epoch_end(self) -> None:
        self.log("validation/episode_return", np.mean(self.validation_returns), prog_bar=True)
        self.log("validation/success_rate", np.mean(self.validation_successes), prog_bar=True)
        self.validation_returns.clear()
        self.validation_successes.clear()


class SOLDTrainer(OnlineTrainer):
    def __init__(self, savi: SAVi, env: ImageEnv,
                 dynamics_predictor: partial[DynamicsModel],
                 actor: partial[GaussianPredictor], critic: partial[TwoHotPredictor],
                 reward_predictor: partial[TwoHotPredictor], learning_rate: float, collect_interval: int) -> None:
        super().__init__(env)
        self.automatic_optimization = False

        regression_infos = {"max_episode_steps": env.max_episode_steps,  "num_slots": savi.corrector.num_slots,
                            "slot_dim": savi.corrector.slot_dim}
        self.savi = savi
        self.actor = actor(**regression_infos, output_dim=env.action_space.shape[0])
        self.critic = critic(**regression_infos)
        self.critic_target = copy.deepcopy(self.critic)
        self.reward_predictor = reward_predictor(**regression_infos)

        self.dynamics_predictor = AutoregressiveWrapper(
            dynamics_predictor(
                num_slots=self.savi.num_slots, slot_dim=self.savi.slot_dim, sequence_length=15,
                action_dim=env.action_space.shape[0]))

        self.collect_interval = collect_interval
        self.learning_rate = learning_rate

        self.num_context = 1
        self.num_predictions = 15

        self.finetune_savi = False
        self.imagination_horizon = 15

        self.action_lambda = 0.95
        self.discount_factor = 0.96
        self.critic_ema_decay = 0.98

        self.return_moments = Moments()
        self.register_buffer("discounts", torch.full((1, self.imagination_horizon), self.discount_factor))
        self.discounts = torch.cumprod(self.discounts, dim=1) / self.discount_factor

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return [torch.optim.Adam(self.dynamics_predictor.parameters(), lr=self.learning_rate),
                torch.optim.Adam(self.reward_predictor.parameters(), lr=self.learning_rate),
                torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate),
                torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)]

    def training_step(self, batch, batch_index: int) -> STEP_OUTPUT:
        dynamics_optimizer, reward_optimizer, actor_optimizer, critic_optimizer = self.optimizers()

        images, actions, rewards, is_firsts = batch
        outputs = SAViTrainer.compute_reconstruction_loss(self, images)

        if self.finetune_savi:
            assert False
            # self.savi_optimizer.zero_grad()
            # reconstruction_loss = self.savi.compute_reconstruction_loss(images, log_visualizations=batch_index == 0,
            #                                                             logger=self.logger)
            # self.log("reconstruction_loss", reconstruction_loss.item())
            # reconstruction_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.savi.parameters(), 0.05)
            # self.savi_optimizer.step()

        with torch.no_grad():
            slots = self.savi(images, reconstruct=False)

        # Learn to predict dynamics in slot-space.
        dynamics_optimizer.zero_grad()
        outputs |= self.compute_dynamics_loss(images, slots, actions)
        self.manual_backward(outputs["dynamics_loss"])
        self.clip_gradients(dynamics_optimizer, gradient_clip_val=0.05, gradient_clip_algorithm="norm")
        dynamics_optimizer.step()

        # Learn to predict rewards from slot representation.
        reward_optimizer.zero_grad()
        outputs |= self.compute_reward_loss(slots, rewards, is_firsts)
        self.manual_backward(outputs["reward_loss"])
        self.clip_gradients(reward_optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        reward_optimizer.step()

        # Perform latent imagination to learn the actor and critic.
        lambda_returns, predicted_values_targ, predicted_values, action_entropies = self.imagine_ahead(slots, actions)

        # Learn the actor.
        actor_optimizer.zero_grad()
        outputs |= self.compute_actor_loss(lambda_returns, action_entropies)
        self.manual_backward(outputs["actor_loss"])
        self.clip_gradients(actor_optimizer, gradient_clip_val=1, gradient_clip_algorithm="norm")
        actor_optimizer.step()

        # Learn the critic.
        critic_optimizer.zero_grad()
        outputs |= self.compute_critic_loss(lambda_returns, predicted_values, predicted_values_targ)
        self.manual_backward(outputs["critic_loss"])
        self.clip_gradients(critic_optimizer, gradient_clip_val=1, gradient_clip_algorithm="norm")
        critic_optimizer.step()

        # Log all losses.
        for key, value in outputs.items():
            if key.endswith("_loss"):
                self.log(f"train/{key}", value.item())
        return outputs

    def compute_dynamics_loss(self, images: torch.Tensor, slots: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
        batch_size, sequence_length, num_slots, slot_dim = slots.shape
        context_slots = slots[:, :self.num_context].detach()
        future_slots = self.dynamics_predictor.predict_slots(self.num_predictions, context_slots, actions[:, 1:].clone().detach())
        predicted_slots = torch.cat([context_slots, future_slots], dim=1)

        predicted_rgbs, predicted_masks = self.savi.decoder(predicted_slots.flatten(end_dim=1))
        predicted_rgbs = predicted_rgbs.reshape(batch_size, sequence_length, num_slots, 3, *self.env.image_size)
        predicted_masks = predicted_masks.reshape(batch_size, sequence_length, num_slots, 1, *self.env.image_size)
        predicted_images = torch.clamp(torch.sum(predicted_rgbs * predicted_masks, dim=2), 0., 1.)

        slot_loss = F.mse_loss(predicted_slots[:, self.num_context:], slots[:, self.num_context:])
        image_loss = F.mse_loss(predicted_images[:, self.num_context:], images[:, self.num_context:])
        return {"slot_loss": slot_loss, "image_loss": image_loss, "dynamics_loss": slot_loss + image_loss,
                "predicted_images": predicted_images, "predicted_rgbs": predicted_rgbs,
                "predicted_masks": predicted_masks}

    def compute_reward_loss(self, slots: torch.Tensor, rewards: torch.Tensor, is_firsts: torch.Tensor) -> Dict[str, Any]:
        predicted_rewards = TwoHotEncodingDistribution(self.reward_predictor(slots.detach().clone()), dims=1)
        log_probs = predicted_rewards.log_prob(rewards.detach().unsqueeze(2))
        masked_log_probs = log_probs[~is_firsts]
        return {"reward_loss": -masked_log_probs.mean(), "rewards": rewards, "predicted_rewards": predicted_rewards.mean.squeeze(2)}

    def imagine_ahead(self, slots: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:

        for vp, tvp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tvp.data.copy_((1 - self.critic_ema_decay) * vp.data + self.critic_ema_decay * tvp.data)

        batch_size, sequence_length, num_slots, slot_dim = slots.shape

        action_entropies = []
        # randomly sample starting states (and their corresponding actions)
        start_index = torch.randint(self.num_context, sequence_length, (1,)).item()
        slot_history = slots[:, :start_index].detach()
        action_history = actions[:, 1:start_index].detach()
        # Actor update
        # Freeze models except action model and imagine next states
        with self.FreezeActor(self.reward_predictor, self.critic):
            for t in range(self.imagination_horizon):
                # select actions
                action_dist = self.actor(slot_history.detach(), start=slot_history.shape[1] - 1)
                selected_action = action_dist.rsample().squeeze(1)
                # clip action
                #action_clip = torch.full_like(selected_action, self.max_action)
                # selected_action = selected_action * (
                #             action_clip / torch.maximum(action_clip, torch.abs(selected_action))).detach()

                action_history = torch.cat([action_history, selected_action.unsqueeze(1)], dim=1)
                # save entropy
                action_entropies.append(action_dist.entropy())
                # predict states
                predicted_slots = self.dynamics_predictor.predict_slots(1, slot_history, action_history)
                slot_history = torch.cat([slot_history, predicted_slots], dim=1)

            predicted_rewards = TwoHotEncodingDistribution(self.reward_predictor(slot_history, start=start_index), dims=1).mean.squeeze()
            predicted_values = TwoHotEncodingDistribution(self.critic(slot_history, start=start_index), dims=1).mean.squeeze()

        lambda_returns = self.compute_value_estimates(predicted_rewards, predicted_values)

        action_entropies = torch.stack(action_entropies, dim=1)

        # Value update
        slot_history = slot_history.detach()
        # predict imagined values
        predicted_values_targ = TwoHotEncodingDistribution(self.critic_target(slot_history[:, :-1], start=start_index - 1),
                                                   dims=1).mean.squeeze()
        predicted_values = TwoHotEncodingDistribution(self.critic(slot_history[:, :-1], start=start_index - 1), dims=1)

        return lambda_returns, predicted_values_targ, predicted_values, action_entropies

    def compute_actor_loss(self, lambda_returns: torch.Tensor, action_entropies: torch.Tensor) -> Dict[str, Any]:
        entropy_loss_weight = 0.001
        _, invscale = self.return_moments(lambda_returns)
        norm_value_estimates = lambda_returns / invscale

        actor_return_loss = -torch.mean(self.discounts.detach() * norm_value_estimates)
        actor_entropy_loss = -torch.mean(self.discounts.detach() * action_entropies)
        return {"actor_loss": actor_return_loss + entropy_loss_weight * actor_entropy_loss, "actor_return_loss": actor_return_loss,
                "actor_entropy_loss": entropy_loss_weight * actor_entropy_loss}

    def compute_critic_loss(self, lambda_returns: torch.Tensor, predicted_values: Distribution, predicted_values_targ: torch.Tensor) -> Dict[str, Any]:
        regularization_loss_weight = 0.1
        return_loss = torch.mean(self.discounts.detach() * (-predicted_values.log_prob(lambda_returns.detach().unsqueeze(2))))
        target_regularization_loss = torch.mean(self.discounts.detach() * (-predicted_values.log_prob(predicted_values_targ.detach().unsqueeze(2))))
        return {"critic_loss": return_loss + regularization_loss_weight * target_regularization_loss, "critic_return_loss": return_loss,
                "critic_target_regularization_loss": regularization_loss_weight * target_regularization_loss}

    class FreezeActor:
        def __init__(self, reward_model, value_model):
            self.reward_model = reward_model
            self.value_model = value_model

        def __enter__(self):
            for param in list(self.reward_model.parameters()) + list(self.value_model.parameters()):
                param.requires_grad = False

        def __exit__(self, exc_type, exc_val, exc_tb):
            for param in list(self.reward_model.parameters()) + list(self.value_model.parameters()):
                param.requires_grad = True

    def compute_value_estimates(self, rewards, values):
        vals = [values[:, -1:]]
        interm = rewards + self.discount_factor * values * (1 - self.action_lambda)
        for t in reversed(range(self.imagination_horizon)):
            vals.append(interm[:, t].unsqueeze(1) + self.discount_factor * self.action_lambda * vals[-1])
        ret = torch.cat(list(reversed(vals)), dim=1)[:, :-1]
        return ret

    def select_action(self, observation: torch.Tensor, is_first: bool = False, sample: bool = False) -> np.ndarray:
        observation = observation.unsqueeze(0)  # Expand batch dimension (1, 3, 64, 64).

        # Encode image into slots and append to context.
        last_slots = None if is_first else self._slot_history[:, -1]
        step_offset = 0 if is_first else 1
        slots = self.savi(observation.unsqueeze(1), prior_slots=last_slots, step_offset=step_offset, reconstruct=False)  # Expand sequence dimension on image.
        self._slot_history = slots if is_first else torch.cat([self._slot_history, slots], dim=1)

        # Query actor with slot history.
        action_dist = self.actor(self._slot_history, start=self._slot_history.shape[1] - 1)
        selected_action = action_dist.sample().squeeze() if sample else action_dist.mode.squeeze()
        return selected_action.clamp_(self.env.action_space.low[0], self.env.action_space.high[0]).cpu().numpy()


@hydra.main(config_path="../configs", config_name="sold")
def train(cfg: DictConfig):
    seed_everything(cfg.experiment.seed)
    sold = hydra.utils.instantiate(cfg.model)
    trainer = instantiate_trainer(cfg)
    trainer.fit(sold)


if __name__ == "__main__":
    train()
