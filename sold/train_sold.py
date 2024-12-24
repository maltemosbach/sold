import os
os.environ["HYDRA_FULL_ERROR"] = "1"
from collections import defaultdict
import copy
from functools import partial
import gymnasium as gym
import hydra
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from modeling.distributions import TwoHotEncodingDistribution, Moments
from modeling.savi.model import SAVi
from modeling.sold.dynamics import OCVPSeqDynamicsModel
from modeling.sold.prediction import GaussianPredictor, TwoHotPredictor
import numpy as np
from omegaconf import DictConfig
import torch
from torch.distributions import Distribution
import torch.nn.functional as F
from train_savi import SAViModule
from typing import Any, Dict, Tuple
from utils.instantiate import instantiate_trainer
from utils.module import FreezeParameters
from utils.training import set_seed, OnlineModule
from utils.visualization import visualize_dynamics_prediction, visualize_savi_decomposition, visualize_reward_prediction, visualize_output_attention, visualize_reward_predictor_attention, get_attention_weights


class SOLDModule(OnlineModule):
    def __init__(self, savi: SAVi, dynamics_predictor: partial[OCVPSeqDynamicsModel],
                 actor: partial[GaussianPredictor], critic: partial[TwoHotPredictor],
                 reward_predictor: partial[TwoHotPredictor], dynamics_learning_rate: float, dynamics_grad_clip: float,
                 actor_learning_rate: float, actor_grad_clip: float, critic_learning_rate: float,
                 critic_grad_clip: float, reward_learning_rate: float, reward_grad_clip: float,
                 finetune_savi: bool, savi_learning_rate: float, savi_grad_clip: float, num_context: Tuple[int, int],
                 imagination_horizon: int, start_imagination_from_every: bool, actor_entropy_loss_weight: float,
                 actor_gradients: str, return_lambda: float, discount_factor: float, critic_ema_decay: float,
                 env: gym.Env, max_steps: int, num_seed: int, update_freq: int, num_updates: int, eval_freq: int,
                 num_eval_episodes: int, batch_size: int, buffer_capacity: int) -> None:
        sequence_length = imagination_horizon + num_context[1]

        super().__init__(env, max_steps, num_seed, update_freq, num_updates, eval_freq, num_eval_episodes, batch_size,
                         sequence_length, buffer_capacity)
        self.automatic_optimization = False
        self.save_hyperparameters(logger=False, ignore=['savi', 'env'])

        regression_infos = {"max_episode_steps": env.max_episode_steps,  "num_slots": savi.corrector.num_slots,
                            "slot_dim": savi.corrector.slot_dim}
        self.savi = savi
        self.actor = actor(**regression_infos, output_dim=env.action_space.shape[0], lower_bound=env.action_space.low, upper_bound=env.action_space.high)
        self.critic = critic(**regression_infos)
        self.critic_target = copy.deepcopy(self.critic)
        self.reward_predictor = reward_predictor(**regression_infos)
        self.dynamics_predictor = dynamics_predictor(
                num_slots=self.savi.num_slots, slot_dim=self.savi.slot_dim, sequence_length=15,
                action_dim=env.action_space.shape[0], input_buffer_size=sequence_length)

        self.dynamics_learning_rate = dynamics_learning_rate
        self.dynamics_grad_clip = dynamics_grad_clip
        self.actor_learning_rate = actor_learning_rate
        self.actor_grad_clip = actor_grad_clip
        self.actor_entropy_loss_weight = actor_entropy_loss_weight
        self.actor_gradients = actor_gradients
        self.critic_learning_rate = critic_learning_rate
        self.critic_grad_clip = critic_grad_clip
        self.reward_learning_rate = reward_learning_rate
        self.reward_grad_clip = reward_grad_clip

        self.finetune_savi = finetune_savi
        self.savi_grad_clip = savi_grad_clip

        self.min_num_context, self.max_num_context = num_context
        if self.min_num_context > self.max_num_context:
            raise ValueError("min_num_context must be less than or equal to max_num_context.")
        self.imagination_horizon = imagination_horizon
        self.start_imagination_from_every = start_imagination_from_every
        self.return_lambda = return_lambda
        self.discount_factor = discount_factor
        self.critic_ema_decay = critic_ema_decay

        self.return_moments = Moments()
        self.register_buffer("discounts", torch.full((1, self.imagination_horizon), self.discount_factor))
        self.discounts = torch.cumprod(self.discounts, dim=1) / self.discount_factor
        self.savi_optimizer = torch.optim.Adam(self.savi.parameters(), lr=savi_learning_rate)
        self.current_losses = defaultdict(list)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return [torch.optim.Adam(self.dynamics_predictor.parameters(), lr=self.dynamics_learning_rate),
                torch.optim.Adam(self.reward_predictor.parameters(), lr=self.reward_learning_rate),
                torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate),
                torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)]

    def training_step(self, batch, batch_index: int) -> STEP_OUTPUT:
        dynamics_optimizer, reward_optimizer, actor_optimizer, critic_optimizer = self.optimizers()
        images, actions, rewards = batch["obs"].squeeze(0), batch["action"].squeeze(0), batch["reward"].squeeze(0)

        if self.finetune_savi:
            self.savi_optimizer.zero_grad()
        outputs = SAViModule.compute_reconstruction_loss(self, images, actions)
        if self.finetune_savi:
            outputs["reconstruction_loss"].backward()
            self.clip_gradients(self.savi_optimizer, gradient_clip_val=self.savi_grad_clip, gradient_clip_algorithm="norm")
            self.savi_optimizer.step()

        if self.after_eval:
            savi_image = visualize_savi_decomposition(images[0], outputs["reconstructions"][0], outputs["rgbs"][0], outputs["masks"][0])
            self.log("savi_decomposition", savi_image)

        # Detach slots to prevent gradients from flowing back to the SAVi model.
        slots = outputs["slots"].detach()

        # Learn to predict dynamics in slot-space.
        dynamics_optimizer.zero_grad()
        outputs |= self.compute_dynamics_loss(images, slots, actions)
        self.manual_backward(outputs["dynamics_loss"])
        self.clip_gradients(dynamics_optimizer, gradient_clip_val=self.dynamics_grad_clip, gradient_clip_algorithm="norm")
        dynamics_optimizer.step()

        # Learn to predict rewards from slot representation.
        reward_optimizer.zero_grad()
        outputs |= self.compute_reward_loss(images, outputs["reconstructions"], slots, rewards)
        self.manual_backward(outputs["reward_loss"])
        self.clip_gradients(reward_optimizer, gradient_clip_val=self.reward_grad_clip, gradient_clip_algorithm="norm")
        reward_optimizer.step()

        # Update the target critic network.
        for critic_param, critic_target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            critic_target_param.data.copy_((1 - self.critic_ema_decay) * critic_param.data + self.critic_ema_decay * critic_target_param.data)

        # Perform latent imagination to train the actor and critic.
        lambda_returns, predicted_values_targ, predicted_values, action_log_probs, action_entropies = self.imagine_ahead(slots, actions)

        # Learn the actor.
        actor_optimizer.zero_grad()
        outputs |= self.compute_actor_loss(lambda_returns, predicted_values_targ, action_log_probs, action_entropies)
        self.manual_backward(outputs["actor_loss"])
        self.clip_gradients(actor_optimizer, gradient_clip_val=self.actor_grad_clip, gradient_clip_algorithm="norm")
        actor_optimizer.step()

        # Learn the critic.
        critic_optimizer.zero_grad()
        outputs |= self.compute_critic_loss(lambda_returns, predicted_values, predicted_values_targ)
        self.manual_backward(outputs["critic_loss"])
        self.clip_gradients(critic_optimizer, gradient_clip_val=self.critic_grad_clip, gradient_clip_algorithm="norm")
        critic_optimizer.step()

        # Log all losses.
        for key, value in outputs.items():
            if key.endswith("_loss"):
                self.log("train/" + key, value)
        self.log_gradients(model_names=("reward_predictor", "actor", "critic"))
        return outputs

    def compute_dynamics_loss(self, images: torch.Tensor, slots: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
        batch_size, sequence_length, num_slots, slot_dim = slots.shape
        num_context = torch.randint(self.min_num_context, self.max_num_context + 1, (1,)).item()
        context_slots = slots[:, :num_context].detach()
        future_slots = self.dynamics_predictor.predict_slots(slots, actions[:, 1:].clone().detach(), steps=self.imagination_horizon, num_context=num_context)
        predicted_slots = torch.cat([context_slots, future_slots], dim=1)

        predicted_rgbs, predicted_masks = self.savi.decoder(predicted_slots.flatten(end_dim=1))
        predicted_rgbs = predicted_rgbs.reshape(batch_size, num_context + self.imagination_horizon, num_slots, 3, *self.env.image_size)
        predicted_masks = predicted_masks.reshape(batch_size, num_context + self.imagination_horizon, num_slots, 1, *self.env.image_size)
        predicted_images = torch.clamp(torch.sum(predicted_rgbs * predicted_masks, dim=2), 0., 1.)

        slot_loss = F.mse_loss(predicted_slots[:, num_context:], slots[:, num_context:num_context + self.imagination_horizon])
        image_loss = F.mse_loss(predicted_images[:, num_context:], images[:, num_context:num_context + self.imagination_horizon])

        if self.after_eval:
            dynamics_image = visualize_dynamics_prediction(predicted_images[0], predicted_rgbs[0], predicted_masks[0], num_context, images[0, :num_context + self.imagination_horizon])
            self.log("dynamics_prediction", dynamics_image)

        return {"slot_loss": slot_loss, "image_loss": image_loss, "dynamics_loss": slot_loss + image_loss, "predicted_images": predicted_images}

    def compute_reward_loss(self, images: torch.Tensor, reconstructions: torch.Tensor, slots: torch.Tensor, rewards: torch.Tensor) -> Dict[str, Any]:
        is_firsts = torch.isnan(rewards)  # We add NaN as a reward on the first time-step.
        predicted_rewards = TwoHotEncodingDistribution(self.reward_predictor(slots.detach().clone()), dims=1)
        log_probs = predicted_rewards.log_prob(rewards.detach().unsqueeze(2))
        masked_log_probs = log_probs[~is_firsts]

        # Log visualizations related to reward prediction.
        if self.after_eval:
            with torch.no_grad():
                # Log prediction vs ground truth reward over the sequence.
                reward_image = visualize_reward_prediction(
                    images[0], reconstructions[0], rewards[0],
                    predicted_rewards.mean.squeeze(2)[0])
                self.log("reward_prediction", reward_image)

                # Log visualization of reward predictor attention to inspect reward-predictive elements.
                output_weights = get_attention_weights(self.reward_predictor, slots[:1,])
                predicted_rgbs, predicted_masks = self.savi.decoder(
                    slots[:1].flatten(end_dim=1))
                attention_image = visualize_reward_predictor_attention(images[0], reconstructions[0], rewards[0], predicted_rewards.mean.squeeze(2)[0], output_weights, predicted_rgbs, predicted_masks)
                self.log("reward_predictor_attention", attention_image)

        return {"reward_loss": -masked_log_probs.mean()}

    def imagine_ahead(self, slots: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, num_slots, slot_dim = slots.shape
        action_log_probs, action_entropies = [], []

        if self.start_imagination_from_every:
            num_context = self.max_num_context
            slots_context = slots.unfold(dimension=1, size=self.max_num_context, step=1).flatten(end_dim=1).permute(0, 3, 1, 2)
            actions_context = actions.unfold(dimension=1, size=self.max_num_context, step=1).flatten(end_dim=1).permute(0, 2, 1)[:, 1:]
        else:
            #num_context = torch.randint(self.min_num_context, self.max_num_context + 1, (1,)).item()
            num_context = self.max_num_context
            slots_context = slots[:, :num_context].detach()
            actions_context = actions[:, 1:num_context].detach()

        # Actor update
        # Freeze models except action model and imagine next states
        with FreezeParameters([self.reward_predictor, self.critic]):
            for t in range(self.imagination_horizon):
                action_dist = self.actor(slots_context.detach(), start=slots_context.shape[1] - 1)
                selected_action = action_dist.rsample().squeeze(1)
                actions_context = torch.cat([actions_context, selected_action.unsqueeze(1)], dim=1)
                action_log_probs.append(action_dist.log_prob(selected_action.unsqueeze(1)))
                action_entropies.append(action_dist.entropy())

                predicted_slots = self.dynamics_predictor.predict_slots(slots_context, actions_context, steps=1, num_context=slots_context.shape[1])
                slots_context = torch.cat([slots_context, predicted_slots], dim=1)

        with FreezeParameters([self.reward_predictor, self.critic]):
            predicted_rewards = TwoHotEncodingDistribution(self.reward_predictor(slots_context, start=num_context), dims=1).mean.squeeze()
            predicted_values = TwoHotEncodingDistribution(self.critic(slots_context, start=num_context), dims=1).mean.squeeze()

        lambda_returns = self.compute_lambda_returns(predicted_rewards, predicted_values)

        action_log_probs = torch.stack(action_log_probs, dim=1).squeeze(2)
        action_entropies = torch.stack(action_entropies, dim=1)

        # Value update
        slots_context = slots_context.detach()
        # Predict imagined values
        predicted_values_targ = TwoHotEncodingDistribution(self.critic_target(slots_context[:, :-1], start=num_context - 1),
                                                   dims=1).mean.squeeze()
        predicted_values = TwoHotEncodingDistribution(self.critic(slots_context[:, :-1], start=num_context - 1), dims=1)

        if self.after_eval:
            with torch.no_grad():
                # Log visualization of a latent imagination sequence.
                predicted_rgbs, predicted_masks = self.savi.decoder(slots_context[0])
                predicted_rgbs = predicted_rgbs.reshape(1, -1, num_slots, 3, *self.env.image_size)
                predicted_masks = predicted_masks.reshape(1, -1, num_slots, 1, *self.env.image_size)
                predicted_images = torch.clamp(torch.sum(predicted_rgbs * predicted_masks, dim=2), 0., 1.)
                imagination_image = visualize_dynamics_prediction(predicted_images[0], predicted_rgbs[0], predicted_masks[0], num_context)
                self.log("latent_imagination", imagination_image)

                # Log visualization of actor attention.
                output_weights = get_attention_weights(self.actor, slots_context[:1, :num_context + self.imagination_horizon])
                actor_attention_image = visualize_output_attention(output_weights, predicted_rgbs[0], predicted_masks[0])
                self.log("actor_attention", actor_attention_image)
        return lambda_returns, predicted_values_targ, predicted_values, action_log_probs, action_entropies

    def compute_actor_loss(self, lambda_returns: torch.Tensor, predicted_values_targ: torch.Tensor,
                           action_log_probs: torch.Tensor, action_entropies: torch.Tensor) -> Dict[str, Any]:
        # Compute advantage estimates.
        offset, invscale = self.return_moments(lambda_returns[:, :-1])
        normed_lambda_returns = (lambda_returns[:, :-1] - offset) / invscale
        normed_base = (predicted_values_targ[:, :-1] - offset) / invscale
        advantage = normed_lambda_returns - normed_base

        if self.actor_gradients == "dynamics":
            actor_return_loss = -torch.mean(self.discounts.detach()[:, :-1] * advantage)
        elif self.actor_gradients == "reinforce":
            actor_return_loss = torch.mean(action_log_probs[:, :-1] * advantage.detach())
        else:
            raise ValueError(f"Invalid actor_gradients: {self.actor_gradients}.")

        actor_entropy_loss = -torch.mean(self.discounts.detach() * action_entropies)
        return {"actor_loss": actor_return_loss + self.actor_entropy_loss_weight * actor_entropy_loss, "actor_return_loss": actor_return_loss,
                "actor_entropy_loss": self.actor_entropy_loss_weight * actor_entropy_loss}

    def compute_critic_loss(self, lambda_returns: torch.Tensor, predicted_values: Distribution, predicted_values_targ: torch.Tensor) -> Dict[str, Any]:
        regularization_loss_weight = 0.1
        return_loss = torch.mean(self.discounts.detach() * (-predicted_values.log_prob(lambda_returns.detach().unsqueeze(2))))
        target_regularization_loss = torch.mean(self.discounts.detach() * (-predicted_values.log_prob(predicted_values_targ.detach().unsqueeze(2))))
        return {"critic_loss": return_loss + regularization_loss_weight * target_regularization_loss, "critic_return_loss": return_loss,
                "critic_target_regularization_loss": regularization_loss_weight * target_regularization_loss}

    def compute_lambda_returns(self, rewards, values):
        vals = [values[:, -1:]]
        interm = rewards + self.discount_factor * values * (1 - self.return_lambda)
        for t in reversed(range(self.imagination_horizon)):
            vals.append(interm[:, t].unsqueeze(1) + self.discount_factor * self.return_lambda * vals[-1])
        ret = torch.cat(list(reversed(vals)), dim=1)[:, :-1]
        return ret

    def select_action(self, observation: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        observation = observation.unsqueeze(0)  # Expand batch dimension (1, 3, height, width).

        # Encode image into slots and append to context.
        last_slots = None if is_first else self._slot_history[:, -1]
        step_offset = 0 if is_first else 1
        slots = self.savi(observation.unsqueeze(1), self.last_action.unsqueeze(0).unsqueeze(1), prior_slots=last_slots,
                          step_offset=step_offset, reconstruct=False)  # Expand sequence (and batch) dimension.
        self._slot_history = slots if is_first else torch.cat([self._slot_history, slots], dim=1)

        if mode == "random":
            selected_action = torch.from_numpy(self.env.action_space.sample().astype(np.float32))
        else:
            action_dist = self.actor(self._slot_history, start=self._slot_history.shape[1] - 1)
            if mode == "train":
                selected_action = action_dist.sample().squeeze()
            elif mode == "eval":
                selected_action = action_dist.mode.squeeze()
            else:
                raise ValueError(f"Invalid mode: {mode}")

        return selected_action.clamp_(self.env.action_space.low[0], self.env.action_space.high[0]).detach()


@hydra.main(config_path="../configs", config_name="train_sold", version_base=None)
def train(cfg: DictConfig):
    if cfg.logger.log_to_wandb:
        import wandb
        wandb.init(project="sold", config=dict(cfg), sync_tensorboard=True)

    set_seed(cfg.seed)
    sold = hydra.utils.instantiate(cfg.model)
    trainer = instantiate_trainer(cfg)
    trainer.fit(sold, ckpt_path=os.path.abspath(cfg.checkpoint) if cfg.checkpoint else None)

    if cfg.logger.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
