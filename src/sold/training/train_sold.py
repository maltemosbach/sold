import gym
import hydra
from omegaconf import DictConfig
from sold.utils.instantiate import instantiate_trainer
from sold.utils.training import set_seed
from functools import partial
import numpy as np
import os
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from sold.modeling.savi.model import SAVi
from sold.modeling.sold.prediction import GaussianPredictor, TwoHotPredictor
from sold.modeling.sold.dynamics import OCVPSeqDynamicsModel
from sold.training.train_savi import SAViTrainer
from sold.utils.module import FreezeParameters
from sold.utils.training import OnlineModule
from sold.modeling.distributions import TwoHotEncodingDistribution, Moments
import copy
from torch.distributions import Distribution
from sold.utils.visualization import visualize_dynamics_prediction, visualize_savi_decomposition, visualize_reward_prediction, AttentionWeightsHook, patch_attention, visualize_output_attention, visualize_reward_predictor_attention, get_attention_weights


class SOLDTrainer(OnlineModule):
    def __init__(self, env: gym.Env, savi: SAVi, dynamics_predictor: partial[OCVPSeqDynamicsModel],
                 actor: partial[GaussianPredictor], critic: partial[TwoHotPredictor],
                 reward_predictor: partial[TwoHotPredictor], learning_rate: float, num_context: Tuple[int, int],
                 imagination_horizon: int, finetune_savi: bool, return_lambda: float, discount_factor: float,
                 critic_ema_decay: float, train_after: int, update_freq: int, num_updates: int, eval_freq: int,
                 num_eval_episodes: int, batch_size: int, buffer_capacity: int, interval: str) -> None:
        sequence_length = imagination_horizon + num_context[1]

        super().__init__(env, train_after, update_freq, num_updates, eval_freq, num_eval_episodes, batch_size,
                         sequence_length, buffer_capacity, interval)
        self.automatic_optimization = False

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

        self.learning_rate = learning_rate

        self.min_num_context, self.max_num_context = num_context
        if self.min_num_context > self.max_num_context:
            raise ValueError("min_num_context must be less than or equal to max_num_context.")

        self.imagination_horizon = imagination_horizon
        self.finetune_savi = finetune_savi
        self.return_lambda = return_lambda
        self.discount_factor = discount_factor
        self.critic_ema_decay = critic_ema_decay

        self.savi_grad_clip = 0.05
        self.prediction_grad_clip = 3.0
        self.rl_grad_clip = 10.0

        self.return_moments = Moments()
        self.register_buffer("discounts", torch.full((1, self.imagination_horizon), self.discount_factor))
        self.discounts = torch.cumprod(self.discounts, dim=1) / self.discount_factor

        self.savi_optimizer = torch.optim.Adam(self.savi.parameters(), lr=self.learning_rate)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return [torch.optim.Adam(self.dynamics_predictor.parameters(), lr=self.learning_rate),
                torch.optim.Adam(self.reward_predictor.parameters(), lr=self.learning_rate),
                torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate),
                torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)]

    def training_step(self, batch, batch_index: int) -> STEP_OUTPUT:
        dynamics_optimizer, reward_optimizer, actor_optimizer, critic_optimizer = self.optimizers()
        images, actions, rewards = batch["obs"], batch["action"], batch["reward"]

        if self.finetune_savi:
            self.savi_optimizer.zero_grad()
        outputs = SAViTrainer.compute_reconstruction_loss(self, images, actions)
        if self.finetune_savi:
            outputs["reconstruction_loss"].backward()
            self.clip_gradients(self.savi_optimizer, gradient_clip_val=self.savi_grad_clip, gradient_clip_algorithm="norm")
            self.savi_optimizer.step()

        if self.after_eval:
            savi_image = visualize_savi_decomposition(images[0], outputs["reconstructions"][0], outputs["rgbs"][0], outputs["masks"][0])
            self.logger.log_image("savi_decomposition", savi_image)

        # Detach slots to prevent gradients from flowing back to the SAVi model.
        slots = outputs["slots"].detach()

        # Learn to predict dynamics in slot-space.
        dynamics_optimizer.zero_grad()
        outputs |= self.compute_dynamics_loss(images, slots, actions)
        self.manual_backward(outputs["dynamics_loss"])
        self.clip_gradients(dynamics_optimizer, gradient_clip_val=self.prediction_grad_clip, gradient_clip_algorithm="norm")
        dynamics_optimizer.step()

        # Learn to predict rewards from slot representation.
        reward_optimizer.zero_grad()
        outputs |= self.compute_reward_loss(images, outputs["reconstructions"], slots, rewards)
        self.manual_backward(outputs["reward_loss"])
        self.clip_gradients(reward_optimizer, gradient_clip_val=self.rl_grad_clip, gradient_clip_algorithm="norm")
        reward_optimizer.step()

        # Update the target critic network.
        for critic_param, critic_target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            critic_target_param.data.copy_((1 - self.critic_ema_decay) * critic_param.data + self.critic_ema_decay * critic_target_param.data)

        # Perform latent imagination to train the actor and critic.
        lambda_returns, predicted_values_targ, predicted_values, action_entropies = self.imagine_ahead(slots, actions)

        # Learn the actor.
        actor_optimizer.zero_grad()
        outputs |= self.compute_actor_loss(lambda_returns, action_entropies)
        self.manual_backward(outputs["actor_loss"])
        self.clip_gradients(actor_optimizer, gradient_clip_val=self.rl_grad_clip, gradient_clip_algorithm="norm")
        actor_optimizer.step()

        # Learn the critic.
        critic_optimizer.zero_grad()
        outputs |= self.compute_critic_loss(lambda_returns, predicted_values, predicted_values_targ)
        self.manual_backward(outputs["critic_loss"])
        self.clip_gradients(critic_optimizer, gradient_clip_val=self.rl_grad_clip, gradient_clip_algorithm="norm")
        critic_optimizer.step()

        # Log all losses.
        for key, value in outputs.items():
            if key.endswith("_loss"):
                self.log(f"train/{key}", value.item())
        return outputs

    def compute_dynamics_loss(self, images: torch.Tensor, slots: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
        batch_size, sequence_length, num_slots, slot_dim = slots.shape
        num_context = torch.randint(self.min_num_context, self.max_num_context + 1, (1,)).item()
        context_slots = slots[:, :num_context].detach()
        future_slots = self.dynamics_predictor.predict_slots(self.imagination_horizon, context_slots, actions[:, 1:num_context + self.imagination_horizon].clone().detach())
        predicted_slots = torch.cat([context_slots, future_slots], dim=1)

        predicted_rgbs, predicted_masks = self.savi.decoder(predicted_slots.flatten(end_dim=1))
        predicted_rgbs = predicted_rgbs.reshape(batch_size, num_context + self.imagination_horizon, num_slots, 3, *self.env.image_size)
        predicted_masks = predicted_masks.reshape(batch_size, num_context + self.imagination_horizon, num_slots, 1, *self.env.image_size)
        predicted_images = torch.clamp(torch.sum(predicted_rgbs * predicted_masks, dim=2), 0., 1.)

        slot_loss = F.mse_loss(predicted_slots[:, num_context:], slots[:, num_context:num_context + self.imagination_horizon])
        image_loss = F.mse_loss(predicted_images[:, num_context:], images[:, num_context:num_context + self.imagination_horizon])

        if self.after_eval:
            dynamics_image = visualize_dynamics_prediction(predicted_images[0], predicted_rgbs[0], predicted_masks[0], num_context, images[0, :num_context + self.imagination_horizon])
            self.logger.log_image("dynamics_prediction", dynamics_image)

        return {"slot_loss": slot_loss, "image_loss": image_loss, "dynamics_loss": slot_loss + image_loss, "predicted_images": predicted_images}

    def compute_iris_dynamics_loss(self, images: torch.Tensor, slots: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
        batch_size, sequence_length, num_slots, slot_dim = slots.shape
        future_slots = self.dynamics_predictor.predict_slots(slots, actions[:, 1:].clone().detach())

        dynamics_residual = True
        if dynamics_residual:
            prev_slots = slots[:, :-1]
            bias = prev_slots.reshape(batch_size, -1, slot_dim)[:, 1:]
            future_slots[:, self.savi.num_slots:-1] = future_slots[:, self.savi.num_slots:-1] + bias

        from einops import rearrange
        predictions = future_slots[:, :-1]
        labels = rearrange(slots, 'b t k s -> b (t k) s')[:, 1:]

        full_slot_predictions = torch.cat((slots[:, :1, 0], predictions), dim=1).reshape(batch_size, sequence_length, num_slots, slot_dim)

        # print("future_slots.shape:", future_slots.shape)
        #
        # print("slots.shape:", slots.shape)
        #
        # print("predictions.shape:", predictions.shape)
        # print("labels.shape:", labels.shape)

        #future_slots = future_slots.reshape(batch_size, sequence_length, num_slots, slot_dim)

        #predicted_slots = torch.cat((slots[:, :self.num_context], future_slots[:, self.num_context:]), dim=1)

        predicted_slots = full_slot_predictions

        predicted_rgbs, predicted_masks = self.savi.decoder(predicted_slots.flatten(end_dim=1))
        predicted_rgbs = predicted_rgbs.reshape(batch_size, sequence_length, num_slots, 3, *self.env.image_size)
        predicted_masks = predicted_masks.reshape(batch_size, sequence_length, num_slots, 1, *self.env.image_size)
        predicted_images = torch.clamp(torch.sum(predicted_rgbs * predicted_masks, dim=2), 0., 1.)

        slot_loss = F.mse_loss(predictions[:, self.savi.num_slots:], labels[:, self.savi.num_slots:])
        image_loss = F.mse_loss(predicted_images[:, self.num_context:], images[:, self.num_context:])

        if self.after_eval:
            dynamics_image = visualize_dynamics_prediction(images[0], predicted_images[0], predicted_rgbs[0], predicted_masks[0], self.num_context)
            self.logger.log_image("dynamics_prediction", dynamics_image)

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
                self.logger.log_image("reward_prediction", reward_image)

                # Log visualization of reward predictor attention to inspect reward-predictive elements.
                output_weights = get_attention_weights(self.reward_predictor, slots[:1,])
                predicted_rgbs, predicted_masks = self.savi.decoder(
                    slots[:1].flatten(end_dim=1))
                attention_image = visualize_reward_predictor_attention(images[0], reconstructions[0], rewards[0], predicted_rewards.mean.squeeze(2)[0], output_weights, predicted_rgbs, predicted_masks)
                self.logger.log_image("reward_predictor_attention", attention_image)

        return {"reward_loss": -masked_log_probs.mean()}

    def imagine_ahead(self, slots: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, num_slots, slot_dim = slots.shape

        action_entropies = []
        # randomly sample starting states (and their corresponding actions)
        longer_imagination_context = True
        if longer_imagination_context:
            num_context = torch.randint(self.min_num_context, sequence_length + 1, (1,)).item()
        else:
            num_context = torch.randint(self.min_num_context, self.max_num_context + 1, (1,)).item()
        slot_history = slots[:, :num_context].detach()
        action_history = actions[:, 1:num_context].detach()

        # Actor update
        # Freeze models except action model and imagine next states
        with FreezeParameters([self.reward_predictor, self.critic]):
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

            predicted_rewards = TwoHotEncodingDistribution(self.reward_predictor(slot_history, start=num_context), dims=1).mean.squeeze()
            predicted_values = TwoHotEncodingDistribution(self.critic(slot_history, start=num_context), dims=1).mean.squeeze()

        lambda_returns = self.compute_lambda_returns(predicted_rewards, predicted_values)

        action_entropies = torch.stack(action_entropies, dim=1)

        # Value update
        slot_history = slot_history.detach()
        # Predict imagined values
        predicted_values_targ = TwoHotEncodingDistribution(self.critic_target(slot_history[:, :-1], start=num_context - 1),
                                                   dims=1).mean.squeeze()
        predicted_values = TwoHotEncodingDistribution(self.critic(slot_history[:, :-1], start=num_context - 1), dims=1)

        if self.after_eval:
            with torch.no_grad():
                # Log visualization of a latent imagination sequence.
                predicted_rgbs, predicted_masks = self.savi.decoder(slot_history[0])
                predicted_rgbs = predicted_rgbs.reshape(1, -1, num_slots, 3, *self.env.image_size)
                predicted_masks = predicted_masks.reshape(1, -1, num_slots, 1, *self.env.image_size)
                predicted_images = torch.clamp(torch.sum(predicted_rgbs * predicted_masks, dim=2), 0., 1.)
                imagination_image = visualize_dynamics_prediction(predicted_images[0], predicted_rgbs[0], predicted_masks[0], num_context)
                self.logger.log_image("latent_imagination", imagination_image)

                # Log visualization of actor attention.
                output_weights = get_attention_weights(self.actor, slot_history[:1, :num_context + self.imagination_horizon])
                actor_attention_image = visualize_output_attention(output_weights, predicted_rgbs[0], predicted_masks[0])
                self.logger.log_image("actor_attention", actor_attention_image)
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
        slots = self.savi(observation.unsqueeze(1), self.last_action.unsqueeze(1), prior_slots=last_slots, step_offset=step_offset, reconstruct=False)  # Expand sequence dimension on image.
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


@hydra.main(config_path="../configs", config_name="sold")
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
