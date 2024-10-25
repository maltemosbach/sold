import hydra
from omegaconf import DictConfig
from sold.utils.train import seed_everything, instantiate_trainer


from functools import partial
from typing import Any
from lightning import LightningModule
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT, TRAIN_DATALOADERS

import sold
from sold.models.savi.model import SAVi
from sold.utils.replay_buffer import ExperienceReplay
from sold.models.sold.prediction import GaussianPredictor, TwoHotPredictor
from sold.datasets.experience_source import ExperienceSourceDataset
import numpy as np
from torchvision import transforms
from sold.models.sold.dynamics import DynamicsModel, PredictorWrapper
from sold.envs import load_env
from sold.envs.image_env import ImageEnv


class SOLDTrainer(LightningModule):
    def __init__(self, env: ImageEnv, savi: SAVi, actor: partial[GaussianPredictor], critic: partial[TwoHotPredictor],
                 reward_predictor: partial[TwoHotPredictor], learning_rate: float, collect_interval: int) -> None:
        super().__init__()
        self.env = env
        self.savi = savi
        regression_infos = {"max_episode_steps": env.max_episode_steps,  "num_slots": savi.corrector.num_slots,
                            "slot_dim": savi.corrector.slot_dim}
        self.actor = actor(**regression_infos, output_dim=env.action_space.shape[0])
        self.critic = critic(**regression_infos)
        self.reward_predictor = reward_predictor(**regression_infos)

        self.dynamics_predictor = PredictorWrapper(DynamicsModel(self.savi.num_slots, self.savi.slot_dim, sequence_length=15,
                                                   action_dim=env.action_space.shape[0]))

        self.collect_interval = collect_interval
        self.learning_rate = learning_rate

        self.savi_optimizer = torch.optim.Adam(self.savi.parameters(), lr=0.0001)

        self.batch_size = 2
        self.sequence_length = 16
        self.num_seed_episodes = 10

    @property
    def current_episode(self) -> int:
        return self.current_epoch + self.num_seed_episodes

    def on_fit_start(self) -> None:
        """Populate the replay buffer with random seed episodes."""
        self.replay_buffer = ExperienceReplay(self.logger.log_dir + "/replay_buffer", int(1e6),
                                              self.env.max_episode_steps // self.env.action_repeat, self.env.image_size,
                                              self.env.action_space.shape[0])

        for episode_idx in range(self.num_seed_episodes):
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

        self.savi = self.savi.to(self.device)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = ExperienceSourceDataset(partial(self.replay_buffer.sample, n=self.batch_size, length=self.sequence_length), collect_interval=self.collect_interval)
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
        dynamics_predictor_weights = list(self.dynamics_predictor.parameters())
        return torch.optim.Adam([
            {'params': actor_weights, 'lr': self.learning_rate},
            {'params': critic_weights, 'lr': self.learning_rate},
            {'params': reward_predictor_weights, 'lr': self.learning_rate},
            {'params': dynamics_predictor_weights, 'lr': self.learning_rate}
        ])

    def training_step(self, batch, batch_index: int) -> STEP_OUTPUT:
        images, actions, rewards, is_firsts = batch

        # import matplotlib.pyplot as plt
        # for t in range(images.shape[1]):
        #     plt.imshow(images[0, t].cpu().permute(1, 2, 0))
        #     plt.show()


        # print("batch_idx:", batch_idx)
        # print("images.shape:", images.shape)
        # print("actions.shape:", actions.shape)
        # print("rewards.shape:", rewards.shape)
        # print("is_firsts.shape:", is_firsts.shape)
        # input()

        self.finetune_savi(batch, batch_index)

        loss = self.compute_sold_loss(batch)
        return loss

    def finetune_savi(self, batch, batch_index: int) -> None:
        images, actions, rewards, is_firsts = batch

        self.savi_optimizer.zero_grad()
        reconstruction_loss = self.savi.compute_reconstruction_loss(images, log_visualizations=batch_index == 0,
                                                                    logger=self.logger)
        self.log("reconstruction_loss", reconstruction_loss.item())
        reconstruction_loss.backward()
        # Apply gradient clipping as before.
        # print("self.savi.trainer.gradient_clip_val:", self.savi.trainer.gradient_clip_val)
        # input()

        torch.nn.utils.clip_grad_norm_(self.savi.parameters(), 0.05)
        self.savi_optimizer.step()

    def compute_sold_loss(self, batch) -> torch.Tensor:
        images, actions, rewards, is_firsts = batch
        dynamics_loss = self.compute_dynamics_loss(images, actions)

        loss = dynamics_loss
        return loss

    def compute_dynamics_loss(self, images: torch.Tensor, actions: torch.Tensor, log_visualizations: bool = False
                              ) -> torch.Tensor:
        with torch.no_grad():
            slots = self.savi(images, reconstruct=False)

        num_context = 1
        predictor_input = slots[:, :num_context].detach()

        num_preds = 15


        pred_slots = self.dynamics_predictor.predict_slots(num_preds, predictor_input, actions[:, 1:].clone().detach())

        batch_size, sequence_length, num_slots, slot_dim = slots.size()


        predicted_images = self.savi.decode(pred_slots.reshape(-1, num_slots, slot_dim))[0].reshape(-1, num_preds, 3, *self.env.image_size)


        slot_loss = F.mse_loss(pred_slots, slots[:, num_context:])
        reconstruction_loss = F.mse_loss(predicted_images, images[:, num_context:])

        dynamics_loss = slot_loss + reconstruction_loss
        self.log("dynamics_loss", dynamics_loss.item(), prog_bar=True)
        return dynamics_loss

    def on_train_epoch_end(self) -> None:
        """Collect data samples after every epoch."""
        images = []
        actions = []
        rewards = []
        is_first = []

        image = self.env.reset()
        image = transforms.ToTensor()(image.copy()).unsqueeze(dim=0).to(self.device)  # Expand batch dimension.
        images.append(image.cpu())
        actions.append(torch.zeros(self.env.action_space.shape[0]))
        rewards.append(0.0)
        is_first.append(True)

        slot_history = None
        for time_step in range(self.env.max_episode_steps // self.env.action_repeat):
            # Encode image into slots and append to the slot history.
            last_slots = slot_history[:, -1] if slot_history is not None else None
            step_offset = 1 if slot_history is not None else 0

            # print("image.shape:", image.shape)
            #print("last_slots.shape:", last_slots.shape)
            #input()

            # print("self.savi.device:", self.savi.device)
            # print("image.device:", image.device)
            slots = self.savi(image.unsqueeze(1), prior_slots=last_slots, step_offset=step_offset, reconstruct=False)  # Expand sequence dimension on image.
            slot_history = torch.cat([slot_history, slots], dim=1) if slot_history is not None else slots

            action = self.select_action(slot_history, sample=True)
            image, reward, done = self.env.step(action)
            image = transforms.ToTensor()(image.copy()).unsqueeze(dim=0).to(self.device)  # Expand batch dimension.

            # print("slots.shape:", slots.shape)
            #
            # print("action:", action)
            # print("action.shape:", action.shape)

            images.append(image.cpu())
            actions.append(torch.from_numpy(action).float())
            rewards.append(reward)
            is_first.append(False)

        # import matplotlib.pyplot as plt
        # for image in images:
        #     plt.imshow(image[0].cpu().permute(1, 2, 0))
        #     plt.show()

        self.replay_buffer.append(images, actions, rewards, is_first)
        self.env.close()
        self.log("train_return", np.sum(rewards))
        self.log("current_episode", self.current_episode)

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return None

    def on_validation_epoch_end(self) -> None:
        # Collect validation episodes (Episodes that are not added to the replay buffer or used for training)

        print("on_validation_epoch_end")

        num_validation_episodes = 10
        # save observations and actions for later
        observations = [[] for _ in range(num_validation_episodes)]
        actions = [[] for _ in range(num_validation_episodes)]
        rewards = [[] for _ in range(num_validation_episodes)]
        total_rewards = np.zeros((num_validation_episodes,))
        success = 0

        for episode_index in range(num_validation_episodes):
            validation_env = load_env(self.env.suite, self.env.name, self.env.image_size, self.env.max_episode_steps, self.env.action_repeat)
            print("validation_env:", validation_env)
            input()

        #     observation = test_env.reset()
        #
        #     slot_history = None
        #     observations[i].append(observation)
        #     progress_bar = tqdm(range(math.ceil(
        #         self.exp_params["environment"]["max_episode_length"] / self.exp_params["environment"]["action_repeat"] /
        #         self.exp_params["planning"]["num_action_select"])))
        #     for t in progress_bar:
        #         action, observation, reward, done, slot_history = self.planner.select_action(test_env,
        #                                                                                      observation.unsqueeze(
        #                                                                                          0).to(self.device),
        #                                                                                      slot_history)
        #
        #         for j in range(len(reward)):
        #             total_rewards[i] += reward[j]
        #             rewards[i].append(reward[j])
        #             observations[i].append(observation[j].unsqueeze(0))
        #             actions[i].append(action[:, j])
        #
        #         progress_bar.set_description(
        #             f"Test episode {i + 1} iter {t}: current reward {reward[-1]:.5f}     total reward {total_rewards[i]:.5f}. ")
        #         if done[-1]:
        #             progress_bar.close()
        #             break
        #
        #     success += test_env.success()
        #
        #     # Close test environment
        #     test_env.close()
        #
        # self.writer.log_full_dictionary(
        #     dict={'test_rewards/mean': np.mean(total_rewards.tolist()),
        #           'test_rewards/max': np.max(total_rewards.tolist()),
        #           'test_rewards/min': np.min(total_rewards.tolist())},
        #     step=episode + 1,
        #     plot_name="test_rewards",
        #     dir="Rewards",
        # )
        # self.writer.add_scalar('Rewards/test_success_rate', success, episode + 1)
        #
        # self.visualizations(observations[:10], actions[:10], rewards[:10], episode)

    def select_action(self, slot_history: torch.Tensor, sample: bool = False) -> np.ndarray:
        action_dist = self.actor(slot_history, start=slot_history.shape[1] - 1)
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
