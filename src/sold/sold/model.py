from functools import partial
from lightning import LightningModule
import torch
import sold
from sold.savi.model import SAVi
from sold.sold import Actor, Critic, RewardPredictor


class SOLD(LightningModule):
    def __init__(self, env: sold.Env, savi: SAVi, actor: partial[Actor], critic: Critic,
                 reward_predictor: RewardPredictor) -> None:
        super().__init__()
        self.env = env
        self.savi = savi
        self.actor = actor(action_dim=env.action_space.shape[0])
        self.critic = critic
        self.reward_predictor = reward_predictor

        print("self.savi:", self.savi)

        print("self.env", self.env)

        print("self.actor", self.actor)

        input()
