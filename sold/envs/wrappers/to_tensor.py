import gym
import torch
from typing import Tuple


class ToTensor(gym.Wrapper):
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        obs, reward, done, info = self.env.step(action.detach().cpu().numpy())
        return torch.from_numpy(obs).float(), reward, done, info

    def reset(self) -> torch.Tensor:
        return torch.from_numpy(self.env.reset()).float()
