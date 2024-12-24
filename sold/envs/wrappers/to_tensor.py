import gymnasium as gym
import torch
from torchvision import transforms
from typing import Tuple


class ToTensor(gym.Wrapper):
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        obs, reward, done, info = self.env.step(action.detach().cpu().numpy())
        return transforms.ToTensor()(obs.copy()), reward, done, info

    def reset(self) -> torch.Tensor:
        return transforms.ToTensor()(self.env.reset().copy())
