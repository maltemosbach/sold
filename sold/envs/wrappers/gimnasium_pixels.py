import gymnasium as gym
from multi_object_fetch.env import MultiObjectFetchEnv
from PIL import Image
import numpy as np
from typing import Tuple
import torchvision

class GymnasiumPixels(gym.Wrapper):
    def __init__(self, env: gym.Env, image_size: Tuple[int, int]) -> None:
        super().__init__(env)
        self.image_size = image_size
        # self.resize = torchvision.transforms.Resize(self.image_size)

    def _process_images(self, obs):
        for k, v in obs.items():
            if "image" in k:
                image = Image.fromarray(v.astype("uint8"), "RGB")
                image = image.resize(self.image_size)
                obs[k] = np.array(image)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_images(obs)
        return obs, reward, terminated, truncated, info

    def reset(self) -> np.ndarray:
        obs, info = self.env.reset()
        obs = self._process_images(obs)
        return obs, info

