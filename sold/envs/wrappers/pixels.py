import gymnasium as gym
from multi_object_fetch.env import MultiObjectFetchEnv
from PIL import Image
import numpy as np
from typing import Tuple


class Pixels(gym.Wrapper):
    def __init__(self, env: gym.Env, image_size: Tuple[int, int]) -> None:
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,) + image_size, dtype=float)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)
        return self._get_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        if isinstance(self.env.unwrapped, MultiObjectFetchEnv):
            image = self.env.render(mode='rgb_array', size=self.image_size)
        else:
            image = Image.fromarray(self.env.render(mode='rgb_array'))
            image = np.array(image.resize(self.image_size))
        return np.moveaxis(image, -1, 0) / 255.0
