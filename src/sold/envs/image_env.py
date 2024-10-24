from abc import ABC
import gym
import numpy as np
from typing import Tuple


class ImageEnv(ABC):
    """Base class for pixel-based environments."""
    def __init__(self, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int,
                 seed: int = 0, accumulate_reward: bool = False) -> None:
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.action_repeat = action_repeat
        self.seed = seed
        self.accumulate_reward = accumulate_reward
        self._time_step = 0

    @property
    def time_step(self) -> int:
        return self._time_step

    @property
    def action_space(self) -> gym.Space:
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        self._time_step = 0
        return self._reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        accumulated_reward = 0.
        for k in range(self.action_repeat):
            reward, done = self._step(action)
            if self.accumulate_reward:
                accumulated_reward += reward
            else:
                accumulated_reward = reward  # Only keep the last reward if not accumulating
            self._time_step += 1
            done = done or self._time_step == self.max_episode_steps
            if done:
                break
        obs = self.render()
        return obs, accumulated_reward, done

    def _reset(self) -> np.ndarray:
        raise NotImplementedError

    def _step(self, action: np.ndarray) -> Tuple[float, bool]:
        raise NotImplementedError

    def success(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
