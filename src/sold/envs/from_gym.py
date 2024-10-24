import gym
import logging
import numpy as np
from sold.envs.image_env import ImageEnv
from typing import Tuple


class FromGym(ImageEnv):
    def __init__(self, name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int,
                 seed: int = 0, accumulate_reward: bool = False) -> None:
        gym.logger.set_level(logging.ERROR)
        super().__init__(image_size, max_episode_steps, action_repeat, seed, accumulate_reward)
        self._env = gym.make(name)
        self._env.seed(seed)

    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

    def _step(self, action: np.ndarray) -> Tuple[float, bool]:
        state, reward, done, info = self._env.step(action)
        return reward, done

    def _reset(self) -> np.ndarray:
        self._env.reset()
        return self.render()

    def success(self):
        return self._env.success()

    def render(self) -> np.ndarray:
        return self._env.render(mode='rgb_array', size=self.image_size)

    def close(self):
        self._env.close()
