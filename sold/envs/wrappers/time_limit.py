import gym
import numpy as np
from typing import Tuple, Optional


class TimeLimit(gym.Wrapper):
    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None) -> None:
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self.max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        self._elapsed_steps = 0
        return self.env.reset()
