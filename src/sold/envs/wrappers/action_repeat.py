import gym
import numpy as np
from typing import Tuple, Optional


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, action_repeat: int = 1) -> None:
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        accumulated_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, done, info = self.env.step(action)
            accumulated_reward += reward
            if done:
                break
        return obs, accumulated_reward, done, info
