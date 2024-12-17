import gym
import numpy as np
import random
from envs.from_gym import make_env as make_gym_env
from typing import Callable, Tuple


class VariableDistractorsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, create_random_env: Callable) -> None:
        super().__init__(env)
        self.create_random_env = create_random_env

    def reset(self) -> np.ndarray:
        self.env = self.create_random_env()
        return self.env.reset()


def make_env(name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0):
    # Register MOF environments and parse the number of distractors.
    import multi_object_fetch
    task, distractors, reward = name.split('_')
    min_distractors, max_distractors = map(int, distractors[:-len('Distractors')].split('to'))
    env_names = []
    for num_distractors in range(min_distractors, max_distractors + 1):
        env_names.append(f'{task}_{num_distractors}Distractors_{reward}')

    def create_random_env():
        name = random.choice(env_names)
        return make_gym_env(name, image_size, max_episode_steps, action_repeat, seed)
    env = VariableDistractorsWrapper(create_random_env(), create_random_env)
    return env
