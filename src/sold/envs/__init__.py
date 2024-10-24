from .image_env import ImageEnv
from .from_gym import FromGym
from .from_mof import FromMOF
from typing import Tuple


def load_env(suite: str, name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int,
             seed: int = 0, accumulate_reward: bool = False) -> ImageEnv:
    if suite == 'gym':
        env = FromGym(name, image_size, max_episode_steps, action_repeat, seed, accumulate_reward)
    elif suite == 'mof':
        env = FromMOF(name, image_size, max_episode_steps, action_repeat, seed, accumulate_reward)
    else:
        raise ValueError(f"Unsupported environment suite: {suite}")

    env.suite = suite
    env.name = name
    return env
