from sold.envs import Env
from sold.envs.from_gym import FromGym
from sold.envs.from_mof import FromMOF
from typing import Tuple


def load_env(suite: str, name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int,
             seed: int = 0, accumulate_reward: bool = False) -> Env:
    if suite == 'gym':
        return FromGym(name, image_size, max_episode_steps, action_repeat, seed, accumulate_reward)
    elif suite == 'mof':
        return FromMOF(name, image_size, max_episode_steps, action_repeat, seed, accumulate_reward)
    else:
        raise ValueError(f"Unsupported environment suite: {suite}")
