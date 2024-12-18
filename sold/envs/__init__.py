import gym
from typing import Tuple
from envs.wrappers.to_tensor import ToTensor


def missing_dependencies(task: str, *args, **kwargs):
    raise ValueError(f"Missing dependencies to run '{task}' task.")


from envs.from_gym import make_env as make_gym_env
from envs.from_mof import make_env as make_mof_env
from envs.from_dmcontrol import make_env as make_dmcontrol_env



def make_env(suite: str, name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int,
             seed: int = 0) -> gym.Env:
    if suite == 'gym':
        env = make_gym_env(name, image_size, max_episode_steps, action_repeat, seed)
    elif suite == 'mof':
        env = make_mof_env(name, image_size, max_episode_steps, action_repeat, seed)
    elif suite == 'dmcontrol':
        env = make_dmcontrol_env(name, image_size, max_episode_steps, action_repeat, seed)
    else:
        raise ValueError(f"Unsupported environment suite: {suite}")
    env = ToTensor(env)
    return env
