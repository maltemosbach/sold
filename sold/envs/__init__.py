import os
os.environ['MUJOCO_GL'] = 'egl'  # Set MuJoCo rendering backend.
from envs.from_dmcontrol import make_env as make_dmcontrol_env
from envs.from_gym import make_env as make_gym_env
from envs.from_survival import make_env as make_survival_env
from envs.from_mof import make_env as make_mof_env
from envs.wrappers.to_tensor import ToTensor
from envs.wrappers.to_tensordict import ToTensorDict
import gymnasium as gym
from typing import Tuple


def make_env(suite: str, name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int,
             seed: int = 0) -> gym.Env:
    if suite == 'gym':
        env = make_gym_env(name, image_size, max_episode_steps, action_repeat, seed)
        env = ToTensor(env)
    elif suite == 'survival':
        env = make_survival_env(name, image_size, max_episode_steps, action_repeat, seed)
        env = ToTensorDict(env)
    elif suite == 'mof':
        env = make_mof_env(name, image_size, max_episode_steps, action_repeat, seed)
        env = ToTensor(env)
    elif suite == 'dmcontrol':
        env = make_dmcontrol_env(name, image_size, max_episode_steps, action_repeat, seed)
        env = ToTensor(env)
    else:
        raise ValueError(f"Unsupported environment suite: {suite}")

    return env
