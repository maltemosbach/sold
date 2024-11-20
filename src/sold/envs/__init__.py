import gym
from typing import Tuple
from sold.envs.wrappers import ToTensor


def missing_dependencies(task: str, *args, **kwargs):
    raise ValueError(f"Missing dependencies to run '{task}' task.")


try:
    from sold.envs.from_gym import make_env as make_gym_env
except:
    make_gym_env = missing_dependencies
#try:
from sold.envs.from_mof import make_env as make_mof_env
# except:
#     make_mof_env = missing_dependencies


def make_env(suite: str, name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int,
             seed: int = 0) -> gym.Env:
    if suite == 'gym':
        env = make_gym_env(name, image_size, max_episode_steps, action_repeat, seed)
    elif suite == 'mof':
        env = make_mof_env(name, image_size, max_episode_steps, action_repeat, seed)
    else:
        raise ValueError(f"Unsupported environment suite: {suite}")
    env = ToTensor(env)

    env.suite = suite  # Necessary?
    env.name = name
    return env
