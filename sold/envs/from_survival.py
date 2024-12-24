import gymnasium as gym
from envs.wrappers.action_repeat import GymnasiumActionRepeat
from envs.wrappers.gimnasium_pixels import GymnasiumPixels
from envs.wrappers.time_limit import GymnasiumTimeLimit
from typing import Tuple

import survivalenv

def make_env(name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0):
    # print("<<<<<<<<<<<<<<<")
    env = gym.make(name, render_cameras=True)
    # print(">>>>>>>>>>>>>>>")
    env = GymnasiumActionRepeat(env, action_repeat)
    env = GymnasiumTimeLimit(env, max_episode_steps)
    env = GymnasiumPixels(env, image_size)
    # print("************************************************************")
    return env
