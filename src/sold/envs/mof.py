import gym
import random
from sold.envs.gym import make_env as make_gym_env
from typing import Tuple


class VariableDistractorsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, env_names: List[str]) -> None:
        super().__init__(env)
        self.image_size = image_size

    def reset(self) -> np.ndarray:
        name = random.choice(self.env_names)
        self.env = gym.make(name)
        return self.env.reset()


def make_env(name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0):
    # Register MOF environments and parse the number of distractors.
    import multi_object_fetch
    task, distractors, reward = name.split('_')
    min_distractors, max_distractors = map(int, distractors[:-len('Distractors')].split('to'))
    env_names = []
    for num_distractors in range(min_distractors, max_distractors + 1):
        env_names.append(f'{task}_{num_distractors}Distractors_{reward}')
    env = make_gym_env(env_names[0], image_size, max_episode_steps, action_repeat, seed)
    env = VariableDistractorsWrapper(env, env_names)
    return env
