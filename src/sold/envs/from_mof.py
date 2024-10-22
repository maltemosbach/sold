import gym
import random
from sold.envs.from_gym import FromGym
from typing import Tuple


class FromMOF(FromGym):
    def __init__(self, name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int,
                 seed: int = 0, accumulate_reward: bool = False) -> None:
        """Environment wrapper for multi-object fetch environments."""
        import multi_object_fetch  # Register environments.
        task, distractors, reward = name.split('_')
        min_distractors, max_distractors = map(int, distractors[:-len('Distractors')].split('to'))
        self.env_names = []
        for num_distractors in range(min_distractors, max_distractors + 1):
            self.env_names.append(f'{task}_{num_distractors}Distractors_{reward}')
        super().__init__(self.env_names[0], image_size, max_episode_steps, action_repeat, seed, accumulate_reward)

    def reset(self, env_num=None, seed=None):
        seed = self.seed if seed is None else seed
        self.seed += 1
        name = random.choice(self.env_names) if env_num is None else self.env_names[env_num % len(self.env_names)]
        self._env = gym.make(name)
        return super().reset()
