import gym


class Pixels(gym.Wrapper):
    def __init__(self, env: gym.Env, image_size: Tuple[int, int]) -> None:
        super().__init__(env)
        self.image_size = image_size

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)
        return self.render(), reward, done, info

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self.render()

    def render(self) -> np.ndarray:
        return self.env.render(mode='rgb_array', size=self.image_size)
