import gym
import hydra
import numpy as np
from omegaconf import DictConfig
import os
import torch
from torchvision.transforms import ToPILImage
import tqdm


def save_episode(path: str, env: gym.Env) -> None:
    os.mkdir(path)

    step_count, actions, rewards = 0, [], []
    obs, done = env.reset(), False
    ToPILImage()(obs).save(os.path.join(path, f'{step_count}.png'))

    while not done:
        action = torch.from_numpy(env.action_space.sample().astype(np.float32))
        obs, reward, done, _ = env.step(action)
        step_count += 1
        ToPILImage()(obs).save(os.path.join(path, f'{step_count}.png'))
        actions.append(action.cpu().numpy())
        rewards.append(reward)

    np.save(os.path.join(path, 'actions.npy'), actions)
    np.save(os.path.join(path, 'rewards.npy'), rewards)


@hydra.main(config_path="../configs", config_name="generate_dataset", version_base=None)
def generate_dataset(cfg: DictConfig) -> None:
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    env = hydra.utils.instantiate(cfg.env)

    for split in ["train", "val", "test"]:
        num_episodes = getattr(cfg, f"num_{split}")
        os.mkdir(os.path.join(output_dir, split))
        for episode in tqdm.tqdm(range(num_episodes), desc=split.capitalize()):
            save_episode(os.path.join(output_dir, split, str(episode)), env,)


if __name__ == "__main__":
    generate_dataset()
