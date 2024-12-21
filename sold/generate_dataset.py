import gym
import hydra
import numpy as np
from omegaconf import DictConfig
import os
import torch
from torchvision.transforms import ToPILImage
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def save_episode(path: str, cfg: DictConfig) -> None:
    os.mkdir(path)
    env = hydra.utils.instantiate(cfg.env)

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

    for split in ["train", "val", "test"]:
        num_episodes = getattr(cfg, f"num_{split}")
        os.mkdir(os.path.join(output_dir, split))

        paths = [os.path.join(output_dir, split, str(episode)) for episode in range(num_episodes)]
        cfgs = [cfg] * len(paths)

        with ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
            futures = [executor.submit(save_episode, path, cfg) for path, cfg in zip(paths, cfgs)]
            for _ in tqdm.tqdm(as_completed(futures), total=len(futures), desc=split.capitalize()):
                pass


if __name__ == "__main__":
    generate_dataset()
