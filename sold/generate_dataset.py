import gymnasium as gym
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
    obs, _ = env.reset()
    for k,v in obs.items():
        if "image" in k:
            ToPILImage()(v).save(os.path.join(path, f'{step_count}__{k}.png'))

    done = False
    while not done:
        action = torch.from_numpy(env.action_space.sample().astype(np.float32))
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated is True or truncated is True
        # obs = obs_raw["left_image"]
        step_count += 1
        for k,v in obs.items():
            if "image" in k:
                ToPILImage()(v).save(os.path.join(path, f'{step_count}__{k}.png'))
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

        if cfg.num_workers > 1:
            with ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
                futures = [executor.submit(save_episode, path, cfg) for path, cfg in zip(paths, cfgs)]
                for _ in tqdm.tqdm(as_completed(futures), total=len(futures), desc=split.capitalize()):
                    pass
        else:
            progress = tqdm.tqdm(total=len(paths))
            for path_iter, cfg_iter in zip(paths, cfgs):
                save_episode(path_iter, cfg_iter)
                progress.update(1)

                    

if __name__ == "__main__":
    generate_dataset()
