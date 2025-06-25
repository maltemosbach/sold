import hydra
import numpy as np
from omegaconf import DictConfig
import os
import torch
from torchvision.transforms import ToPILImage
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def save_episode(path: str, cfg: DictConfig) -> None:
    def save_image(image: torch.Tensor) -> None:
        if cfg.save_format == "png":
            ToPILImage()(image).save(os.path.join(path, f'{step_count}.png'))
        elif cfg.save_format == "npz":
            images.append(np.array(ToPILImage()(image)))
        else:
            raise ValueError(f"Unsupported save format: {cfg.save_format}")

    os.mkdir(path)
    env = hydra.utils.instantiate(cfg.env)
    step_count, images, actions, rewards = 0, [], [], []
    obs, done = env.reset(), False
    save_image(obs)

    while not done:
        action = torch.from_numpy(env.action_space.sample().astype(np.float32))
        obs, reward, done, _ = env.step(action)
        step_count += 1
        save_image(obs)
        actions.append(action.cpu().numpy())
        rewards.append(reward)

    episode_dict = {"actions": np.stack(actions), "rewards": np.array(rewards)}
    if cfg.save_format == "npz":
        episode_dict["images"] = np.stack(images)
    np.savez_compressed(os.path.join(path, 'episode.npz'), **episode_dict)


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
            for path, cfg in zip(paths, cfgs):
                save_episode(path, cfg)
                progress.update(1)


if __name__ == "__main__":
    generate_dataset()
