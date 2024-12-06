from collections import defaultdict
import hydra
from omegaconf import DictConfig, OmegaConf
from sold.train_sold import SOLDModule
from sold.utils.training import set_seed
import os
import torch
from torchvision.io import write_video
from torchvision import transforms
from typing import Any, Dict

os.environ["HYDRA_FULL_ERROR"] = "1"


@torch.no_grad()
def play_episode(sold: SOLDModule, mode: str = "eval") -> Dict[str, Any]:
    obs, done, info = sold.env.reset(), False, {}
    episode = defaultdict(list)
    episode["obs"].append(obs)
    episode["high_res"].append(transforms.ToTensor()(sold.env.render(size=(1024, 1024)).copy()))
    while not done:
        last_action = sold.select_action(obs.to(sold.device), is_first=len(episode["obs"]) == 1, mode=mode).cpu()
        obs, reward, done, info = sold.env.step(last_action)
        episode["obs"].append(obs.cpu())
        episode["high_res"].append(transforms.ToTensor()(sold.env.render(size=(1024, 1024)).copy()))
        episode["reward"].append(reward)

    if "success" in info:
        episode["success"] = info["success"]
    return episode


@hydra.main(config_path="./configs", config_name="eval_sold")
def evaluate(cfg: DictConfig):
    set_seed(cfg.seed)
    env = hydra.utils.instantiate(cfg.env)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    sold = SOLDModule.load_from_checkpoint(cfg.checkpoint, env=env)

    for episode_index in range(cfg.eval_episodes):
        episode = play_episode(sold, mode="eval")
        print("episode.keys():", episode.keys())

        torch.stack(episode["obs"])
        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
        write_video(os.path.join(output_dir, f"videos/episode_obs_{episode_index}.mp4"),
                    (torch.stack(episode["obs"]).permute(0, 2, 3, 1) * 255).to(torch.uint8), fps=10)
        write_video(os.path.join(output_dir, f"videos/episode_high_res_{episode_index}.mp4"),
                    (torch.stack(episode["high_res"]).permute(0, 2, 3, 1) * 255).to(torch.uint8), fps=10)

    print("done")
    input()


if __name__ == "__main__":
    evaluate()