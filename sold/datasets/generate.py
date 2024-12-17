from argparse import ArgumentParser
from envs import make_env
import gym
import numpy as np
import os
import shutil
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--suite", type=str, default="dmcontrol", help="The suite of environments to use.")
    parser.add_argument("--name", type=str, default="cheetah-run", help="The name of the environment to use.")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Directory in which to save the dataset.")
    parser.add_argument("--num_train", type=int, default=20000, help="The number of training videos to generate.")
    parser.add_argument("--num_val", type=int, default=2000, help="The number of validation videos to generate.")
    parser.add_argument("--num_test", type=int, default=2000, help="The number of test videos to generate.")
    parser.add_argument("--width", type=int, default=64, help="The width of the images.")
    parser.add_argument("--height", type=int, default=64, help="The height of the images.")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Maximum number of steps after which to truncate an episode.")
    parser.add_argument("--action_repeat", type=int, default=2, help="The number of times to repeat each action.")

    args = parser.parse_args()

    env = make_env(args.suite, args.name, (args.height, args.width), args.max_episode_steps, args.action_repeat, seed=0)

    if os.path.exists(args.dataset_dir):
        shutil.rmtree(args.dataset_dir)
    os.mkdir(args.dataset_dir)

    for split in ["train", "val", "test"]:
        num_episodes = getattr(args, f"num_{split}")
        os.mkdir(os.path.join(args.dataset_dir, split))
        for episode in tqdm.tqdm(range(num_episodes), desc=split.capitalize()):
            save_episode(os.path.join(args.dataset_dir, split, str(episode)), env,)
