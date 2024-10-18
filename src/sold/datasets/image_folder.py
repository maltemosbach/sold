import glob
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple


class ImageFolderDataset:
    def __init__(self, path: str, data_dir: str, split: str, sequence_length: int = 8,
                 image_size: Tuple[int, int] = (64, 64), **kwargs) -> None:
        self.path = path
        self.data_dir = data_dir
        self.split = split
        self.sequence_length = sequence_length
        self.image_size = image_size

        self.root = os.path.join(self.path, self.data_dir, split)

        self.dirs = []
        for file in os.listdir(self.root):
            try:
                self.dirs.append(int(file))
            except ValueError:
                continue
        self.dirs.sort()

        self.episodes = []
        for d in self.dirs:
            dir_name = os.path.join(self.root, str(d))
            paths = list(glob.glob(os.path.join(dir_name, '*.png')))
            paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.episodes.append(paths)

    def __getitem__(self, index: int):
        episode = self.episodes[index]
        start_index = np.random.randint(0, len(episode) - self.sequence_length) if self.split == "train" else 0

        image_sequence = []
        for image_index in range(start_index, start_index + self.sequence_length):
            image = Image.open(episode[image_index])
            image = image.resize(self.image_size)
            image = transforms.ToTensor()(image)[:3]
            image_sequence.append(image)
        image_sequence = torch.stack(image_sequence, dim=0).float()
        actions = torch.from_numpy(
            np.load("/" + os.path.join(*(episode[0].split("/")[:-1] + ["actions.npy"])))[start_index:start_index + self.sequence_length])

        return image_sequence, actions

    def __len__(self) -> int:
        return len(self.episodes)
