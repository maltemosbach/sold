import glob
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
from typing import Any, Dict


class ImageFolderDataset:
    def __init__(self, path: str, data_dir: str, split: str, sequence_length: int = 8, **kwargs) -> None:
        self.path = path
        self.data_dir = data_dir
        self.split = split
        self.sequence_length = sequence_length
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

        image_sequence, actions = self[0]
        self.image_size = list(image_sequence[0].shape[1:])
        self.action_dim = actions[0].shape[0]

    @property
    def dataset_infos(self) -> Dict[str, Any]:
        return {"image_size": self.image_size, "action_dim": self.action_dim}

    def __getitem__(self, index: int):
        episode = self.episodes[index]
        start_index = np.random.randint(0, len(episode) - self.sequence_length) if self.split == "train" else 0

        image_sequence = []
        for image_index in range(start_index, start_index + self.sequence_length):
            image = Image.open(episode[image_index])
            image = transforms.ToTensor()(image)[:3]
            image_sequence.append(image)
        image_sequence = torch.stack(image_sequence, dim=0).float()
        actions = torch.from_numpy(
            np.load("/" + os.path.join(*(episode[0].split("/")[:-1] + ["actions.npy"])))[start_index:start_index + self.sequence_length])

        return image_sequence, actions

    def __len__(self) -> int:
        return len(self.episodes)
