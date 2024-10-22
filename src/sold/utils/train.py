import hydra
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import random
import torch
from torch.utils.data import DataLoader
from typing import List


def seed_everything(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def instantiate_dataloaders(cfg: DictConfig) -> List[DataLoader]:
    return [DataLoader(hydra.utils.instantiate(cfg, split=split), batch_size=cfg.batch_size, shuffle=(split == "train"),
                       num_workers=cfg.num_workers) for split in ["train", "val"]]


def instantiate_trainer(cfg: DictConfig) -> Trainer:
    return hydra.utils.instantiate(
        cfg.trainer, logger=TensorBoardLogger(save_dir="logs"),
        callbacks=[hydra.utils.instantiate(callback_cfg) for _, callback_cfg in cfg.callbacks.items()])
