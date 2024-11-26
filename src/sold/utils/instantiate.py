import hydra
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing import List


def instantiate_dataloaders(cfg: DictConfig) -> List[DataLoader]:
    return [DataLoader(hydra.utils.instantiate(cfg, split=split), batch_size=cfg.batch_size, shuffle=(split == "train"),
                       num_workers=cfg.num_workers) for split in ["train", "val"]]


def instantiate_many(cfg: DictConfig) -> List:
    instantiated_objects = []
    for _, conf in cfg.items():
        if isinstance(conf, DictConfig) and "_target_" in conf:
            instantiated_objects.append(hydra.utils.instantiate(conf))
    return instantiated_objects


def instantiate_trainer(cfg: DictConfig) -> Trainer:
    return hydra.utils.instantiate(cfg.trainer, logger=instantiate_many(cfg.logger),
                                   callbacks=instantiate_many(cfg.callbacks))

