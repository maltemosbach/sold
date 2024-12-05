import hydra
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Tuple


def instantiate_dataloaders(cfg: DictConfig) -> Tuple:
    datasets = [hydra.utils.instantiate(cfg, split=split) for split in ["train", "val"]]
    dataloaders = [DataLoader(dataset, batch_size=cfg.batch_size, shuffle=(dataset.split == "train"),
                              num_workers=cfg.num_workers) for dataset in datasets]
    return *dataloaders, datasets[0].dataset_infos


def instantiate_many(cfg: DictConfig) -> List:
    instantiated_objects = []
    for _, conf in cfg.items():
        if isinstance(conf, DictConfig) and "_target_" in conf:
            instantiated_objects.append(hydra.utils.instantiate(conf))
    return instantiated_objects


def instantiate_trainer(cfg: DictConfig) -> Trainer:
    return hydra.utils.instantiate(cfg.trainer, logger=instantiate_many(cfg.logger),
                                   callbacks=instantiate_many(cfg.callbacks))


def fill_in_missing(cfg: DictConfig, infos: Dict[str, Any]):
    missing_keys = OmegaConf.missing_keys(cfg)
    for key, value in infos.items():
        for missing_key in missing_keys:
            if missing_key.endswith(key):
                OmegaConf.update(cfg, missing_key, value)
