import hydra
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from typing import List
from lightning.pytorch.callbacks import ModelCheckpoint


def instantiate_dataloaders(cfg: DictConfig) -> List[DataLoader]:
    return [DataLoader(hydra.utils.instantiate(cfg, split=split), batch_size=cfg.batch_size, shuffle=(split == "train"),
                       num_workers=cfg.num_workers) for split in ["train", "val"]]


def instantiate_trainer(cfg: DictConfig) -> Trainer:
    return hydra.utils.instantiate(
        cfg.trainer, logger=TensorBoardLogger(save_dir="logs/"),
        callbacks=[hydra.utils.instantiate(callback_cfg) for _, callback_cfg in cfg.callbacks.items()])


@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    train_dataloader, val_dataloader = instantiate_dataloaders(cfg.savi.dataset)
    savi = hydra.utils.instantiate(cfg.savi.model)
    trainer = instantiate_trainer(cfg.savi)
    trainer.fit(savi, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
