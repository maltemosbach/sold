import hydra
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from omegaconf import DictConfig, OmegaConf
#from sold.datasets.image_folder import ImageFolderDataset
from torch.utils.data import DataLoader

from sold.savi.model import SAVi
from sold.savi.trainer import SAViTrainer


@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_dataset = hydra.utils.instantiate(cfg.savi.dataset, split="train")
    val_dataset = hydra.utils.instantiate(cfg.savi.dataset, split="val")
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.savi.dataset.batch_size, shuffle=True,
                                  num_workers=cfg.savi.dataset.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.savi.dataset.batch_size, shuffle=False,
                                num_workers=cfg.savi.dataset.num_workers)

    savi = hydra.utils.instantiate(cfg.savi.model)
    trainer = hydra.utils.instantiate(cfg.savi.trainer, logger=TensorBoardLogger(save_dir="logs/"),
                                      callbacks=[LearningRateMonitor(logging_interval='step')])

    print("train_dataset.split:", train_dataset.split)
    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    print("train_dataloader:", train_dataloader)

    trainer.fit(savi, train_dataloader, val_dataloader)

    print("dataset:", train_dataset)

    #print("savi:", savi)


if __name__ == "__main__":
    train()
