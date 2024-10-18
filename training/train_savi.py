import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
#from sold.datasets.image_folder import ImageFolderDataset
from sold.savi.savi import SAVi
from torch.utils.data import DataLoader


@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
    val_dataset = hydra.utils.instantiate(cfg.dataset, split="val")
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.dataset.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.dataset.num_workers)

    #savi = hydra.utils.instantiate(cfg.savi)


    trainer = L.Trainer(max_epochs=100)

    print("train_dataset.split:", train_dataset.split)
    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    print("train_dataloader:", train_dataloader)

    #trainer.fit(savi, train_dataloader, val_dataloader)


    print("dataset:", train_dataset)

    #print("savi:", savi)


if __name__ == "__main__":
    train()
