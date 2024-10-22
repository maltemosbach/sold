import hydra
from omegaconf import DictConfig
from sold.utils.train import seed_everything, instantiate_dataloaders, instantiate_trainer
from sold.savi.model import SAVi


@hydra.main(config_path="../configs/", config_name="savi")
def train(cfg: DictConfig):
    seed_everything(cfg.experiment.seed)
    train_dataloader, val_dataloader = instantiate_dataloaders(cfg.dataset)
    savi = hydra.utils.instantiate(cfg.model)
    trainer = instantiate_trainer(cfg)
    trainer.fit(savi, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
