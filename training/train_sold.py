import hydra
from omegaconf import DictConfig
from sold.utils.train import seed_everything, instantiate_trainer
from sold.sold.model import SOLD


@hydra.main(config_path="../configs", config_name="sold")
def train(cfg: DictConfig):
    seed_everything(cfg.experiment.seed)
    sold = hydra.utils.instantiate(cfg.model)
    trainer = instantiate_trainer(cfg)

    print("sold: ", sold)

    input()


if __name__ == "__main__":
    train()
