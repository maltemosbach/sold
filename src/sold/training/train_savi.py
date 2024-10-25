import hydra
from sold.utils.train import seed_everything, instantiate_dataloaders, instantiate_trainer
from functools import partial
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from omegaconf import DictConfig
from sold.models.savi.model import SAVi
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.optim import Optimizer, lr_scheduler


class SAViTrainer(LightningModule):
    def __init__(self, savi: SAVi, optimizer: partial[Optimizer], scheduler: partial[lr_scheduler]) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.savi = savi
        self.create_optimizer = optimizer
        self.create_scheduler = scheduler
        self.training_step_outputs = None

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.create_optimizer(params=self.parameters())
        scheduler = {
            'scheduler': self.create_scheduler(optimizer=optimizer),
            'interval': 'step',
            'name': 'Learning Rate',
        }
        return [optimizer, ], [scheduler, ]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        images, actions = batch
        slots, reconstructions, rgbs, masks = self.savi(images)
        loss = F.mse_loss(reconstructions.clamp(0, 1), images.clamp(0, 1))
        self.log("train_loss", loss, prog_bar=True)
        self.training_step_outputs = (images, reconstructions, rgbs, masks)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        images, actions = batch
        slots, reconstructions, rgbs, masks = self.savi(images)
        loss = F.mse_loss(reconstructions.clamp(0, 1), images.clamp(0, 1))
        self.log("val_loss", loss)
        return None


def load_savi(checkpoint_path: str, finetune: DictConfig):
    # TODO: redo
    savi_model = SAVi.load_from_checkpoint(checkpoint_path)
    savi_model.finetune = finetune
    savi_model.automatic_optimization = False
    return savi_model


@hydra.main(config_path="../configs/", config_name="savi")
def train(cfg: DictConfig):
    seed_everything(cfg.experiment.seed)
    train_dataloader, val_dataloader = instantiate_dataloaders(cfg.dataset)
    savi = hydra.utils.instantiate(cfg.model)
    trainer = instantiate_trainer(cfg)
    trainer.fit(savi, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
