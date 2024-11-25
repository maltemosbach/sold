import hydra
from sold.training.utils import seed_everything, instantiate_dataloaders, instantiate_trainer
from lightning import LightningModule
from lightning.pytorch.utilities.types import Optimizer, OptimizerLRScheduler, STEP_OUTPUT
from omegaconf import DictConfig
from sold.models.savi.model import SAVi
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


class SAViTrainer(LightningModule):
    def __init__(self, savi: SAVi, optimizer: Callable[[Iterable], Optimizer],
                 scheduler: Optional[DictConfig] = None) -> None:
        super().__init__()
        self.savi = savi
        self._create_optimizer = optimizer
        self._scheduler_params = scheduler

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self._create_optimizer(self.savi.parameters())
        if self._scheduler_params is not None:
            scheduler = self._scheduler_params.scheduler(optimizer)
            scheduler_dict = {"scheduler": scheduler}
            if self._scheduler_params.get("extras"):
                for key, value in self._scheduler_params.get("extras").items():
                    scheduler_dict[key] = value
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return {"optimizer": optimizer}

    def compute_reconstruction_loss(self, images: torch.Tensor) -> Dict[str, Any]:
        slots, reconstructions, rgbs, masks = self.savi(images)
        loss = F.mse_loss(reconstructions.clamp(0, 1), images.clamp(0, 1))
        return {"reconstruction_loss": loss, "images": images, "reconstructions": reconstructions, "rgbs": rgbs,
                "masks": masks}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        images, actions = batch
        outputs = self.compute_reconstruction_loss(images)
        self.log("train/reconstruction_loss", outputs["reconstruction_loss"], prog_bar=True)
        return outputs | {"loss": outputs["reconstruction_loss"]}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        images, actions = batch
        outputs = self.compute_reconstruction_loss(images)
        self.log("valid/reconstruction_loss", outputs["reconstruction_loss"], prog_bar=True)
        return {"loss": outputs["reconstruction_loss"]}


def load_savi(checkpoint_path: str, finetune: DictConfig):
    savi_trainer = SAViTrainer.load_from_checkpoint(checkpoint_path)
    savi_model = savi_trainer.savi
    return savi_model


@hydra.main(config_path="../configs/", config_name="savi")
def train(cfg: DictConfig):
    if cfg.logger.log_to_wandb:
        import wandb
        wandb.init(project="sold", config=dict(cfg), sync_tensorboard=True)

    seed_everything(cfg.experiment.seed)
    train_dataloader, val_dataloader = instantiate_dataloaders(cfg.dataset)
    savi = hydra.utils.instantiate(cfg.model)
    trainer = instantiate_trainer(cfg)
    trainer.fit(savi, train_dataloader, val_dataloader)

    if cfg.logger.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
