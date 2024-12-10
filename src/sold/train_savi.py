import hydra
from lightning.pytorch.utilities.types import Optimizer, OptimizerLRScheduler, STEP_OUTPUT
from omegaconf import DictConfig
import os
from sold.modeling.savi.model import SAVi
from sold.utils.instantiate import instantiate_trainer, instantiate_dataloaders, fill_in_missing
from sold.utils.training import set_seed
from lightning import LightningModule
from sold.utils.logging import LoggingStepMixin
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

os.environ["HYDRA_FULL_ERROR"] = "1"


class SAViModule(LoggingStepMixin, LightningModule):
    def __init__(self, savi: SAVi, optimizer: Callable[[Iterable], Optimizer],
                 scheduler: Optional[DictConfig] = None) -> None:
        super().__init__()
        self.savi = savi
        self._create_optimizer = optimizer
        self._scheduler_params = scheduler
        self.save_hyperparameters(logger=False)

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

    def compute_reconstruction_loss(self, images: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
        slots, reconstructions, rgbs, masks = self.savi(images, actions[:, 1:])
        loss = F.mse_loss(reconstructions.clamp(0, 1), images.clamp(0, 1))
        return {"reconstruction_loss": loss, "images": images, "reconstructions": reconstructions.clamp(0, 1), "rgbs": rgbs,
                "masks": masks, "slots": slots}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        images, actions = batch
        outputs = self.compute_reconstruction_loss(images, actions)
        self.log("train/reconstruction_loss", outputs["reconstruction_loss"], prog_bar=True)
        return outputs | {"loss": outputs["reconstruction_loss"]}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        images, actions = batch
        outputs = self.compute_reconstruction_loss(images, actions)
        self.log("valid/reconstruction_loss", outputs["reconstruction_loss"], prog_bar=True)
        self.log("valid_loss", outputs["reconstruction_loss"], logger=False)  # Used in checkpoint names.
        return None


def load_savi(checkpoint_path: str):
    savi_module = SAViModule.load_from_checkpoint(checkpoint_path)
    return savi_module.savi


@hydra.main(config_path="./configs", config_name="train_savi")
def train(cfg: DictConfig):
    set_seed(cfg.seed)
    trainer = instantiate_trainer(cfg)
    cfg.dataset.batch_size = cfg.dataset.batch_size // trainer.world_size  # Adjust batch size for distributed training
    train_dataloader, val_dataloader, dataset_infos = instantiate_dataloaders(cfg.dataset)
    fill_in_missing(cfg, dataset_infos)
    savi = hydra.utils.instantiate(cfg.model)

    if cfg.logger.log_to_wandb:
        import wandb
        wandb.init(project="sold", config=dict(cfg), sync_tensorboard=True)
    trainer.fit(savi, train_dataloader, val_dataloader, ckpt_path=os.path.abspath(cfg.checkpoint) if cfg.checkpoint else None)
    if cfg.logger.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
