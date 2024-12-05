from abc import ABC
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import os
from sold.utils.visualization import visualize_savi_decomposition
import torch
from torchvision.utils import save_image
from torchvision.io import write_video
from typing import Any, Dict, Optional, Tuple


class ExtendedLoggingModule(LightningModule, ABC):

    def log(self, name: str, value: Any, *args, **kwargs) -> None:
        if isinstance(self.logger, ExtendedTensorBoardLogger):
            if isinstance(value, torch.Tensor):
                if value.dim() == 3:
                    return self.logger.log_image(name, value, step=self.current_epoch)
                elif value.dim() == 4:
                    return self.logger.log_video(name, value, step=self.current_epoch)

        return super().log(name, value, *args, **kwargs)


class ExtendedTensorBoardLogger(TensorBoardLogger):
    def log_image(self, name: str, image: torch.Tensor, step: int) -> None:
        # Add to Tensorboard.
        self.experiment.add_image(name, image, step)

        # Save image to disk.
        save_dir = os.path.join(self.log_dir, "images")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(image, os.path.join(save_dir, name) + f"-step={step}.png")

    def log_video(self, name: str, video: torch.Tensor, step: int, fps: int = 10) -> None:
        # Add to Tensorboard.
        self.experiment.add_video(name, np.expand_dims(video.cpu().numpy(), 0), global_step=step)

        # Save video to disk.
        name = name.replace("/", "_")  # Turn tensorboard grouping into valid file name.
        save_dir = os.path.join(self.log_dir, "videos")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        write_video(os.path.join(save_dir, name) + f"-step={step}.mp4", (video.permute(0, 2, 3, 1) * 255).to(torch.uint8), fps)


class LoggingCallback(Callback):
    def __init__(self, every_n_epochs: int = 1, save_dir: Optional[str] = None) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.save_dir = save_dir

    def should_log(self, pl_module: LightningModule) -> bool:
        """Log after last batch every n epochs."""
        return pl_module.trainer.is_last_batch and pl_module.current_epoch % self.every_n_epochs == 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.save_dir is not None:
            self.save_dir = os.path.join(pl_module.logger.log_dir, self.save_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)


class LogDecomposition(LoggingCallback):
    def __init__(self, every_n_epochs: int = 1, save_dir: Optional[str] = None) -> None:
        super().__init__(every_n_epochs, save_dir)
        self.batch_index = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any],
                           batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> None:
        if self.should_log(pl_module):
            images = outputs["images"][self.batch_index].cpu().detach()
            rgbs = outputs["rgbs"][self.batch_index].detach().cpu()
            reconstructions = outputs["reconstructions"][self.batch_index].detach().cpu()
            masks = outputs["masks"][self.batch_index].detach().cpu()

            combined_reconstructions = visualize_savi_decomposition(images, reconstructions, rgbs, masks)

            sequence_length, num_slots, _, _, _ = rgbs.size()
            n_cols = sequence_length

            rgbs = torch.cat([rgbs[:, s] for s in range(num_slots)], dim=-2)[:n_cols]
            rgbs = torch.cat([rgbs[t, :, :, :] for t in range(n_cols)], dim=-1)
            masks = torch.cat([masks[:, s] for s in range(num_slots)], dim=-2)[:n_cols]
            masks = torch.cat([masks[t, :, :, :] for t in range(n_cols)], dim=-1)

            pl_module.logger.log_image("savi_decomposition", combined_reconstructions)
            pl_module.logger.log_image("rgb_predictions", rgbs)
            pl_module.logger.log_image("mask_predictions", masks)
