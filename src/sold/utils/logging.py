from abc import ABC
import json
from lightning import LightningModule, Trainer
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import os
from sold.utils.visualization import visualize_savi_decomposition
import torch
from torchvision.utils import save_image
from torchvision.io import write_video
from typing import Any, Dict, Mapping, Optional, Tuple, Union


class LoggingStepMixin(ABC):
    """Directly specify a 'logging_step' instead of using Pytorch Lightning's automatic logging."""

    @property
    def logging_step(self) -> int:
        return self.current_epoch

    def log(self, name: str, value: Any, *args, **kwargs) -> None:
        if isinstance(self.logger, ExtendedTensorBoardLogger):
            if isinstance(value, torch.Tensor):
                if value.dim() == 3:
                    self.logger.log_image(name, value, step=self.logging_step)
                elif value.dim() == 4:
                    self.logger.log_video(name, value, step=self.logging_step)
            else:
                kwargs["logger"] = False
                super().log(name, value, *args, **kwargs)
                self.logger.log_metrics({name: value}, step=self.logging_step)


class ExtendedTensorBoardLogger(TensorBoardLogger):

    def __init__(
            self,
            save_dir: _PATH,
            name: Optional[str] = "lightning_logs",
            version: Optional[Union[int, str]] = None,
            log_graph: bool = False,
            default_hp_metric: bool = True,
            prefix: str = "",
            sub_dir: Optional[_PATH] = None,
            **kwargs: Any,
    ):
        super().__init__(save_dir, name, version, log_graph, default_hp_metric, prefix, sub_dir, **kwargs)

        self.save_dirs = {}
        for subdir in ["metrics", "images", "videos"]:
            self.save_dirs[subdir] = os.path.join(self.log_dir, subdir)
            os.makedirs(self.save_dirs[subdir], exist_ok=True)

        self.current_step = 0
        self.accumulated_metrics = {}
        self.metrics_file = open(os.path.join(self.save_dirs["metrics"], "metrics.jsonl"), mode="a")

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        super().log_metrics(metrics, step)
        if self.current_step is not None and step > self.current_step:
            self._flush_metrics()
        self.current_step = step
        self.accumulated_metrics.update(metrics)

    def _flush_metrics(self) -> None:
        if self.accumulated_metrics:
            record = {"step": self.current_step, **self.accumulated_metrics}
            self.metrics_file.write(json.dumps(record) + "\n")
            self.metrics_file.flush()
            self.accumulated_metrics.clear()

    def log_image(self, name: str, image: torch.Tensor, step: int) -> None:
        self.experiment.add_image(name, image, step)
        save_image(image, os.path.join(self.save_dirs["images"], name) + f"-step={step}.png")

    def log_video(self, name: str, video: torch.Tensor, step: int, fps: int = 10) -> None:
        self.experiment.add_video(name, np.expand_dims(video.cpu().numpy(), 0), global_step=step)
        name = name.replace("/", "_")  # Turn tensorboard grouping into valid file name.
        write_video(os.path.join(self.save_dirs["videos"], name) + f"-step={step}.mp4", (video.permute(0, 2, 3, 1) * 255).to(torch.uint8), fps)


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

            pl_module.log("savi_decomposition", combined_reconstructions)
            pl_module.log("rgb_predictions", rgbs)
            pl_module.log("mask_predictions", masks)
