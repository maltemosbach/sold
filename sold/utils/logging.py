from abc import ABC
from collections import defaultdict
import json
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import os
import torch
from torchvision.utils import save_image
from torchvision.io import write_video
from typing import Any, Mapping, Optional, Tuple, Union


class LoggingStepMixin(ABC):
    """Directly specify a 'logging_step' instead of using Pytorch Lightning's automatic logging."""

    def __init__(self, max_logging_freq: int = 1) -> None:
        super().__init__()
        self.max_logging_freq = max_logging_freq
        self.last_logging_step = defaultdict(lambda: -max_logging_freq)

    @property
    def logging_step(self) -> int:
        return self.current_epoch

    def log(self, name: str, value: Any, *args, **kwargs) -> None:
        if isinstance(self.logger, ExtendedTensorBoardLogger):
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                if value.dim() == 3:
                    self.logger.log_image(name, value, step=self.logging_step)
                elif value.dim() == 4:
                    self.logger.log_video(name, value, step=self.logging_step)
            else:
                value = self._LightningModule__to_tensor(value, name).item()
                if kwargs.get("logger", True):
                    # Log metrics after the last batch.
                    if self.trainer.is_last_batch:
                        if self.logging_step - self.last_logging_step[name] >= self.max_logging_freq:
                            self.last_logging_step[name] = self.logging_step
                            self.logger.log_metrics({name: value}, step=self.logging_step)

                kwargs["logger"] = False
                super().log(name, value, *args, **kwargs)

        else:
            super().log(name, value, *args, **kwargs)

    def log_gradients(self, model_names: Tuple[str, ...]) -> None:
        for model_name in model_names:
            model = getattr(self, model_name)
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            norm = torch.cat(grads).norm()
            self.log(f"gradients/{model_name}", norm.item())


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
        if video.dtype != torch.uint8:
            video = (video * 255).to(torch.uint8)

        self.experiment.add_video(name, np.expand_dims(video.cpu().numpy(), 0), global_step=step)
        name = name.replace("/", "_")  # Turn tensorboard grouping into valid file name.
        write_video(os.path.join(self.save_dirs["videos"], name) + f"-step={step}.mp4", video.permute(0, 2, 3, 1), fps)


