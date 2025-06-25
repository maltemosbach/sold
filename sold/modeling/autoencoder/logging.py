from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
import os
import torch
from typing import Any, Dict, Optional, Tuple


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


class LogReconstruction(LoggingCallback):
    def __init__(self, every_n_epochs: int = 1, save_dir: Optional[str] = None) -> None:
        super().__init__(every_n_epochs, save_dir)
        self.batch_index = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any],
                           batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> None:
        if self.should_log(pl_module):
            pl_module.log("reconstruction", pl_module.autoencoder.visualize_reconstruction({k: v[0] for k, v in outputs.items() if v.ndim > 0}))
