from functools import partial
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from lightning.pytorch.callbacks import Callback
import math
from omegaconf import DictConfig
from sold.models.savi import Corrector, Decoder, Encoder, SlotInitializer, Predictor
from sold.utils.model_blocks import SoftPositionEmbed
from sold.utils.model_utils import init_xavier_
from sold.utils.visualization import visualize_decomposition

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.optim import Optimizer, lr_scheduler
import torch.nn as nn


class SAVi(nn.Module):
    def __init__(self, corrector: Corrector, predictor: Predictor, encoder: Encoder, decoder: Decoder,
                 initializer: SlotInitializer) -> None:
        super().__init__()
        self.corrector = corrector
        self.predictor = predictor
        self.encoder = encoder
        self.decoder = decoder
        self.initializer = initializer
        self.num_slots = corrector.num_slots
        self.slot_dim = corrector.slot_dim
        self._initialize_parameters()

    @torch.no_grad()
    def _initialize_parameters(self):
        """
        Initalization of the model parameters
        Adapted from:
            https://github.com/addtt/object-centric-library/blob/main/models/slot_attention/trainer.py
        """
        init_xavier_(self)
        torch.nn.init.zeros_(self.corrector.gru.bias_ih)
        torch.nn.init.zeros_(self.corrector.gru.bias_hh)
        torch.nn.init.orthogonal_(self.corrector.gru.weight_hh)
        if hasattr(self.corrector, "slots_mu"):
            limit = math.sqrt(6.0 / (1 + self.corrector.dim_slots))
            torch.nn.init.uniform_(self.corrector.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.corrector.slots_sigma, -limit, limit)
        return

    def forward(self, input, prior_slots=None, step_offset=0, reconstruct=True, **kwargs):
        """
        Forward pass through the model

        Args:
        -----
        input: torch Tensor
            Images to process with SAVi. Shape is (B, NumImgs, C, H, W)
        num_imgs: int
            Number of images to recursively encode into object slots.

        Returns:
        --------
        slot_history: torch Tensor
            Object slots encoded at every time step. Shape is (B, num_imgs, num_slots, slot_dim)
        recons_history: torch Tensor
            Rendered video frames by decoding and combining the slots. Shape is (B, num_imgs, C, H, W)
        ind_recons_history: torch Tensor
            Rendered objects by decoding slots. Shape is (B, num_imgs, num_slots, C, H, W)
        masks_history: torch Tensor
            Rendered object masks by decoding slots. Shape is (B, num_imgs, num_slots, 1, H, W)
        """
        slot_history = []
        reconstruction_history = []
        individual_recons_history = []
        masks_history = []

        num_imgs = input.shape[1]

        # initializing slots by randomly sampling them or encoding some representations (e.g. BBox)
        predicted_slots = self.initializer(batch_size=input.shape[0],
                                           **kwargs) if prior_slots is None else self.predictor(prior_slots)

        # recursively mapping video frames into object slots
        for t in range(num_imgs):
            imgs = input[:, t]
            img_feats = self.encoder(imgs)
            slots = self.apply_attention(img_feats, predicted_slots=predicted_slots, step=t + step_offset)
            predicted_slots = self.predictor(slots)
            slot_history.append(slots)
            if reconstruct:
                rgb, masks = self.decoder(slots)
                recon_combined = torch.sum(rgb * masks, dim=1)
                reconstruction_history.append(recon_combined)
                individual_recons_history.append(rgb)
                masks_history.append(masks)

        slot_history = torch.stack(slot_history, dim=1)
        if reconstruct:
            reconstruction_history = torch.stack(reconstruction_history, dim=1)
            individual_recons_history = torch.stack(individual_recons_history, dim=1)
            masks_history = torch.stack(masks_history, dim=1)
        return (
        slot_history, reconstruction_history, individual_recons_history, masks_history) if reconstruct else slot_history

    def apply_attention(self, x, predicted_slots=None, step=0):
        slots = self.corrector(x, slots=predicted_slots, step=step)  # slots ~ (B, N_slots, Slot_dim)
        return slots


class LogSAViDecomposition(Callback):



class SAViTrainer(LightningModule):
    def __init__(self, savi: SAVi, optimizer: partial[Optimizer], scheduler: partial[lr_scheduler]) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.savi = savi
        self.create_optimizer = optimizer
        self.create_scheduler = scheduler

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.create_optimizer(params=self.parameters())
        scheduler = {
            'scheduler': self.create_scheduler(optimizer=optimizer),
            'interval': 'step',
            'name': 'Learning Rate',
        }
        return [optimizer,], [scheduler,]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        images, actions = batch
        loss = self.compute_reconstruction_loss(images, log_visualizations=batch_index == 0)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        images, actions = batch
        loss = self.compute_reconstruction_loss(images)
        self.log("val_loss", loss)
        return None

    def compute_reconstruction_loss(self, images: torch.Tensor, log_visualizations: bool = False, batch_index: int = 0) -> torch.Tensor:
        slots, reconstructions, rgbs, masks = self(images)
        if log_visualizations:
            visualize_decomposition(
                images[batch_index], reconstructions[batch_index], rgbs[batch_index].clamp(0, 1),
                masks[batch_index].clamp(0, 1), self.logger.experiment, self.current_epoch,
                savepath=self.logger.log_dir + "/images")
        return F.mse_loss(reconstructions.clamp(0, 1), images.clamp(0, 1))


def load_savi(checkpoint_path: str, finetune: DictConfig):
    savi_model = SAVi.load_from_checkpoint(checkpoint_path)
    savi_model.finetune = finetune
    savi_model.automatic_optimization = False
    return savi_model
