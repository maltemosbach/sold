from functools import partial
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
import math
from omegaconf import DictConfig
from sold.savi import Corrector, Decoder, Encoder, SlotInitializer, Predictor
from sold.utils.model_blocks import SoftPositionEmbed
from sold.utils.model_utils import init_xavier_
from sold.utils.visualization import visualize_decompositions
import torch
import torch.nn.functional as F
from typing import Tuple
from torch.optim import Optimizer, lr_scheduler


class SAVi(LightningModule):
    def __init__(self, corrector: Corrector, predictor: Predictor, encoder: Encoder, decoder: Decoder,
                 initializer: SlotInitializer, optimizer: partial[Optimizer], scheduler: partial[lr_scheduler]) -> None:
        """Create a trainable SAVi model by the combination of its components (corrector-initializer) and optimization
        parameters."""
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.corrector = corrector
        self.predictor = predictor
        self.encoder = encoder
        self.decoder = decoder
        self.initializer = initializer
        self.create_optimizer = optimizer
        self.create_scheduler = scheduler

        self.decoder_positional_encoding = SoftPositionEmbed(
            hidden_size=corrector.slot_dim,
            resolution=encoder.image_size
        )
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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.create_optimizer(params=self.parameters())
        scheduler = {
            'scheduler': self.create_scheduler(optimizer=optimizer),
            'interval': 'step',
            'name': 'Learning Rate',
        }
        return [optimizer,], [scheduler,]

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
                recon_combined, (recons, masks) = self.decode(slots)
                reconstruction_history.append(recon_combined)
                individual_recons_history.append(recons)
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

    def decode(self, slots):
        """
        Decoding slots into objects and masks
        """
        B, N_S, S_DIM = slots.shape

        # adding broadcasing for the dissentangled decoder
        slots = slots.reshape((-1, 1, 1, S_DIM))
        slots = slots.repeat(
            (1, self.encoder.image_size[0], self.encoder.image_size[1], 1)
        )  # slots ~ (B*N_slots, H, W, Slot_dim)

        # adding positional embeddings to reshaped features
        slots = self.decoder_positional_encoding(slots)  # slots ~ (B*N_slots, H, W, Slot_dim)
        slots = slots.permute(0, 3, 1, 2)

        y = self.decoder(slots)  # slots ~ (B*N_slots, Slot_dim, H, W)

        # recons and masks have shapes [B, N_S, C, H, W] & [B, N_S, 1, H, W] respectively
        y_reshaped = y.reshape(B, -1, self.encoder.in_channels + 1, y.shape[2], y.shape[3])
        recons, masks = y_reshaped.split([self.encoder.in_channels, 1], dim=2)

        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)

        return recon_combined, (recons, masks)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        loss = self.compute_reconstruction_loss(batch, log_visualizations=batch_index == 0)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        loss = self.compute_reconstruction_loss(batch)
        self.log("val_loss", loss)
        return None

    def compute_reconstruction_loss(self, batch: Tuple[torch.Tensor, torch.Tensor], log_visualizations: bool = False) -> torch.Tensor:
        videos, actions = batch
        slots, reconstructions, individual_reconstructions, masks = self(videos)
        if log_visualizations:
            self._log_visualizations(videos, reconstructions, individual_reconstructions, masks)
        return F.mse_loss(reconstructions.clamp(0, 1), videos.clamp(0, 1))

    @torch.no_grad()
    def _log_visualizations(self, videos: torch.Tensor, reconstructions: torch.Tensor,
                            individual_reconstructions: torch.Tensor, masks: torch.Tensor) -> None:
        visualize_decompositions(videos[0], reconstructions[0], individual_reconstructions[0].clamp(0, 1),
                                 masks[0].clamp(0, 1), self.logger.experiment, self.current_epoch,
                                 savepath=self.logger.log_dir + "/images")


def load_savi(checkpoint_path: str, finetune: DictConfig):
    model = SAVi.load_from_checkpoint(checkpoint_path)
    model.finetune = finetune
    model.automatic_optimization = False
    return model
