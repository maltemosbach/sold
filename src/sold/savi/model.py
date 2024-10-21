"""
Implementation of the SAVi model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT

from sold.utils.model_utils import init_xavier_

from .corrector import Corrector
from .decoder import Decoder
from .encoder import Encoder
from .initializer import SlotInitializer
from .predictor import Predictor
from lightning import LightningModule
from typing import Tuple, Any

from sold.utils.model_blocks import SoftPositionEmbed

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class SAVi(LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder, initializer: SlotInitializer, predictor: Predictor,
                 corrector: Corrector, image_size: Tuple[int, int] = (64, 64), num_slots: int = 6, slot_dim: int = 64,
                 learning_rate: float = 0.0001, warmup_epochs: int = 2, max_epochs: int = 1000) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.initializer = initializer
        self.predictor = predictor
        self.corrector = corrector

        # Encoder MLP after the actual encoder.
        self.out_features = self.encoder.num_channels[-1]
        self.encoder_positional_encoding = SoftPositionEmbed(
                hidden_size=self.out_features,
                resolution=image_size
            )
        mlp_encoder_dim = corrector.feature_dim
        self.encoder_mlp = nn.Sequential(
            nn.LayerNorm(self.out_features),
            nn.Linear(self.out_features, mlp_encoder_dim),
            nn.ReLU(),
            nn.Linear(mlp_encoder_dim, mlp_encoder_dim),
        )

        self.decoder_positional_encoding = SoftPositionEmbed(
            hidden_size=slot_dim,
            resolution=image_size
        )

        self.image_size = image_size

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, self.warmup_epochs, self.max_epochs)
        return [optimizer,],  [lr_scheduler,]

    def forward(self, input, num_imgs=10, prior_slots=None, step_offset=0, reconstruct=True, **kwargs):
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

        # initializing slots by randomly sampling them or encoding some representations (e.g. BBox)
        predicted_slots = self.initializer(batch_size=input.shape[0],
                                           **kwargs) if prior_slots is None else self.predictor(prior_slots)

        # recursively mapping video frames into object slots
        for t in range(num_imgs):
            imgs = input[:, t]
            img_feats = self.encode(imgs)
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

    def encode(self, input):
        """
        Encoding an image into image features
        """

        # TODO: All of should get refactored into the SAVi encoder later.
        B, C, H, W = input.shape

        # encoding input frame and adding positional encodding
        x = self.encoder(input)  # x ~ (B,C,H,W)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_positional_encoding(x)  # x ~ (B,H,W,C)

        # further encodding with 1x1 Conv (implemented as shared MLP)
        x = torch.flatten(x, 1, 2)
        x = self.encoder_mlp(x)  # x ~ (B, N, Dim)
        return x

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
            (1, self.image_size[0], self.image_size[1], 1)
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

    def training_step(self, batch, batch_index: int) -> STEP_OUTPUT:
        print("Training step")
        print("batch:", batch)

        input()
        return 0.0
