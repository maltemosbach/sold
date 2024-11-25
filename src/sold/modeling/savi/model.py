import math
from sold.modeling.savi import Corrector, Decoder, Encoder, SlotInitializer, Predictor
from sold.modeling.blocks import init_xavier_
import torch
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
