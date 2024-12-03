import functools
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
        """Adapted from: https://github.com/addtt/object-centric-library/blob/main/models/slot_attention/trainer.py"""

        init_xavier_(self)
        torch.nn.init.zeros_(self.corrector.gru.bias_ih)
        torch.nn.init.zeros_(self.corrector.gru.bias_hh)
        torch.nn.init.orthogonal_(self.corrector.gru.weight_hh)
        if hasattr(self.corrector, "slots_mu"):
            limit = math.sqrt(6.0 / (1 + self.corrector.dim_slots))
            torch.nn.init.uniform_(self.corrector.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.corrector.slots_sigma, -limit, limit)

    def forward(self, images: torch.Tensor, actions: torch.Tensor, prior_slots=None, step_offset=0, reconstruct=True, **kwargs):
        """

        Args:
            images (torch.Tensor): Image sequence of shape (B, sequence_length, C, H, W).
            actions (torch.Tensor): Action sequence of shape (B, sequence_length - 1, action_dim).

        Returns:
            torch.Tensor: Slots encoded at every time step of shape (B, sequence_length, num_slots, slot_dim)
            torch.Tensor: Reconstructed video frames by decoding and combining the slots of shape (B, sequence_length, C, H, W)
            torch.Tensor: Rendered objects of individual slots. Shape is (B, sequence_length, num_slots, C, H, W)
            torch.Tensor: Rendered object masks of individual slots. Shape is (B, sequence_length, num_slots, 1, H, W)
        """

        # Initialize predictor via the action_dim.
        if not isinstance(self.predictor, Predictor):
            self.predictor = self.predictor(action_dim=actions.shape[-1]).to(images.device)

        slots_sequence = []
        reconstruction_sequence = []
        rgbs_sequence = []
        masks_sequence = []

        sequence_length = images.shape[1]

        # Initialize slots by randomly sampling them or encoding some representations (e.g. BBox)
        predicted_slots = self.initializer(batch_size=images.shape[0]) if prior_slots is None else self.predictor(prior_slots, actions[:, 0])

        # Recursively map video frames into slots.
        for t in range(sequence_length):
            imgs = images[:, t]
            img_feats = self.encoder(imgs)
            slots = self.apply_attention(img_feats, predicted_slots=predicted_slots, step=t + step_offset)
            if t < sequence_length - 1:
               predicted_slots = self.predictor(slots, actions[:, t])
            slots_sequence.append(slots)
            if reconstruct:
                rgb, masks = self.decoder(slots)
                recon_combined = torch.sum(rgb * masks, dim=1)
                reconstruction_sequence.append(recon_combined)
                rgbs_sequence.append(rgb)
                masks_sequence.append(masks)

        slots_sequence = torch.stack(slots_sequence, dim=1)
        if reconstruct:
            reconstruction_sequence = torch.stack(reconstruction_sequence, dim=1)
            rgbs_sequence = torch.stack(rgbs_sequence, dim=1)
            masks_sequence = torch.stack(masks_sequence, dim=1)
        return (
        slots_sequence, reconstruction_sequence, rgbs_sequence, masks_sequence) if reconstruct else slots_sequence

    def apply_attention(self, x, predicted_slots=None, step=0):
        slots = self.corrector(x, slots=predicted_slots, step=step)  # slots ~ (B, N_slots, Slot_dim)
        return slots
