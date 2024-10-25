import os
from torchvision.utils import save_image
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
import torch
from typing import Optional, Tuple

colors = [
    (1, 0, 0),          # Red
    (0, 1, 0),          # Green
    (0, 0, 1),          # Blue
    (1, 1, 0),          # Yellow
    (0.5, 0, 0.5),      # Purple
    (0, 1, 1),          # Cyan
    (1, 0.65, 0),       # Orange
    (1, 0, 1),          # Magenta
    (0.75, 1, 0),       # Lime
    (0.65, 0.16, 0.16), # Brown
    (1, 0.75, 0.8),     # Pink
    (0.5, 0.5, 0.5)     # Gray
]


class SAViDecomposition(Callback):
    def __init__(self, every_n_epochs: int = 1, max_sequence_length: int = 10, save_dir: Optional[str] = None) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.max_sequence_length = max_sequence_length
        self.save_dir = save_dir
        self.batch_index = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.current_epoch % self.every_n_epochs == 0:
            images, reconstructions, rgbs, masks = pl_module.training_step_outputs
            sequence_length, num_slots, _, _, _ = rgbs[self.batch_index].size()
            n_cols = min(sequence_length, self.max_sequence_length)

            images = images[self.batch_index].cpu().detach()
            reconstructions = reconstructions[self.batch_index].cpu().detach()
            error = (reconstructions - images + 1.0) / 2
            segmentations = self.create_segmentations(masks[self.batch_index]).cpu().detach()

            combined_reconstructions = masks[self.batch_index] * rgbs[self.batch_index]
            combined_reconstructions = torch.cat([combined_reconstructions[:, s] for s in range(num_slots)],
                                                 dim=-2).detach().cpu()
            combined_reconstructions = torch.cat(
                [images, reconstructions, error, segmentations, combined_reconstructions], dim=-2)[:, :n_cols]
            combined_reconstructions = torch.cat([combined_reconstructions[t, :, :, :] for t in range(n_cols)], dim=-1)

            rgbs = torch.cat([rgbs[self.batch_index, :, s] for s in range(num_slots)], dim=-2)[:n_cols]
            rgbs = torch.cat([rgbs[t, :, :, :] for t in range(n_cols)], dim=-1)
            masks = torch.cat([masks[self.batch_index, :, s] for s in range(num_slots)], dim=-2)[:n_cols]
            masks = torch.cat([masks[t, :, :, :] for t in range(n_cols)], dim=-1)

            pl_module.logger.experiment.add_image("Combined Reconstructions", combined_reconstructions, global_step=pl_module.current_epoch)
            pl_module.logger.experiment.add_image("RGB", rgbs, global_step=pl_module.current_epoch)
            pl_module.logger.experiment.add_image("Masks", masks, global_step=pl_module.current_epoch)

            if self.save_dir is not None:
                save_dir = os.path.join(pl_module.logger.log_dir, self.save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_image(combined_reconstructions, save_dir + f"/savi_combined-epoch={pl_module.current_epoch}.png")
                save_image(rgbs, save_dir + f"/savi_rgb-epoch={pl_module.current_epoch}.png")
                save_image(masks, save_dir + f"/savi_masks-epoch={pl_module.current_epoch}.png")

    def get_background_slot_index(self, masks: torch.Tensor) -> torch.Tensor:
        # Assuming masks is of shape (num_slots, 1, width, height)
        # Calculate the bounding box size for each mask
        bbox_sizes = []
        for i in range(masks.shape[0]):
            mask = masks[i, 0]
            rows = torch.any(mask, dim=1)
            cols = torch.any(mask, dim=0)
            if torch.any(rows) and torch.any(cols):
                  rmin, rmax = torch.where(rows)[0][[0, -1]]
                  cmin, cmax = torch.where(cols)[0][[0, -1]]
                  bbox_sizes.append((rmax - rmin) * (cmax - cmin))
            else:
                  bbox_sizes.append(0)  # Assign size 0 if the mask is empty

        # The background is likely the mask with the largest bounding box
        background_index = torch.argmax(torch.tensor(bbox_sizes))
        return background_index

    def create_segmentations(self, masks: torch.Tensor) -> torch.Tensor:
        sequence_length, num_slots, _, width, height = masks.size()
        background_index = self.get_background_slot_index(masks)
        segmentations = torch.zeros((sequence_length, 3, width, height), device=masks.device)
        for slot_index in range(num_slots):
            if slot_index == background_index:
                continue
            for c in range(3):
                segmentations[:, c] += masks[:, slot_index, 0] * colors[slot_index][c]
        return segmentations
