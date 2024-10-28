import os
from torchvision.utils import save_image
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid
from torchvision.transforms.functional import rgb_to_grayscale
import torch
from typing import Optional, Tuple, Dict, Any
from PIL import ImageDraw, ImageFont
import torchvision


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

BACKGROUND_COLOR = (1, 1, 1)


def get_background_slot_index(masks: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    # Assuming masks is of shape (num_slots, 1, width, height)
    # Calculate the bounding box size for each mask
    bbox_sizes = []

    for i in range(masks.shape[0]):
        mask = masks[i, 0] > threshold
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


def create_segmentations(masks: torch.Tensor) -> torch.Tensor:
    sequence_length, num_slots, _, width, height = masks.size()
    background_index = get_background_slot_index(masks[0])  # Search for background at time-step 0.
    segmentations = torch.zeros((sequence_length, 3, width, height), device=masks.device)

    for slot_index in range(num_slots):
        for c in range(3):
            if slot_index == background_index:
                segmentations[:, c] += masks[:, slot_index, 0] * BACKGROUND_COLOR[c]
            else:
                segmentations[:, c] += masks[:, slot_index, 0] * colors[slot_index][c]
    return segmentations


def create_segmentation_overlay(images: torch.Tensor, masks: torch.Tensor, background_brightness: float = 0.4) -> torch.Tensor:
    sequence_length, num_slots, _, width, height = masks.size()
    background_index = get_background_slot_index(masks[0])  # Search for background at time-step 0.
    segmentations = background_brightness * rgb_to_grayscale(images, num_output_channels=3)

    for slot_index in range(num_slots):
        if slot_index == background_index:
            continue
        for c in range(3):
            segmentations[:, c] += (1 - background_brightness) * masks[:, slot_index, 0] * colors[slot_index][c]
    return segmentations


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


def draw_reward(observation, reward):
    imgs = []

    for i, img in enumerate(observation):
        img = torchvision.transforms.functional.to_pil_image(img)

        draw = ImageDraw.Draw(img)
        draw.text((0.25 * img.width, 0.8 * img.height), f"{reward[i]:.3f}", (255, 255, 255))

        imgs.append(torchvision.transforms.functional.pil_to_tensor(img))

    return torch.stack(imgs)


class LogDecomposition(LoggingCallback):
    def __init__(self, max_sequence_length: int = 10, every_n_epochs: int = 1, save_dir: Optional[str] = None) -> None:
        super().__init__(every_n_epochs, save_dir)
        self.max_sequence_length = max_sequence_length
        self.batch_index = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any],
                           batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> None:
        if self.should_log(pl_module):
            images = outputs["images"][self.batch_index].cpu().detach()
            rgbs = outputs["rgbs"][self.batch_index].detach().cpu()
            reconstructions = outputs["reconstructions"][self.batch_index].detach().cpu()
            masks = outputs["masks"][self.batch_index].detach().cpu()

            sequence_length, num_slots, _, _, _ = rgbs.size()
            n_cols = min(sequence_length, self.max_sequence_length)

            error = (reconstructions - images + 1.0) / 2

            segmentation_overlay = create_segmentation_overlay(images, masks).cpu().detach()

            segmentations = create_segmentations(masks).cpu().detach()

            combined_reconstructions = masks * rgbs
            combined_reconstructions = torch.cat([combined_reconstructions[:, s] for s in range(num_slots)],
                                                 dim=-2).detach().cpu()
            combined_reconstructions = torch.cat(
                [images, reconstructions, error, segmentation_overlay, segmentations, combined_reconstructions], dim=-2)[:, :n_cols]
            combined_reconstructions = torch.cat([combined_reconstructions[t, :, :, :] for t in range(n_cols)], dim=-1)

            rgbs = torch.cat([rgbs[:, s] for s in range(num_slots)], dim=-2)[:n_cols]
            rgbs = torch.cat([rgbs[t, :, :, :] for t in range(n_cols)], dim=-1)
            masks = torch.cat([masks[:, s] for s in range(num_slots)], dim=-2)[:n_cols]
            masks = torch.cat([masks[t, :, :, :] for t in range(n_cols)], dim=-1)

            pl_module.logger.experiment.add_image("Combined Reconstructions", combined_reconstructions, global_step=pl_module.current_epoch)
            pl_module.logger.experiment.add_image("RGB", rgbs, global_step=pl_module.current_epoch)
            pl_module.logger.experiment.add_image("Masks", masks, global_step=pl_module.current_epoch)

            if self.save_dir is not None:
                save_image(combined_reconstructions, self.save_dir + f"/savi_combined-epoch={pl_module.current_epoch}.png")
                save_image(rgbs, self.save_dir + f"/savi_rgb-epoch={pl_module.current_epoch}.png")
                save_image(masks, self.save_dir + f"/savi_masks-epoch={pl_module.current_epoch}.png")


class LogDynamicsPrediction(LoggingCallback):
    batch_index = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any],
                           batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> None:
        if self.should_log(pl_module):
            images = outputs["images"][self.batch_index].cpu().detach()
            predicted_images = outputs["predicted_images"][self.batch_index].detach().cpu()

            error = (predicted_images - images + 1.0) / 2

            context_images = [images[t] for t in range(pl_module.num_context)]
            predicted_context_images = [predicted_images[t] for t in range(pl_module.num_context)]
            context_error = [error[t] for t in range(pl_module.num_context)]

            future_images = [images[t] for t in range(pl_module.num_context, images.shape[0])]
            predicted_future_images = [predicted_images[t] for t in range(pl_module.num_context, images.shape[0])]
            future_error = [error[t] for t in range(pl_module.num_context, images.shape[0])]

            context_grid = make_grid(context_images + predicted_context_images + context_error, nrow=pl_module.num_context, padding=1)
            prediction_grid = make_grid(future_images + predicted_future_images + future_error, nrow=pl_module.num_predictions, padding=1)
            grid = torch.cat([context_grid, torch.ones(3, context_grid.size(1), 4), prediction_grid], dim=2)

            pl_module.logger.experiment.add_image("Dynamics Prediction", grid,
                                                  global_step=pl_module.current_epoch)

            if self.save_dir is not None:
                save_image(grid, self.save_dir + f"/dynamics_prediction-epoch={pl_module.current_epoch}.png")


class LogRewardPrediction(LoggingCallback):
    batch_index = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any],
                           batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> None:
        if self.should_log(pl_module):
            images = outputs["images"][self.batch_index].cpu().detach()
            predicted_images = outputs["predicted_images"][self.batch_index].detach().cpu()

            images = draw_reward(images, outputs["rewards"][self.batch_index].detach().cpu()) / 255
            predicted_images = draw_reward(predicted_images,
                                           outputs["predicted_rewards"][self.batch_index].detach().cpu()) / 255

            context_images = [images[t] for t in range(pl_module.num_context)]
            predicted_context_images = [predicted_images[t] for t in range(pl_module.num_context)]

            future_images = [images[t] for t in range(pl_module.num_context, images.shape[0])]
            predicted_future_images = [predicted_images[t] for t in range(pl_module.num_context, images.shape[0])]

            context_grid = make_grid(context_images + predicted_context_images,
                                     nrow=pl_module.num_context, padding=1)
            prediction_grid = make_grid(future_images + predicted_future_images,
                                        nrow=pl_module.num_predictions, padding=1)
            grid = torch.cat([context_grid, torch.ones(3, context_grid.size(1), 4), prediction_grid], dim=2)

            pl_module.logger.experiment.add_image("Dynamics Prediction", grid,
                                                  global_step=pl_module.current_epoch)

            if self.save_dir is not None:
                save_image(grid, self.save_dir + f"/reward_prediction-epoch={pl_module.current_epoch}.png")


