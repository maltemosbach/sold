import os
from torchvision.utils import save_image
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchvision.transforms.functional import rgb_to_grayscale
import torch
from typing import Optional, Tuple, Dict, Any, Union, List
from PIL import ImageDraw, ImageFont
import torchvision
import math
import numpy as np
import cv2


colors = [
    torch.tensor([1.0, 0.0, 0.0]),         # Slot 1 - Bright Red
    torch.tensor([1.0, 0.647, 0.0]),       # Slot 2 - Orange
    torch.tensor([1.0, 1.0, 0.0]),         # Slot 3 - Yellow
    torch.tensor([0.678, 1.0, 0.184]),     # Slot 4 - Lime Green
    torch.tensor([0.0, 1.0, 0.0]),         # Slot 5 - Green
    torch.tensor([0.0, 0.8, 0.6]),         # Slot 6 - Deeper Aqua
    torch.tensor([0.0, 0.9, 1.0]),         # Slot 7 - Cyan
    torch.tensor([0.6, 0.8, 1.0]),         # Slot 8 - Lighter Sky Blue
    torch.tensor([0.0, 0.0, 1.0]),         # Slot 9 - Blue
    torch.tensor([0.502, 0.0, 0.502]),     # Slot 10 - Purple
    torch.tensor([1.0, 0.0, 1.0]),         # Slot 11 - Magenta
    torch.tensor([1.0, 0.412, 0.706])      # Slot 12 - Pink
]

BACKGROUND_COLOR = (1, 1, 1)


def make_grid(
    tensor: torch.Tensor,
    num_columns: int,
    padding: int = 2,
    pad_color: torch.Tensor = torch.Tensor([0., 0., 0.]),
) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"tensor expected, got {type(tensor)}")

    assert pad_color.size(0) == 3, "pad_color must have 3 elements"

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(num_columns, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    #grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    grid = pad_color.unsqueeze(1).unsqueeze(2).repeat(1, height * ymaps + padding, width * xmaps + padding).to(tensor.device, tensor.dtype)

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


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
        # if slot_index == background_index:
        #     continue
        segmentations[:] += (1 - background_brightness) * masks[:, slot_index].repeat(1, 3, 1, 1) * colors[slot_index].unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(sequence_length, 1, width, height)
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
                           batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int, padding: int = 2) -> None:
        if self.should_log(pl_module):
            images = outputs["images"][self.batch_index].cpu().detach()
            predicted_images = outputs["predicted_images"][self.batch_index].detach().cpu()

            predicted_masks = outputs["predicted_masks"][self.batch_index].detach().cpu()
            predicted_rbgs = outputs["predicted_rgbs"][self.batch_index].detach().cpu()
            sequence_length, num_slots, _, _, _ = predicted_rbgs.size()

            predicted_individual_slots = predicted_masks * predicted_rbgs

            error = (predicted_images - images + 1.0) / 2

            width_spacing = 4
            height_spacing = 2

            # True vs Model rows.
            true_context = make_grid(images[:pl_module.num_context], padding=padding, num_columns=pl_module.num_context)
            model_context = make_grid(predicted_images[:pl_module.num_context], padding=padding, num_columns=pl_module.num_context)
            true_future = make_grid(images[pl_module.num_context:], padding=padding, num_columns=pl_module.num_predictions)
            model_future = make_grid(predicted_images[pl_module.num_context:], padding=padding, num_columns=pl_module.num_predictions)
            true_row = torch.cat([true_context, torch.ones(3, true_context.size(1), width_spacing), true_future], dim=2)
            model_row = torch.cat([model_context, torch.ones(3, model_context.size(1), width_spacing), model_future], dim=2)

            # Segmentation row.
            segmentation = create_segmentation_overlay(predicted_images, predicted_masks, background_brightness=0.0).cpu().detach()
            segmentation_context = make_grid(segmentation[:pl_module.num_context], padding=padding, num_columns=pl_module.num_context)
            segmentation_future = make_grid(segmentation[pl_module.num_context:], padding=padding, num_columns=pl_module.num_predictions)
            segmentation_row = torch.cat([segmentation_context, torch.ones(3, segmentation_context.size(1), width_spacing), segmentation_future], dim=2)

            # Slot rows.
            slot_rows = []
            for slot_index in range(num_slots):
                slot_context = make_grid(predicted_individual_slots[:pl_module.num_context, slot_index], padding=padding, num_columns=pl_module.num_context, pad_color=colors[slot_index])
                slot_future = make_grid(predicted_individual_slots[pl_module.num_context:, slot_index], padding=padding, num_columns=pl_module.num_predictions, pad_color=colors[slot_index])
                slot_row = torch.cat([slot_context, torch.ones(3, slot_context.size(1), width_spacing), slot_future], dim=2)
                slot_rows.append(slot_row)

            # Combine rows.
            rows = [true_row, model_row, segmentation_row] + slot_rows
            grid = []
            for row_index, row in enumerate(rows):
                grid.append(row)
                if row_index < len(rows) - 1:
                    grid.append(torch.ones(3, height_spacing, row.size(2)))
            grid = torch.cat(grid, dim=1)

            pl_module.logger.experiment.add_image("Dynamics Prediction", grid, global_step=pl_module.current_epoch)
            if self.save_dir is not None:
                save_image(grid, self.save_dir + f"/dynamics_prediction-epoch={pl_module.current_epoch}.png")


class LogRewardPrediction(LoggingCallback):
    batch_index = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any],
                           batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int, padding: int = 2) -> None:
        if self.should_log(pl_module):
            images = outputs["images"][self.batch_index].cpu().detach()
            predicted_images = outputs["predicted_images"][self.batch_index].detach().cpu()

            images = draw_reward(images, outputs["rewards"][self.batch_index].detach().cpu()) / 255
            predicted_images = draw_reward(predicted_images, outputs["predicted_rewards"][self.batch_index].detach().cpu()) / 255

            width_spacing = 4
            height_spacing = 2

            true_context = make_grid(images[:pl_module.num_context], padding=padding, num_columns=pl_module.num_context)
            model_context = make_grid(predicted_images[:pl_module.num_context], padding=padding,
                                      num_columns=pl_module.num_context)
            true_future = make_grid(images[pl_module.num_context:], padding=padding,
                                    num_columns=pl_module.num_predictions)
            model_future = make_grid(predicted_images[pl_module.num_context:], padding=padding,
                                     num_columns=pl_module.num_predictions)
            true_row = torch.cat([true_context, torch.ones(3, true_context.size(1), width_spacing), true_future], dim=2)
            model_row = torch.cat([model_context, torch.ones(3, model_context.size(1), width_spacing), model_future],
                                  dim=2)

            # Combine rows.
            rows = [true_row, model_row]
            grid = []
            for row_index, row in enumerate(rows):
                grid.append(row)
                if row_index < len(rows) - 1:
                    grid.append(torch.ones(3, height_spacing, row.size(2)))
            grid = torch.cat(grid, dim=1)

            pl_module.logger.experiment.add_image("Reward Prediction", grid, global_step=pl_module.current_epoch)
            if self.save_dir is not None:
                save_image(grid, self.save_dir + f"/reward_prediction-epoch={pl_module.current_epoch}.png")


class LogValidationEpisode(LoggingCallback):
    def __init__(self, every_n_epochs: int = 1, save_dir: Optional[str] = None) -> None:
        super().__init__(every_n_epochs, save_dir)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any],
                                batch: Any, batch_index: int) -> None:
        if self.should_log(pl_module):
            images = np.concatenate(outputs["images"])  # (episode_length, 3, width, height)
            episode_return = np.sum(outputs["rewards"])

            pl_module.logger.experiment.add_video(f"validation/episode_{batch_index}", np.expand_dims(images, 0), global_step=pl_module.current_epoch)

            fps = 15
            if self.save_dir is not None:
                out = cv2.VideoWriter(self.save_dir + f"/validation_episode-epoch={pl_module.current_epoch}-index={batch_index}-return={episode_return}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, images.shape[2:])
                for image in images:
                    image = np.moveaxis(image, 0, -1)
                    bgr_image = (cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
                    out.write(bgr_image)
                out.release()
