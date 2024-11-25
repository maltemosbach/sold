from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar, _update_n
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
import math
import numpy as np
import os
from PIL import ImageDraw
import torch
import torchvision
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.utils import save_image
from torchvision.io import write_video
from typing import Any, Dict, Union, Mapping, Optional, Tuple


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


class OnlineProgressBar(TQDMProgressBar):
    """Custom progress bar for online RL training loops."""

    def on_train_start(self, *_: Any) -> None:
        super().on_train_start()
        self.train_progress_bar.reset(total=self.total_train_batches)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_progress_bar.set_description(f"Steps {trainer.current_epoch}, Episodes {pl_module.replay_buffer.num_episodes}")

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        n = trainer.current_epoch
        _update_n(self.train_progress_bar, n)

    @property
    def total_train_batches(self) -> Union[int, float]:
        return self.trainer.max_steps


class CustomTensorBoardLogger(TensorBoardLogger):
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        super().log_metrics(metrics, self._model.current_step)

    def log_image(self, name: str, image: torch.Tensor) -> None:
        # Add to Tensorboard.
        self.experiment.add_image(name, image, self._model.current_step)

        # Save image to disk.
        save_dir = os.path.join(self.log_dir, "images")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(image, os.path.join(save_dir, name) + f"-current_step={self._model.current_step}.png")

    def log_video(self, name: str, video: torch.Tensor, fps: int = 10) -> None:
        # Add to Tensorboard.
        self.experiment.add_video(name, np.expand_dims(video.cpu().numpy(), 0), global_step=self._model.current_step)

        # Save video to disk.
        name = name.replace("/", "_")  # Turn tensorboard grouping into valid file name.
        save_dir = os.path.join(self.log_dir, "videos")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        write_video(os.path.join(save_dir, name) + f"-current_step={self._model.current_step}.mp4", (video.permute(0, 2, 3, 1) * 255).to(torch.uint8), fps)


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


def create_segmentation_overlay(images: torch.Tensor, masks: torch.Tensor, background_brightness: float = 0.4) -> torch.Tensor:
    sequence_length, num_slots, _, width, height = masks.size()
    segmentations = background_brightness * rgb_to_grayscale(images, num_output_channels=3)

    for slot_index in range(num_slots):
        # if slot_index == background_index:
        #     continue
        segmentations[:] += (1 - background_brightness) * masks[:, slot_index].repeat(1, 3, 1, 1) * colors[slot_index].unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(sequence_length, 1, width, height)
    return segmentations


def visualize_savi_decomposition(images, reconstructions, rgbs, masks, max_sequence_length: Optional[int] = 10,
                                 padding: int = 2) -> torch.Tensor:
    images = images.cpu()
    reconstructions = reconstructions.cpu()
    rgbs = rgbs.cpu()
    masks = masks.cpu()

    height_spacing = 2
    sequence_length, num_slots, _, _, _ = rgbs.size()
    n_cols = min(sequence_length, max_sequence_length) if max_sequence_length is not None else sequence_length

    true_row = make_grid(images[:n_cols], padding=padding, num_columns=n_cols)
    reconstruction_row = make_grid(reconstructions[:n_cols], padding=padding, num_columns=n_cols)

    segmentations = create_segmentation_overlay(images[:n_cols], masks[:n_cols], background_brightness=0.0).cpu().detach()
    segmentation_row = make_grid(segmentations, padding=padding, num_columns=n_cols)
    individual_slots = masks * rgbs

    # Slot rows.
    slot_rows = []
    for slot_index in range(num_slots):
        slots = make_grid(individual_slots[:n_cols, slot_index], padding=padding, num_columns=n_cols, pad_color=colors[slot_index])
        slot_rows.append(slots)

    # Combine rows.
    rows = [true_row, reconstruction_row, segmentation_row] + slot_rows
    grid = []
    for row_index, row in enumerate(rows):
        grid.append(row)
        if row_index < len(rows) - 1:
            grid.append(torch.ones(3, height_spacing, row.size(2)))

    grid = torch.cat(grid, dim=1)
    return grid


def visualize_dynamics_prediction(images, predicted_images, predicted_rgbs, predicted_masks, num_context: int, padding: int = 2) -> torch.Tensor:
    images = images.cpu()
    predicted_images = predicted_images.cpu()
    predicted_rgbs = predicted_rgbs.cpu()
    predicted_masks = predicted_masks.cpu()

    sequence_length, num_slots, _, _, _ = predicted_rgbs.size()

    predicted_individual_slots = predicted_masks * predicted_rgbs

    num_predictions = sequence_length - num_context

    width_spacing = 4
    height_spacing = 2

    # True vs Model rows.
    true_context = make_grid(images[:num_context], padding=padding, num_columns=num_context)
    model_context = make_grid(predicted_images[:num_context], padding=padding, num_columns=num_context)
    true_future = make_grid(images[num_context:], padding=padding, num_columns=num_predictions)
    model_future = make_grid(predicted_images[num_context:], padding=padding, num_columns=num_predictions)
    true_row = torch.cat([true_context, torch.ones(3, true_context.size(1), width_spacing), true_future], dim=2)
    model_row = torch.cat([model_context, torch.ones(3, model_context.size(1), width_spacing), model_future], dim=2)

    # Segmentation row.
    segmentation = create_segmentation_overlay(predicted_images, predicted_masks, background_brightness=0.0).cpu().detach()
    segmentation_context = make_grid(segmentation[:num_context], padding=padding, num_columns=num_context)
    segmentation_future = make_grid(segmentation[num_context:], padding=padding, num_columns=num_predictions)
    segmentation_row = torch.cat([segmentation_context, torch.ones(3, segmentation_context.size(1), width_spacing), segmentation_future], dim=2)

    # Slot rows.
    slot_rows = []
    for slot_index in range(num_slots):
        slot_context = make_grid(predicted_individual_slots[:num_context, slot_index], padding=padding, num_columns=num_context, pad_color=colors[slot_index])
        slot_future = make_grid(predicted_individual_slots[num_context:, slot_index], padding=padding, num_columns=num_predictions, pad_color=colors[slot_index])
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
    return grid


def visualize_reward_prediction(images, predicted_images, rewards, predicted_rewards, num_context: int, padding: int = 2) -> torch.Tensor:
    images = images.cpu()
    predicted_images = predicted_images.cpu()

    images = draw_reward(images, rewards.detach().cpu()) / 255
    predicted_images = draw_reward(predicted_images, predicted_rewards.detach().cpu()) / 255

    sequence_length, _, _, _ = images.size()
    num_predictions = sequence_length - num_context

    width_spacing = 4
    height_spacing = 2

    true_context = make_grid(images[:num_context], padding=padding, num_columns=num_context)
    model_context = make_grid(predicted_images[:num_context], padding=padding,
                              num_columns=num_context)
    true_future = make_grid(images[num_context:], padding=padding,
                            num_columns=num_predictions)
    model_future = make_grid(predicted_images[num_context:], padding=padding,
                             num_columns=num_predictions)
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
    return grid


def draw_reward(observation, reward):
    imgs = []
    for i, img in enumerate(observation):
        img = torchvision.transforms.functional.to_pil_image(img)
        draw = ImageDraw.Draw(img)
        draw.text((0.25 * img.width, 0.8 * img.height), f"{reward[i]:.3f}", (255, 255, 255))
        imgs.append(torchvision.transforms.functional.pil_to_tensor(img))
    return torch.stack(imgs)


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


class LogDecomposition(LoggingCallback):
    def __init__(self, every_n_epochs: int = 1, save_dir: Optional[str] = None) -> None:
        super().__init__(every_n_epochs, save_dir)
        self.batch_index = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any],
                           batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> None:
        if self.should_log(pl_module):
            images = outputs["images"][self.batch_index].cpu().detach()
            rgbs = outputs["rgbs"][self.batch_index].detach().cpu()
            reconstructions = outputs["reconstructions"][self.batch_index].detach().cpu()
            masks = outputs["masks"][self.batch_index].detach().cpu()

            combined_reconstructions = visualize_savi_decomposition(images, reconstructions, rgbs, masks)

            sequence_length, num_slots, _, _, _ = rgbs.size()
            n_cols = sequence_length

            rgbs = torch.cat([rgbs[:, s] for s in range(num_slots)], dim=-2)[:n_cols]
            rgbs = torch.cat([rgbs[t, :, :, :] for t in range(n_cols)], dim=-1)
            masks = torch.cat([masks[:, s] for s in range(num_slots)], dim=-2)[:n_cols]
            masks = torch.cat([masks[t, :, :, :] for t in range(n_cols)], dim=-1)

            pl_module.logger.experiment.add_image("savi_decomposition", combined_reconstructions, global_step=pl_module.current_epoch)
            pl_module.logger.experiment.add_image("rgb_predictions", rgbs, global_step=pl_module.current_epoch)
            pl_module.logger.experiment.add_image("mask_predictions", masks, global_step=pl_module.current_epoch)

            if self.save_dir is not None:
                save_image(combined_reconstructions, self.save_dir + f"/savi_combined-epoch={pl_module.current_epoch}.png")
                save_image(rgbs, self.save_dir + f"/savi_rgb-epoch={pl_module.current_epoch}.png")
                save_image(masks, self.save_dir + f"/savi_masks-epoch={pl_module.current_epoch}.png")
