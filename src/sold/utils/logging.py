import cv2
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar, _update_n
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
import math
import numpy as np
import os
import torch
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.utils import save_image
from typing import Any, Dict, List, Union, Mapping, Optional

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
        save_image(image, os.path.join(save_dir, name) + name + f"-current_step={self._model.current_step}.png")

    def log_video(self, name: str, video: torch.Tensor, fps: int = 10) -> None:
        # Add to Tensorboard.
        self.experiment.add_video(name, np.expand_dims(video.cpu().numpy(), 0), global_step=self._model.current_step)

        # Save video to disk.
        save_dir = os.path.join(self.log_dir, "videos")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        out = cv2.VideoWriter(os.path.join(save_dir, name) + f"-current_step={self._model.current_step}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, video.shape[2:])
        for image in video.cpu().numpy():
            image = np.moveaxis(image, 0, -1)
            bgr_image = (cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
            out.write(bgr_image)
        out.release()


def log_eval_episode(logger, episode: Dict[str, List], current_step: int, episode_index: int) -> None:
    images = np.stack(episode["obs"])  # (episode_length, 3, width, height)
    episode_return = np.sum(episode["reward"])

    # Log to Tensorboard.
    logger.experiment.add_video(f"eval/episode_{episode_index}", np.expand_dims(images, 0), global_step=current_step)

    # Save video to disk.
    fps = 10
    save_dir = os.path.join(logger.log_dir, "videos")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out = cv2.VideoWriter(save_dir + f"/eval_episode-epoch={current_step}-index={episode_index}-return={episode_return}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, images.shape[2:])
    for image in images:
        image = np.moveaxis(image, 0, -1)
        bgr_image = (cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
        out.write(bgr_image)
    out.release()


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
    background_index = get_background_slot_index(masks[0])  # Search for background at time-step 0.
    segmentations = background_brightness * rgb_to_grayscale(images, num_output_channels=3)

    for slot_index in range(num_slots):
        # if slot_index == background_index:
        #     continue
        segmentations[:] += (1 - background_brightness) * masks[:, slot_index].repeat(1, 3, 1, 1) * colors[slot_index].unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(sequence_length, 1, width, height)
    return segmentations


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


def visualize_dynamics_prediction(images, predicted_images, predicted_rgbs, predicted_masks, num_context: int, padding: int = 2) -> None:
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

    pl_module.logger.experiment.add_image("Dynamics Prediction", grid, global_step=pl_module.current_epoch)
    if self.save_dir is not None:
        save_image(grid, self.save_dir + f"/dynamics_prediction-epoch={pl_module.current_epoch}.png")
