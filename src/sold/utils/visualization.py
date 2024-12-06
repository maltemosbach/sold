from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar, _update_n
from lightning.pytorch.utilities.types import STEP_OUTPUT
import math
import matplotlib as mpl
from PIL import ImageDraw
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms.functional import rgb_to_grayscale
from typing import Any, Union, Optional


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

PADDING = 2
HEIGHT_SPACING = 2
WIDTH_SPACING = 4


class OnlineProgressBar(TQDMProgressBar):
    """Custom progress bar for online RL training loops."""

    def on_train_start(self, *_: Any) -> None:
        super().on_train_start()
        self.train_progress_bar.reset(total=self.total_train_batches)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_progress_bar.set_description(f"Steps {pl_module.num_steps}, Episodes {pl_module.num_episodes}")

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        n = pl_module.num_steps
        _update_n(self.train_progress_bar, n)

    @property
    def total_train_batches(self) -> Union[int, float]:
        return self.trainer.model.max_steps


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


def visualize_savi_decomposition(images, reconstructions, rgbs, masks, max_sequence_length: Optional[int] = 10) -> torch.Tensor:
    images = images.cpu()
    reconstructions = reconstructions.cpu()
    rgbs = rgbs.cpu()
    masks = masks.cpu()

    sequence_length, num_slots, _, _, _ = rgbs.size()
    n_cols = min(sequence_length, max_sequence_length) if max_sequence_length is not None else sequence_length

    true_row = make_grid(images[:n_cols], padding=PADDING, num_columns=n_cols)
    reconstruction_row = make_grid(reconstructions[:n_cols], padding=PADDING, num_columns=n_cols)

    segmentations = create_segmentation_overlay(images[:n_cols], masks[:n_cols], background_brightness=0.0).cpu().detach()
    segmentation_row = make_grid(segmentations, padding=PADDING, num_columns=n_cols)
    individual_slots = masks * rgbs

    # Slot rows.
    slot_rows = []
    for slot_index in range(num_slots):
        slots = make_grid(individual_slots[:n_cols, slot_index], padding=PADDING, num_columns=n_cols, pad_color=colors[slot_index])
        slot_rows.append(slots)

    # Combine rows.
    rows = [true_row, reconstruction_row, segmentation_row] + slot_rows
    grid = []
    for row_index, row in enumerate(rows):
        grid.append(row)
        if row_index < len(rows) - 1:
            grid.append(torch.ones(3, HEIGHT_SPACING, row.size(2)))

    grid = torch.cat(grid, dim=1)
    return grid


def visualize_dynamics_prediction(predicted_images, predicted_rgbs, predicted_masks, num_context: int, images=None) -> torch.Tensor:

    predicted_images = predicted_images.cpu()
    predicted_rgbs = predicted_rgbs.cpu()
    predicted_masks = predicted_masks.cpu()

    sequence_length, num_slots, _, _, _ = predicted_rgbs.size()
    predicted_individual_slots = predicted_masks * predicted_rgbs
    num_predictions = sequence_length - num_context

    # True vs Model rows.
    rows = []
    if images is not None:
        true_context = make_grid(images[:num_context].cpu(), padding=PADDING, num_columns=num_context)
        true_future = make_grid(images[num_context:].cpu(), padding=PADDING, num_columns=num_predictions)
        true_row = torch.cat([true_context, torch.ones(3, true_context.size(1), WIDTH_SPACING), true_future], dim=2)
        rows.append(true_row)

    model_context = make_grid(predicted_images[:num_context], padding=PADDING, num_columns=num_context)
    model_future = make_grid(predicted_images[num_context:], padding=PADDING, num_columns=num_predictions)
    model_row = torch.cat([model_context, torch.ones(3, model_context.size(1), WIDTH_SPACING), model_future], dim=2)
    rows.append(model_row)

    # Segmentation row.
    segmentation = create_segmentation_overlay(predicted_images, predicted_masks, background_brightness=0.0).cpu().detach()
    segmentation_context = make_grid(segmentation[:num_context], padding=PADDING, num_columns=num_context)
    segmentation_future = make_grid(segmentation[num_context:], padding=PADDING, num_columns=num_predictions)
    segmentation_row = torch.cat([segmentation_context, torch.ones(3, segmentation_context.size(1), WIDTH_SPACING), segmentation_future], dim=2)
    rows.append(segmentation_row)

    # Slot rows.
    for slot_index in range(num_slots):
        slot_context = make_grid(predicted_individual_slots[:num_context, slot_index], padding=PADDING, num_columns=num_context, pad_color=colors[slot_index])
        slot_future = make_grid(predicted_individual_slots[num_context:, slot_index], padding=PADDING, num_columns=num_predictions, pad_color=colors[slot_index])
        slot_row = torch.cat([slot_context, torch.ones(3, slot_context.size(1), WIDTH_SPACING), slot_future], dim=2)
        rows.append(slot_row)

    # Combine rows.
    grid = []
    for row_index, row in enumerate(rows):
        grid.append(row)
        if row_index < len(rows) - 1:
            grid.append(torch.ones(3, HEIGHT_SPACING, row.size(2)))
    grid = torch.cat(grid, dim=1)
    return grid


def visualize_reward_prediction(images, reconstructions, rewards, predicted_rewards) -> torch.Tensor:
    images = images.cpu()
    reconstructions = reconstructions.cpu()

    images = draw_reward(images, rewards.detach().cpu()) / 255
    reconstructions = draw_reward(reconstructions, predicted_rewards.detach().cpu()) / 255

    sequence_length, _, _, _ = images.size()

    true_row = make_grid(images, padding=PADDING, num_columns=sequence_length)
    model_row = make_grid(reconstructions, padding=PADDING, num_columns=sequence_length)

    # Combine rows.
    rows = [true_row, model_row]
    grid = []
    for row_index, row in enumerate(rows):
        grid.append(row)
        if row_index < len(rows) - 1:
            grid.append(torch.ones(3, HEIGHT_SPACING, row.size(2)))
    grid = torch.cat(grid, dim=1)
    return grid


def draw_reward(observation, reward, color = (255, 255, 255)):
    imgs = []
    for i, img in enumerate(observation):
        img = torchvision.transforms.functional.to_pil_image(img)
        draw = ImageDraw.Draw(img)
        draw.text((0.25 * img.width, 0.8 * img.height), f"{reward[i]:.3f}", color)
        imgs.append(torchvision.transforms.functional.pil_to_tensor(img))
    return torch.stack(imgs)


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class AttentionWeightsHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

    def compute_attention_weights(self, device, num_slots, seq_len):
        """Inspired by https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb"""

        if len(self.outputs) < 1:
            return 0
        else:
            att_mat = torch.stack(self.outputs)

            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mat.size(2)).to(device)
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
            joint_attentions[0] = aug_att_mat[0]

            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        # select attention of output token
        output_attention = joint_attentions[-1, 0, -1, :]
        # Remove the weights for the CLS and register tokens
        output_attention = torch.chunk(output_attention, seq_len, dim=-1)
        output_attention = torch.stack([frame_attention[:num_slots] for frame_attention in output_attention])

        return output_attention / output_attention.max()


@torch.no_grad()
def get_attention_weights(model: nn.Module, slots: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, num_slots, slot_dim = slots.size()
    attention_weights_hook = AttentionWeightsHook()

    hook_handles = []
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            patch_attention(module)
            hook_handles.append(module.register_forward_hook(attention_weights_hook))

    model(slots.detach(), start=slots.shape[1] - 1)
    output_weights = attention_weights_hook.compute_attention_weights(slots.device, num_slots, sequence_length)

    for hook_handle in hook_handles:
        hook_handle.remove()

    return output_weights


def visualize_output_attention(attention_weights, recons_history, masks_history):
    import matplotlib as mpl

    cmap = mpl.colormaps['plasma']

    reconstructed_imgs = torch.sum(recons_history * masks_history, dim=1)
    attention_imgs = torch.sum(
    attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * recons_history * masks_history, dim=1)
    attention_weight_imgs = torch.sum(
    attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * masks_history, dim=1)
    attention_weight_imgs -= attention_weight_imgs.min()
    attention_weight_imgs /= attention_weight_imgs.max()
    attention_weight_imgs = torch.from_numpy(cmap(attention_weight_imgs.cpu().numpy())).float()

    attention_weight_imgs = attention_weight_imgs[:, 0].permute(0, 3, 1, 2)[:, 0:3]

    # Calculate the number of images in each row
    num_images = reconstructed_imgs.shape[0]

    # Convert tensors to numpy arrays and transpose to correct shape
    #img = make_grid(torch.cat(observations), num_columns=num_images).permute(1, 2, 0).numpy()
    reconstructed_img = make_grid(reconstructed_imgs, num_columns=num_images).cpu()
    attention_img = make_grid(attention_imgs, num_columns=num_images).cpu()
    attention_weight_img = make_grid(attention_weight_imgs, num_columns=num_images).cpu()

    # Combine rows.
    rows = [reconstructed_img, attention_img, attention_weight_img]

    grid = []
    for row_index, row in enumerate(rows):
        grid.append(row)
        if row_index < len(rows) - 1:
            grid.append(torch.ones(3, HEIGHT_SPACING, row.size(2)))
    grid = torch.cat(grid, dim=1)
    return grid


@torch.no_grad()
def visualize_reward_predictor_attention(images, reconstructions, rewards, predicted_rewards,
                                         attention_weights, predicted_rgbs, predicted_masks) -> torch.Tensor:
    images = images.cpu()
    reconstructions = reconstructions.cpu()

    images[-1:] = draw_reward(images[-1:], rewards[-1:].detach().cpu(), color=(0, 255, 0)) / 255
    reconstructions[-1:] = draw_reward(reconstructions[-1:], predicted_rewards[-1:].detach().cpu(), color=(255, 0, 0)) / 255

    sequence_length, _, _, _ = images.size()

    true_row = make_grid(images, padding=PADDING, num_columns=sequence_length)
    model_row = make_grid(reconstructions, padding=PADDING, num_columns=sequence_length)

    attention_imgs = torch.sum(
        attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * predicted_rgbs * predicted_masks, dim=1)
    attention_weight_imgs = torch.sum(
        attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * predicted_masks, dim=1)
    attention_weight_imgs -= attention_weight_imgs.min()
    attention_weight_imgs /= attention_weight_imgs.max()
    attention_weight_imgs = torch.from_numpy(mpl.colormaps['plasma'](attention_weight_imgs.cpu().numpy())).float()
    attention_weight_imgs = attention_weight_imgs[:, 0].permute(0, 3, 1, 2)[:, 0:3]

    attention_brightness_row = make_grid(attention_imgs, num_columns=sequence_length, padding=PADDING).cpu()
    attention_colormap_row = make_grid(attention_weight_imgs, num_columns=sequence_length, padding=PADDING).cpu()

    # Combine rows.
    rows = [true_row, model_row, attention_brightness_row, attention_colormap_row]
    grid = []
    for row_index, row in enumerate(rows):
        grid.append(row)
        if row_index < len(rows) - 1:
            grid.append(torch.ones(3, HEIGHT_SPACING, row.size(2)))
    grid = torch.cat(grid, dim=1)
    return grid
