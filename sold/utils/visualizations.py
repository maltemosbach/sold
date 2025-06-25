import colorsys
import math
import matplotlib as mpl
from PIL import ImageDraw
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms.functional import rgb_to_grayscale
from typing import List, Optional, Union


import colorsys
import math
from PIL import ImageDraw
import torch
import torchvision
from torchvision.transforms.functional import rgb_to_grayscale
from typing import List, Optional, Union


def slot_color(slot_index: int, num_slots: int, saturation: float = 1.0, lightness: float = 0.5) -> torch.Tensor:
    num_slots = max(num_slots, 2)
    hue = (slot_index / (num_slots - 1)) * (1 - (1 / num_slots))
    return torch.Tensor(colorsys.hls_to_rgb(hue, lightness, saturation))


def make_grid(images: torch.Tensor, num_columns: int, padding: int = 2, pad_color: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not torch.is_tensor(images):
        raise TypeError(f"Expected images as type tensor, got {type(images)}")

    if pad_color is None:
        pad_color = torch.Tensor([0., 0., 0.])
    if pad_color.numel() != 3:
        raise ValueError("pad_color must have exactly 3 elements (RGB)")

    if images.ndim != 4:
        raise ValueError("Expected images of shape (num_images, num_channels, height, width)")
    num_images, num_channels, height, width = images.size()
    if num_channels != 3:
        raise ValueError("Expected images with 3 channels (RGB)")

    # Create background grid with pad_color.
    num_columns = min(num_columns, num_images)
    num_rows = int(math.ceil(float(num_images) / num_columns))
    height, width = int(height + padding), int(width + padding)
    grid = pad_color.unsqueeze(1).unsqueeze(2).repeat(1, height * num_rows + padding, width * num_columns + padding).to(images.device, images.dtype)

    # Copy each image into its corresponding position in the grid.
    image_index = 0
    for row in range(num_rows):
        for col in range(num_columns):
            if image_index >= num_images:
                break
            grid.narrow(1, row * height + padding, height - padding).narrow(2, col * width + padding, width - padding).copy_(images[image_index])
            image_index += 1
    return grid


def make_row(images: torch.Tensor, max_sequence_length: Optional[int] = None, padding: int = 2, pad_color: Optional[torch.Tensor] = None) -> torch.Tensor:
    sequence_length, _, _, _ = images.size()
    n_cols = min(sequence_length, max_sequence_length) if max_sequence_length is not None else sequence_length
    return make_grid(images[:n_cols], num_columns=n_cols, padding=padding, pad_color=pad_color)


def stack_rows(rows: List[torch.Tensor], height_spacing: int = 2) -> torch.Tensor:
    grid = []
    for row_index, row in enumerate(rows):
        grid.append(row.cpu())
        if row_index < len(rows) - 1:
            grid.append(torch.ones(3, height_spacing, row.size(2)))
    return torch.cat(grid, dim=1)


def stack_columns(images: torch.Tensor, num_context: int, pad_color: Optional[torch.Tensor] = None,
                  width_spacing: int = 2) -> torch.Tensor:
    sequence_length = images.size(0)
    num_predictions = sequence_length - num_context
    context = make_grid(images[:num_context].cpu(), num_columns=num_context, pad_color=pad_color)
    prediction = make_grid(images[num_context:].cpu(), num_columns=num_predictions, pad_color=pad_color)
    return torch.cat([context, torch.ones(3, context.size(1), width_spacing), prediction], dim=2)


def create_segmentation_overlay(images: torch.Tensor, masks: torch.Tensor, background_brightness: float = 0.4) -> torch.Tensor:
    sequence_length, num_slots, _, width, height = masks.size()
    segmentations = background_brightness * rgb_to_grayscale(images, num_output_channels=3)
    for slot_index in range(num_slots):
        segmentations[:] += (1 - background_brightness) * masks[:, slot_index].repeat(1, 3, 1, 1) * slot_color(slot_index, num_slots).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(sequence_length, 1, width, height)
    return segmentations


def draw_ticks(backgrounds: torch.Tensor, ticks, color):
    labeled_images = []
    for index, image in enumerate(backgrounds):
        img = torchvision.transforms.functional.to_pil_image(image)
        draw = ImageDraw.Draw(img)
        draw.text((0.5 * img.width, img.height), ticks[index], color, anchor='md')
        labeled_images.append(torchvision.transforms.functional.pil_to_tensor(img) / 255.)


    return torch.stack(labeled_images)


def visualize_reward_prediction(images, reconstructions, rewards, predicted_rewards) -> torch.Tensor:
    images, reconstructions = images.cpu(), reconstructions.cpu()
    images = draw_reward(images, rewards.detach().cpu()) / 255
    reconstructions = draw_reward(reconstructions, predicted_rewards.detach().cpu()) / 255

    sequence_length, _, _, _ = images.size()
    true_row = make_grid(images, num_columns=sequence_length)
    model_row = make_grid(reconstructions, num_columns=sequence_length)
    return stack_rows([true_row, model_row])


def draw_reward(observation, reward, color = (255, 255, 255)):
    imgs = []
    for i, img in enumerate(observation):
        img = torchvision.transforms.functional.to_pil_image(img)
        draw = ImageDraw.Draw(img)
        draw.text((0.25 * img.width, 0.8 * img.height), f"{reward[i]:.3f}", color)
        imgs.append(torchvision.transforms.functional.pil_to_tensor(img))
    return torch.stack(imgs)


def enable_attention_weights(module: nn.Module) -> None:
    forward_orig = module.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True

        return forward_orig(*args, **kwargs)

    module.forward = wrap


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
            enable_attention_weights(module)
            hook_handles.append(module.register_forward_hook(attention_weights_hook))

    model(slots.detach(), start=slots.shape[1] - 1)
    output_weights = attention_weights_hook.compute_attention_weights(slots.device, num_slots, sequence_length)

    for hook_handle in hook_handles:
        hook_handle.remove()

    return output_weights


def get_output_attention_images(attention_weights, rgbs, masks):
    import matplotlib as mpl
    cmap = mpl.colormaps['plasma']

    attention_brightness_images = torch.sum(
        attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * rgbs * masks, dim=1)

    attention_colormap_images = torch.sum(
        attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * masks, dim=1)
    attention_colormap_images -= attention_colormap_images.min()
    attention_colormap_images /= attention_colormap_images.max()
    attention_colormap_images = torch.from_numpy(cmap(attention_colormap_images.cpu().numpy())).float()
    attention_colormap_images = attention_colormap_images[:, 0].permute(0, 3, 1, 2)[:, 0:3]
    return attention_brightness_images, attention_colormap_images


def visualize_output_attention(attention_weights, reconstructions, rgbs, masks):
    attention_brightness_images, attention_colormap_images = get_output_attention_images(
        attention_weights, rgbs, masks)
    sequence_length, _, _, _ = reconstructions.size()
    reconstructions_row = make_grid(reconstructions, num_columns=sequence_length).cpu()
    attention_brightness_row = make_grid(attention_brightness_images, num_columns=sequence_length).cpu()
    attention_colormap_row = make_grid(attention_colormap_images, num_columns=sequence_length).cpu()
    return stack_rows([reconstructions_row, attention_brightness_row, attention_colormap_row])


@torch.no_grad()
def visualize_reward_predictor_attention(images, reconstructions, rewards, predicted_rewards,
                                         attention_weights, predicted_rgbs, predicted_masks) -> torch.Tensor:
    images[-1:] = draw_reward(images[-1:], rewards[-1:].detach().cpu(), color=(0, 255, 0)) / 255
    reconstructions[-1:] = draw_reward(reconstructions[-1:], predicted_rewards[-1:].detach().cpu(), color=(255, 0, 0)) / 255
    sequence_length, _, _, _ = images.size()

    true_row = make_grid(images, num_columns=sequence_length)
    model_row = make_grid(reconstructions, num_columns=sequence_length)

    attention_brightness_images, attention_colormap_images = get_output_attention_images(
        attention_weights, predicted_rgbs, predicted_masks)
    attention_brightness_row = make_grid(attention_brightness_images, num_columns=sequence_length).cpu()
    attention_colormap_row = make_grid(attention_colormap_images, num_columns=sequence_length).cpu()
    return stack_rows([true_row, model_row, attention_brightness_row, attention_colormap_row])
