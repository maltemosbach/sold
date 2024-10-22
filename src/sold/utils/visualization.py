
import itertools
from math import ceil
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
import imageio
from torchvision.utils import draw_segmentation_masks
from webcolors import name_to_rgb
from tensorboardX import SummaryWriter
from typing import Optional

COLORS = ["white", "blue", "green", "olive", "red", "yellow", "purple", "orange", "cyan",
          "brown", "pink", "darkorange", "goldenrod", "darkviolet", "springgreen",
          "aqua", "royalblue", "navy", "forestgreen", "plum", "magenta", "slategray",
          "maroon", "gold", "peachpuff", "silver", "aquamarine", "indianred", "greenyellow",
          "darkcyan", "sandybrown"]


def visualize_reconstructions(videos, reconstructions, summary_writer: SummaryWriter, global_step: int,
                              savepath: Optional[str] = None, max_n_cols: int = 10) -> None:
    sequence_length, _, _, _ = videos.shape
    videos = videos.cpu().detach()
    reconstructions = reconstructions.cpu().detach()
    error = torch.abs(videos - reconstructions)
    #colorized_error = cm.get_cmap('coolwarm')(error)[..., :3]
    n_cols = min(sequence_length, max_n_cols)
    image_grid = torch.cat([videos, reconstructions, error], dim=-2)[:, :n_cols]
    summary_writer.add_images(tag=f"Reconstructions", img_tensor=np.array(image_grid), global_step=global_step)


def visualize_decompositions(videos, reconstructions, individual_reconstructions, masks, summary_writer: SummaryWriter,
                             global_step: int, savepath: Optional[str] = None, max_n_cols: int = 10) -> None:
    sequence_length, num_slots, _, _, _ = individual_reconstructions.size()
    n_cols = min(sequence_length, max_n_cols)

    videos = videos.cpu().detach()
    reconstructions = reconstructions.cpu().detach()
    error = (videos - reconstructions).pow(2).sum(dim=-3).sqrt()
    colorized_error = torch.from_numpy(cm.get_cmap('coolwarm')((0.5 * error.cpu().numpy()) + 0.5)[..., :3]).permute(0, 3, 1, 2)

    combined_reconstructions = masks[:n_cols] * individual_reconstructions[:n_cols]
    combined_reconstructions = torch.cat([combined_reconstructions[:, s] for s in range(num_slots)], dim=-2).detach().cpu()
    image_grid = torch.cat([videos, reconstructions, colorized_error, combined_reconstructions], dim=-2)[:, :n_cols]

    summary_writer.add_images(tag="Combined Reconstructions", img_tensor=image_grid, global_step=global_step)
    summary_writer.add_images(
        tag="RGB Reconstructions",
        img_tensor=torch.cat([individual_reconstructions[:, s] for s in range(num_slots)], dim=-2)[:n_cols],
        global_step=global_step)
    summary_writer.add_images(
        tag="Mask Reconstructions", img_tensor=torch.cat([masks[:, s] for s in range(num_slots)], dim=-2)[:n_cols],
        global_step=global_step)

def one_hot_to_idx(one_hot):
    """
    Converting a one-hot tensor into idx
    """
    idx_tensor = one_hot.argmax(dim=-3).unsqueeze(-3)
    return idx_tensor

def idx_to_one_hot(x):
    """
    Converting from instance indices into instance-wise one-hot encodings
    """
    num_classes = x.unique().max() + 1
    shape = x.shape
    x = x.flatten().to(torch.int64).view(-1,)
    y = torch.nn.functional.one_hot(x, num_classes=num_classes)
    y = y.view(*shape, num_classes)  # (..., Height, Width, Classes)
    y = y.transpose(-3, -1).transpose(-2, -1)  # (..., Classes, Height, Width)
    return y


def overlay_instances(instances, frames=None, colors=COLORS, alpha=1.):
    """
    Overlay instance segmentations on a sequence of images
    """
    if colors[0] != "white":  # background should always be white
        colors = ["white"] + colors

    # converting instance from one-hot to isntance indices, if necessary
    N, C, H, W = instances.shape
    if C > 1:
        instance = one_hot_to_idx(instances)

    # some preprocessing on images, or adding white canvas
    if frames is None:
        frames = torch.zeros(N, 3, H, W)
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, instance in zip(frames, instances):
        img = overlay_instance(instance, frame, colors, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


def overlay_instance(instance, img=None, colors=COLORS, alpha=0.7):
    """
    Overlaying the segmentation on an image
    """
    if colors[0] != "white":  # background should always be white
        colors = ["white"] + colors

    # converting instance from one-hot to isntance indices, if necessary
    C, H, W = instance.shape
    if C > 1:
        instance = one_hot_to_idx(instance)

    # some preprocessing on images, or adding white canvas
    if img is None:
        img = torch.zeros(3, H, W)
    if img.max() <= 1:
        img = img * 255
    img = img.to(torch.uint8)

    instance_ids = instance.unique()
    instance_masks = (instance[0] == instance_ids[:, None, None].to(instance.device))
    cur_colors = [colors[idx.item()] for idx in instance_ids]

    img_with_seg = draw_segmentation_masks(
            img,
            masks=instance_masks,
            alpha=alpha,
            colors=cur_colors
        )
    return img_with_seg / 255


def one_hot_instances_to_rgb(x, num_channels):
    """
    Converting from multi-channel one-hot instance masks to RGB images for visualization
    """
    x = x.float().round()
    masks_merged = x * torch.arange(num_channels, device=x.device).view(1, 1, -1, 1, 1, 1)
    masks_merged = masks_merged.sum(dim=2)
    masks_rgb = instances_to_rgb(masks_merged, num_channels=num_channels).squeeze(2)
    return masks_rgb


def one_hot_to_instances(x):
    """
    Converting from one-hot multi-channel instance representation to single-channel instance mask
    """
    masks_merged = torch.argmax(x, dim=2)
    return masks_merged


def instances_to_rgb(x, num_channels, colors=None):
    """ Converting from instance masks to RGB images for visualization """
    colors = COLORS if colors is None else colors
    img = torch.zeros(*x.shape, 3)
    background_val = x.flatten(-2).mode(dim=-1)[0]
    for cls in range(num_channels):
        color = colors[cls+1] if cls != background_val else "seashell"
        color_rgb = torch.tensor(name_to_rgb(color)).float()
        img[x == cls, :] = color_rgb / 255
    img = img.transpose(-3, -1).transpose(-2, -1)
    return img


def masks_to_rgb(x):
    """ Converting from SAVi masks to RGB images for visualization """

    # we make the assumption that the background is the mask with the most pixels (mode of distr.)
    num_objs = x.unique().max()
    background_val = x.flatten(-2).mode(dim=-1)[0]

    imgs = []
    for i in range(x.shape[0]):
        img = torch.zeros(*x.shape[1:], 3)
        for cls in range(num_objs + 1):
            color = COLORS[cls+1] if cls != background_val[i] else "seashell"
            color_rgb = torch.tensor(name_to_rgb(color)).float()
            img[x[i] == cls, :] = color_rgb / 255
        imgs.append(img)
    imgs = torch.stack(imgs)
    imgs = imgs.transpose(-3, -1).transpose(-2, -1)
    return imgs


def visualize_tight_row(frames, num_context=5, disp=[0, 1, 2, 3, 4, 9, 14, 19, 24],
                        is_gt=False, savepath=None):
    """
    Visualizing ground truth or predictions in a tight row

    Args:
    -----
    frames: torch tensor
        Frames to visualize
    num_context: int
        Number of seed frames used
    disp: list
        Indices of the prediction frames to visualize. Idx 0 means first prediction frame.
    is_gt: bool
        If True, the given frames correspond to ground truth. Otherwise correspond to predictions
    """
    num_frames_disp = num_context + len(disp)
    frames = frames.clamp(0, 1).cpu().detach()

    fig, ax = plt.subplots(1, num_frames_disp)
    fig.set_size_inches(2 * num_frames_disp, 2)
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(num_frames_disp):
        if is_gt and i < num_context:
            ax[i].imshow(frames[i].permute(1, 2, 0).detach().cpu())
            ax[i].set_xticks([], [])
            ax[i].set_yticks([], [])
            if i == 0:
                ax[i].set_title(f"t={i+1}", fontsize=10)
            else:
                ax[i].set_title(f"{i+1}", fontsize=10)
        elif is_gt and i >= num_context:
            frame_idx = num_context + disp[i - num_context]
            if frame_idx >= len(frames):
                break
            ax[i].imshow(frames[frame_idx].permute(1, 2, 0).detach().cpu())
            ax[i].set_xticks([], [])
            ax[i].set_yticks([], [])
            ax[i].set_title(f"{frame_idx+1}", fontsize=10)
        elif not is_gt and i < num_context:
            ax[i].imshow(torch.ones_like(frames[i]).permute(1, 2, 0).detach().cpu())
            ax[i].set_xticks([], [])
            ax[i].set_yticks([], [])
            ax[i].spines['bottom'].set_color('white')
            ax[i].spines['top'].set_color('white')
            ax[i].spines['right'].set_color('white')
            ax[i].spines['left'].set_color('white')
        elif not is_gt and i >= num_context:
            frame_idx = disp[i - num_context]
            if frame_idx >= len(frames):
                break
            ax[i].imshow(frames[frame_idx].permute(1, 2, 0).detach().cpu())
            ax[i].set_xticks([], [])
            ax[i].set_yticks([], [])
        else:
            raise ValueError("?")

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0.0)
    return fig, ax


def overlay_segmentations(frames, segmentations, colors, num_classes=None, alpha=0.7):
    """
    Overlaying the segmentation on a sequence of images
    """
    if num_classes is None:
        num_classes = segmentations.unique().max() + 1
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, segmentation in zip(frames, segmentations):
        img = overlay_segmentation(frame, segmentation, colors, num_classes, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


def overlay_segmentation(img, segmentation, colors, num_classes, alpha=0.7):
    """
    Overlaying the segmentation on an image
    """
    if img.max() <= 1:
        img = img * 255
    img = img.to(torch.uint8)

    # trying to always make the background of the 'seashell' color
    background_id = segmentation.sum(dim=(-1, -2)).argmax().item()
    cur_colors = colors[1:].copy()
    cur_colors.insert(background_id, "seashell")

    img_with_seg = draw_segmentation_masks(
            img,
            masks=segmentation.to(torch.bool),
            alpha=alpha,
            colors=cur_colors
        )
    return img_with_seg / 255


def hypot(a, b):
    """ """
    y = (a ** 2.0 + b ** 2.0) ** 0.5
    return y


def flow_to_rgb(flow, flow_scaling_factor=50):
    """
    Converting from optical flow to RGB
    """
    height, width = flow.shape[-3], flow.shape[-2]
    scaling = flow_scaling_factor / hypot(height, width)
    x, y = flow[..., 0], flow[..., 1]
    motion_angle = np.arctan2(y, x)
    motion_angle = (motion_angle / np.math.pi + 1.0) / 2.0
    motion_magnitude = hypot(y, x)
    motion_magnitude = np.clip(motion_magnitude * scaling, 0.0, 1.0)
    value_channel = np.ones_like(motion_angle)
    flow_hsv = np.stack([motion_angle, motion_magnitude, value_channel], axis=-1)
    flow_rgb = colors.hsv_to_rgb(flow_hsv)
    return flow_rgb

#
