import torch
from matplotlib import cm
import os
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from typing import Optional


# colors = [
#     'red',     # Class 1
#     'green',   # Class 2
#     'blue',    # Class 3
#     'yellow',  # Class 4
#     'purple',  # Class 5
#     'cyan',    # Class 6
#     'orange',  # Class 7
#     'magenta', # Class 8
#     'lime',    # Class 9
#     'brown',   # Class 10
#     'pink',    # Class 11
#     'gray'     # Class 12
# ]

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


def visualize_decompositions(videos, reconstructions, individual_reconstructions, masks, summary_writer: SummaryWriter,
                             epoch: int, savepath: Optional[str] = None, max_n_cols: int = 10) -> None:
    sequence_length, num_slots, _, _, _ = individual_reconstructions.size()
    n_cols = min(sequence_length, max_n_cols)

    videos = videos.cpu().detach()
    reconstructions = reconstructions.cpu().detach()
    error = (videos - reconstructions).pow(2).sum(dim=-3).sqrt()
    colorized_error = torch.from_numpy(cm.get_cmap('coolwarm')((0.5 * error.cpu().numpy()) + 0.5)[..., :3]).permute(0, 3, 1, 2)
    segmentations = create_segmentations(masks).cpu().detach()

    combined_reconstructions = masks[:n_cols] * individual_reconstructions[:n_cols]
    combined_reconstructions = torch.cat([combined_reconstructions[:, s] for s in range(num_slots)], dim=-2).detach().cpu()
    combined_reconstructions = torch.cat([videos, reconstructions, colorized_error, segmentations, combined_reconstructions], dim=-2)[:, :n_cols]
    combined_reconstructions = torch.cat([combined_reconstructions[t, :, :, :] for t in range(n_cols)], dim=-1)

    rgb_reconstructions = torch.cat([individual_reconstructions[:, s] for s in range(num_slots)], dim=-2)[:n_cols]
    rbg_reconstructions = torch.cat([rgb_reconstructions[t, :, :, :] for t in range(n_cols)], dim=-1)
    mask_reconstructions = torch.cat([masks[:, s] for s in range(num_slots)], dim=-2)[:n_cols]
    mask_reconstructions = torch.cat([mask_reconstructions[t, :, :, :] for t in range(n_cols)], dim=-1)

    summary_writer.add_image("Combined Reconstructions", combined_reconstructions, global_step=epoch)
    summary_writer.add_image("RGB", rbg_reconstructions, global_step=epoch)
    summary_writer.add_image("Masks", mask_reconstructions, global_step=epoch)

    if savepath is not None:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        save_image(combined_reconstructions, savepath + f"/combined-epoch={epoch}.png")
        save_image(rgb_reconstructions, savepath + f"/rgb-epoch={epoch}.png")
        save_image(mask_reconstructions, savepath + f"/masks-epoch={epoch}.png")


def get_background_slot_index(masks: torch.Tensor) -> torch.Tensor:
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


def create_segmentations(masks: torch.Tensor) -> torch.Tensor:
    sequence_length, num_slots, _, width, height = masks.size()
    background_index = get_background_slot_index(masks)
    segmentations = torch.zeros((sequence_length, 3, width, height), device=masks.device)
    for slot_index in range(num_slots):
        if slot_index == background_index:
            continue
        for c in range(3):
            segmentations[:, c] += masks[:, slot_index, 0] * colors[slot_index][c]
    return segmentations
