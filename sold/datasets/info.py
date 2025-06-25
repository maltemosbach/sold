from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class FieldType(ABC):

    @staticmethod
    @abstractmethod
    def is_compatible(data: Any) -> bool:
        """Returns whether the given data can be represented by this field type."""
        pass


@dataclass
class TensorType(FieldType):
    device: Union[torch.device, str]
    dtype: Union[torch.dtype, str]
    shape: Union[torch.Size, Sequence[int]]

    def __post_init__(self) -> None:
        self.device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        self.dtype = self.dtype if isinstance(self.dtype, torch.dtype) else getattr(torch, self.dtype)
        self.shape = self.shape if isinstance(self.shape, torch.Size) else torch.Size(self.shape)

    def element_size(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()

    def step_size(self) -> int:
        return self.element_size() * self.shape.numel()

    @staticmethod
    def is_compatible(data: Any) -> bool:
        return isinstance(data, torch.Tensor)


@dataclass
class Scalar(TensorType):
    shape: Union[torch.Size, Sequence[int]] = field(repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        if len(self.shape) != 0:
            raise ValueError(f"Expected Scalar to be zero-dimensional, but found shape {self.shape}.")

    @staticmethod
    def is_compatible(data: Any) -> bool:
        return TensorType.is_compatible(data) and data.ndim == 0


@dataclass
class Vector(TensorType):
    def __post_init__(self) -> None:
        super().__post_init__()

        if len(self.shape) != 1:
            raise ValueError(f"Expected Vector to be one-dimensional, but found shape {self.shape}.")

    @staticmethod
    def is_compatible(data: Any) -> bool:
        return TensorType.is_compatible(data) and data.ndim == 1


@dataclass
class ColorImage(TensorType):
    """Describes color image tensors of shape [3, height, width]."""
    height: int
    width: int
    dtype: Union[torch.dtype, str] = field(init=False, repr=False)
    shape: Union[torch.Size, Sequence[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dtype = torch.uint8
        self.shape = torch.Size((self.height, self.width, 3))

    @staticmethod
    def is_compatible(data: Any) -> bool:
        return TensorType.is_compatible(data) and data.ndim == 3 and data.dtype == torch.uint8 and data.shape[-1] == 3


class Fields(dict):
    """Describes the structure of the step-data added to a buffer.

    Special dictionary of type `dict[str, FieldType]` that describes the structure of step-data, where keys are
    field-names (e.g. 'observation', 'action', 'reward', etc.), and values are the type of that field, such as a
    tensor of a certain shape, dtype and device.
    """

    @classmethod
    def from_episode(cls, episode: Dict[str, List]) -> 'Fields':
        dic = {}
        for key, value in episode.items():

            if not isinstance(value, list) or len(value) == 0:
                raise ValueError(f"Expected value at key '{key}' to be a list, but found {value}.")

            # Check that value is a PyTorch tensor.
            if TensorType.is_compatible(value[0]):
                if Scalar.is_compatible(value[0]):
                    dic[key] = Scalar(device=value[0].device, dtype=value[0].dtype, shape=value[0].shape)
                elif Vector.is_compatible(value[0]):
                    dic[key] = Vector(device=value[0].device, dtype=value[0].dtype, shape=value[0].shape)
                elif ColorImage.is_compatible(value[0]):
                    dic[key] = ColorImage(height=value[0].shape[0], width=value[0].shape[1], device=value[0].device)
                else:
                    dic[key] = TensorType(device=value[0].device, dtype=value[0].dtype, shape=value[0].shape)
            else:
                raise ValueError(f"Unsupported type {type(value[0])} for field {key}.")

        return cls(**dic)


@dataclass
class EpisodeDatasetInfo:
    fields: Optional[Fields] = None
    num_episodes: int = 0
    num_timesteps: int = 0
    episode_lengths: List[int] = field(default_factory=list)
    is_full: bool = False

    @property
    def is_empty(self) -> bool:
        return self.num_episodes == 0


class EpisodeDatasetInfoMixin:
    def __init__(self, info: EpisodeDatasetInfo) -> None:
        self.info = info
        self.last_episode_return = None

    def _update_stats_on_store(self, episode: Dict[str, List]) -> None:
        self.info.num_episodes += 1
        self.info.num_timesteps += len(next(iter(episode.values())))
        self.info.episode_lengths.append(len(next(iter(episode.values()))))
        self.last_episode_return = sum(episode['reward'][1:])

    def _update_stats_on_remove(self) -> None:
        self.info.num_episodes -= 1
        self.info.num_timesteps -= self.info.episode_lengths.pop(0)
        self.info.is_full = True  # is_full is set to True once episodes start being removed.

    @property
    def num_episodes(self) -> int:
        return self.info.num_episodes

    @property
    def num_timesteps(self) -> int:
        return self.info.num_timesteps

    @property
    def episode_lengths(self) -> List[int]:
        return self.info.episode_lengths

    @property
    def is_empty(self) -> bool:
        return self.info.is_empty

    @property
    def is_full(self) -> bool:
        return self.info.is_full

    @property
    def average_episode_length(self) -> float:
        return sum(self.episode_lengths) / self.num_episodes if self.num_episodes > 0 else 0

    def __len__(self) -> int:
        return self.num_timesteps

    @property
    def fields(self) -> Fields:
        return self.info.fields


