import torch.nn as nn
from typing import Iterable


def get_parameters(modules: Iterable[nn.Module]):
    """For a list of nn.Modules, return a list of their parameters."""
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        """Freeze the parameters of a list of nn.Modules."""
        self.modules = modules
        self.initial_param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self) -> None:
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        for param_index, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.initial_param_states[param_index]
        return False
