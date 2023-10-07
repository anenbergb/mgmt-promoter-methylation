"""
Following the example of https://pytorch.org/tutorials/intermediate/parametrizations.html
"""
import torch.nn.utils.parametrize as parametrize
from fvcore.common.config import CfgNode
from torch import nn
from torch.nn.utils.parametrizations import orthogonal, spectral_norm


def non_batch_dims(ndim):
    return tuple(list(range(ndim))[1:])


class Standardization(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, weight):
        dim = non_batch_dims(weight.ndim)
        mean = weight.mean(dim=dim, keepdim=True)
        std = weight.std(dim=dim, keepdim=True) + self.eps
        weight = (weight - mean) / std
        return weight


def make_conv_parametrization_function(cfg: CfgNode):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html

    remove parametrizations from the module with
    parametrize.remove_parametrizations(layer, "weight")
    https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.remove_parametrizations.html

    """

    def to_return(module: nn.Module):
        if isinstance(module, nn.modules.conv._ConvNd):
            if cfg.PARAMETRIZATION == "Standardization":
                standardization = Standardization(**cfg.PARAMETRIZATION.Standardization)
                parametrize.register_parametrization(module, "weight", standardization)
            elif cfg.PARAMETRIZATION == "SpectralNorm":
                spectral_norm(module, "weight", **cfg.PARAMETRIZATION.SpectralNorm)
            elif cfg.PARAMETRIZATION == "Orthogonal":
                orthogonal(module, "weight", **cfg.PARAMETRIZATION.Orthogonal)
            # elif cfg.PARAMETRIZATION == "WeightNorm":
            #     weight_norm(module, "weight", **cfg.PARAMETRIZATION.WeightNorm)
        return module

    return to_return
