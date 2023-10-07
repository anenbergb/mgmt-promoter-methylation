from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
from monai.networks.blocks import Convolution
from monai.networks.layers.utils import get_dropout_layer
from torch import nn

from mgmt.model.utils import weights_init


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        num_convs: int = 3,
        in_channels: int = 16,
        out_channels: int = 16,
        stride: int = 2,
        act: str = "relu",
        norm: Union[Tuple, str] = ("group", {"eps": 1e-5, "num_groups": 8}),
        dropout: Optional[float] = None,
        dropout_dim: int = 1,
        groups: int = 1,  # dr. chang might actually use 8 here
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        conv_kwargs = {
            "spatial_dims": 3,
            "out_channels": out_channels,
            "kernel_size": 3,
            # norm, dropout, activation
            "adn_ordering": "NDA",
            "act": act,
            "norm": norm,
            "dropout": None,
            "dropout_dim": dropout_dim,
            "groups": groups,
            "bias": False,
        }
        self.append(
            Convolution(
                **conv_kwargs,
                in_channels=in_channels,
                strides=stride,
            )
        )
        for _ in range(num_convs - 1):
            self.append(Convolution(**conv_kwargs, in_channels=out_channels, strides=1))
        if dropout is not None:
            self.append(get_dropout_layer(name=dropout, dropout_dim=dropout_dim))

        self.apply(weights_init)


class BasicBackbone(nn.Module):
    def __init__(
        self,
        block_num_convs: List[int] = [4, 4, 3, 2, 2],
        block_out_channels: List[int] = [16, 40, 64, 88, 112],
        n_input_channels: int = 1,
        act: str = "relu",
        norm: Union[Tuple, str] = ("group", {"eps": 1e-5, "num_groups": 8}),
        dropout: Optional[float] = None,
        dropout_dim: int = 1,
        groups: int = 1,  # baseline sets this to 8
    ):
        super().__init__()
        assert len(block_num_convs) == len(block_out_channels)
        block_kwargs = {
            "act": act,
            "norm": norm,
            "dropout": dropout,
            "dropout_dim": dropout_dim,
            "groups": groups,
        }
        self.block_num_convs = block_num_convs
        self.block_out_channels = block_out_channels
        self.blocks = nn.ModuleDict()
        self.block_strides = []
        for i, (num_convs, out_channels) in enumerate(zip(block_num_convs, block_out_channels)):
            in_channels = n_input_channels if i == 0 else block_out_channels[i - 1]
            stride = 1 if i == 0 else 2
            self.blocks[f"l{i+1}"] = BasicBlock(
                **block_kwargs,
                num_convs=num_convs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            )
            self.block_strides.append(stride)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        out = OrderedDict()
        for name, block in self.blocks.items():
            x = block(x)
            out[name] = x
        return out
