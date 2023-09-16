from typing import Tuple, Type, Union

import torch
from monai.networks.layers.factories import Pool
from monai.networks.layers.utils import get_pool_layer
from torch import nn

from mgmt.model.utils import weights_init


class MultiResolutionWithMask(nn.Module):
    """
    Args:
        pool: adaptiveaverage or adaptivemask
    """

    def __init__(
        self,
        backbone: nn.Module,
        pool: str = "adaptiveavg",
        num_classes: int = 1,
    ):
        super().__init__()
        assert isinstance(backbone.blocks, nn.ModuleDict)
        assert pool in ("adaptiveavg", "adaptivemax")

        pool_type: Type[Union[nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool3d]] = Pool[pool, 3]

        self.backbone = backbone
        self.heads = nn.ModuleDict()
        self.mask_downsample = nn.ModuleDict()

        block = None
        for name, block in backbone.blocks.items():
            if block.stride > 1:
                # typically downsample by 2x
                self.mask_downsample[name] = nn.MaxPool3d(block.stride)
            self.heads[name] = nn.Sequential(
                pool_type(output_size=(1, 1, 1)), nn.Conv3d(block.out_channels, num_classes, 1, bias=True), nn.Flatten()
            )
            block = block

        self.heads["final"] = nn.Sequential(
            pool_type(output_size=(1, 1, 1)), nn.Conv3d(block.out_channels, num_classes, 1, bias=True), nn.Flatten()
        )

        self.apply(weights_init)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_out = self.backbone(x)
        out = {}
        block = None
        for name, block in backbone_out.items():
            if name in self.mask_downsample:
                mask = self.mask_downsample[name](mask)
            block_mask = torch.mul(block, mask)
            out[name] = self.heads[name](block_mask)

        out["final"] = self.heads["final"](block)
        return out
