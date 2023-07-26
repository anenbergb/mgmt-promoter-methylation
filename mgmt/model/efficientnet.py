from typing import Dict, List, NamedTuple, Optional, Tuple, Type, Union

from monai.networks.layers.factories import Act, Conv, Pool
from monai.networks.layers.utils import get_norm_layer
from monai.networks.nets.efficientnet import BlockArgs
from monai.networks.nets.efficientnet import EfficientNet as _EfficientNet
from monai.networks.nets.efficientnet import (
    MBConvBlock,
    _calculate_output_image_size,
    _make_same_padder,
    _round_filters,
    _round_repeats,
)
from torch import nn


class EfficientNet(_EfficientNet):
    def __init__(
        self,
        blocks_args_str: List[str],
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 1000,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        dropout_rate: float = 0.2,
        image_size: int = 224,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        drop_connect_rate: float = 0.2,
        depth_divisor: int = 8,
        stem_kernel_size: int = 3,
        stem_stride: int = 2,
        head_output_filters: int = 1280,
    ) -> None:
        """
        EfficientNet based on `Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/pdf/1905.11946.pdf>`_.
        Adapted from `EfficientNet-PyTorch <https://github.com/lukemelas/EfficientNet-PyTorch>`_.

        Args:
            blocks_args_str: block definitions.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            num_classes: number of output classes.
            width_coefficient: width multiplier coefficient (w in paper).
            depth_coefficient: depth multiplier coefficient (d in paper).
            dropout_rate: dropout rate for dropout layers.
            image_size: input image resolution.
            norm: feature normalization type and arguments.
            drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.
            depth_divisor: depth divisor for channel rounding.

        """
        super(_EfficientNet, self).__init__()

        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims can only be 1, 2 or 3.")

        # select the type of N-Dimensional layers to use
        # these are based on spatial dims and selected from MONAI factories
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv["conv", spatial_dims]
        adaptivepool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            "adaptiveavg", spatial_dims
        ]

        # decode blocks args into arguments for MBConvBlock
        blocks_args = [BlockArgs.from_string(s) for s in blocks_args_str]

        # checks for successful decoding of blocks_args_str
        if not isinstance(blocks_args, list):
            raise ValueError("blocks_args must be a list")

        if blocks_args == []:
            raise ValueError("block_args must be non-empty")

        self._blocks_args = blocks_args
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.drop_connect_rate = drop_connect_rate

        # build MBConv blocks
        num_blocks = 0
        self._blocks = nn.Sequential()

        self.extract_stacks = []

        # update baseline blocks to input/output filters and number of repeats based on width and depth multipliers.
        for idx, block_args in enumerate(self._blocks_args):
            block_args = block_args._replace(
                input_filters=_round_filters(block_args.input_filters, width_coefficient, depth_divisor),
                output_filters=_round_filters(block_args.output_filters, width_coefficient, depth_divisor),
                num_repeat=_round_repeats(block_args.num_repeat, depth_coefficient),
            )
            self._blocks_args[idx] = block_args

            # calculate the total number of blocks - needed for drop_connect estimation
            num_blocks += block_args.num_repeat

            if block_args.stride > 1:
                self.extract_stacks.append(idx)

        self.extract_stacks.append(len(self._blocks_args))

        # expand input image dimensions to list
        current_image_size = [image_size] * spatial_dims

        first_block_input_filters = self._blocks_args[0].input_filters
        # Stem
        out_channels = _round_filters(
            first_block_input_filters, width_coefficient, depth_divisor
        )  # number of output channels
        self._conv_stem = conv_type(
            self.in_channels, out_channels, kernel_size=stem_kernel_size, stride=stem_stride, bias=False
        )
        self._conv_stem_padding = _make_same_padder(self._conv_stem, current_image_size)
        self._bn0 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)
        current_image_size = _calculate_output_image_size(current_image_size, stem_stride)

        # create and add MBConvBlocks to self._blocks
        idx = 0  # block index counter
        for stack_idx, block_args in enumerate(self._blocks_args):
            blk_drop_connect_rate = self.drop_connect_rate

            # scale drop connect_rate
            if blk_drop_connect_rate:
                blk_drop_connect_rate *= float(idx) / num_blocks

            sub_stack = nn.Sequential()
            # the first block needs to take care of stride and filter size increase.
            sub_stack.add_module(
                str(idx),
                MBConvBlock(
                    spatial_dims=spatial_dims,
                    in_channels=block_args.input_filters,
                    out_channels=block_args.output_filters,
                    kernel_size=block_args.kernel_size,
                    stride=block_args.stride,
                    image_size=current_image_size,
                    expand_ratio=block_args.expand_ratio,
                    se_ratio=block_args.se_ratio,
                    id_skip=block_args.id_skip,
                    norm=norm,
                    drop_connect_rate=blk_drop_connect_rate,
                ),
            )
            idx += 1  # increment blocks index counter

            current_image_size = _calculate_output_image_size(current_image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

            # add remaining block repeated num_repeat times
            for _ in range(block_args.num_repeat - 1):
                blk_drop_connect_rate = self.drop_connect_rate

                # scale drop connect_rate
                if blk_drop_connect_rate:
                    blk_drop_connect_rate *= float(idx) / num_blocks

                # add blocks
                sub_stack.add_module(
                    str(idx),
                    MBConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_args.input_filters,
                        out_channels=block_args.output_filters,
                        kernel_size=block_args.kernel_size,
                        stride=block_args.stride,
                        image_size=current_image_size,
                        expand_ratio=block_args.expand_ratio,
                        se_ratio=block_args.se_ratio,
                        id_skip=block_args.id_skip,
                        norm=norm,
                        drop_connect_rate=blk_drop_connect_rate,
                    ),
                )
                idx += 1  # increment blocks index counter

            self._blocks.add_module(str(stack_idx), sub_stack)

        # sanity check to see if len(self._blocks) equal expected num_blocks
        if idx != num_blocks:
            raise ValueError("total number of blocks created != num_blocks")

        # Head
        head_in_channels = block_args.output_filters
        out_channels = _round_filters(head_output_filters, width_coefficient, depth_divisor)
        self._conv_head = conv_type(head_in_channels, out_channels, kernel_size=1, bias=False)
        self._conv_head_padding = _make_same_padder(self._conv_head, current_image_size)
        self._bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)

        # final linear layer
        self._avg_pooling = adaptivepool_type(1)
        self._dropout = nn.Dropout(dropout_rate)
        self._fc = nn.Linear(out_channels, self.num_classes)

        # swish activation to use - using memory efficient swish by default
        # can be switched to normal swish using self.set_swish() function call
        self._swish = Act["memswish"]()

        # initialize weights using Tensorflow's init method from official impl.
        self._initialize_weights()
