import torch
from torch import nn


def weights_init(m):
    conv_type = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    norm_type = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.LocalResponseNorm,
        nn.SyncBatchNorm,
    )

    if isinstance(m, conv_type):
        nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, norm_type):
        nn.init.constant_(torch.as_tensor(m.weight), 1)
        nn.init.constant_(torch.as_tensor(m.bias), 0)
    elif isinstance(m, nn.Linear):
        nn.init.constant_(torch.as_tensor(m.bias), 0)
