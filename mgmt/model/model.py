import monai
from fvcore.common.config import CfgNode

from mgmt.model.efficientnet import EfficientNet


def get_n_input_channels(cfg: CfgNode) -> int:
    n_input_channels = 1
    if cfg.DATA.MODALITY == "concat":
        n_input_channels = len(cfg.DATA.MODALITY_CONCAT)
    return n_input_channels


def build_model(cfg: CfgNode):
    input_channels = get_n_input_channels(cfg)
    if cfg.MODEL.NAME.startswith("resnet"):
        model_args = cfg.MODEL.RESNET
        model_args["n_input_channels"] = input_channels
        monai_resnets = monai.networks.nets.resnet.__dict__
        model = monai_resnets[cfg.MODEL.NAME](**model_args)
    elif cfg.MODEL.NAME == "ResNet":
        model_args = cfg.MODEL.ResNet
        model_args["n_input_channels"] = input_channels
        model = monai.networks.nets.resnet.ResNet(**model_args)
    elif cfg.MODEL.NAME == "EfficientNet":
        model_args = cfg.MODEL.EfficientNet
        model_args["in_channels"] = input_channels
        model = EfficientNet(**model_args)
    return model
