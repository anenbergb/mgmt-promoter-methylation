import monai
from fvcore.common.config import CfgNode


def get_n_input_channels(cfg: CfgNode) -> int:
    n_input_channels = 1
    if cfg.DATA.MODALITY == "concat":
        n_input_channels = len(cfg.DATA.MODALITY_CONCAT)
    return n_input_channels


def build_model(cfg: CfgNode):
    if cfg.MODEL.NAME.startswith("resnet"):
        model_args = cfg.MODEL.RESNET
        model_args["n_input_channels"] = get_n_input_channels(cfg)
        monai_resnets = monai.networks.nets.resnet.__dict__
        model = monai_resnets[cfg.MODEL.NAME](**model_args)

    return model
