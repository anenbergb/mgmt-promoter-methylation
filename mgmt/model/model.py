import monai
from fvcore.common.config import CfgNode


def build_model(cfg: CfgNode):
    if cfg.MODEL.NAME.startswith("resnet"):
        model_args = cfg.MODEL.RESNET
        monai_resnets = monai.networks.nets.resnet.__dict__
        model = monai_resnets[cfg.MODEL.NAME](**model_args)

    return model
