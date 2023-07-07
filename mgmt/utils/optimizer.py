from typing import Any

import torch
from fvcore.common.config import CfgNode


def build_optimizer(params: Any, cfg: CfgNode):
    """ """
    sfg = cfg.SOLVER

    if sfg.OPTIMIZER_NAME == "Adam":
        optimizer = torch.optim.Adam(params, lr=sfg.BASE_LR, weight_decay=sfg.WEIGHT_DECAY, **sfg.ADAM)
    elif sfg.OPTIMIZER_NAME == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=sfg.BASE_LR, weight_decay=sfg.WEIGHT_DECAY, **sfg.ADAM)
    elif sfg.OPTIMIZER_NAME == "NAdam":
        optimizer = torch.optim.NAdam(
            params,
            lr=sfg.BASE_LR,
            weight_decay=sfg.WEIGHT_DECAY,
            **sfg.ADAM,
            **sfg.NADAM,
        )
    elif sfg.OPTIMIZER_NAME == "SGD":
        optimizer = torch.optim.SGD(params, lr=sfg.BASE_LR, weight_decay=sfg.WEIGHT_DECAY, **sfg.SGD)
    else:
        raise ValueError(f"Unknown optimizer {sfg.OPTIMIZER_NAME}")
    return optimizer
