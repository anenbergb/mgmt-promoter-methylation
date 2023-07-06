import math
from typing import Any

import torch
from fvcore.common.config import CfgNode


class WarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup_steps: int, warmup_strategy: str = "linear", last_epoch: int = -1
    ):
        """
        warmup_steps is in epochs
        """
        self.warmup_steps = warmup_steps
        self.warmup_strategy = warmup_strategy
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            if self.warmup_strategy == "linear":
                return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
            elif self.warmup_strategy == "cosine":
                return [
                    base_lr * (1 + math.cos(math.pi * self.last_epoch / self.warmup_steps)) / 2
                    for base_lr in self.base_lrs
                ]
        else:
            return self.base_lrs


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: CfgNode,
    steps_per_epoch: int = 50,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler:
    """
    OneCyleLR
        - https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    """
    sfg = cfg.SOLVER

    if sfg.SCHEDULER_NAME == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            epochs=cfg.TRAINER.max_epochs,
            steps_per_epoch=steps_per_epoch,
            last_epoch=last_epoch,
            **sfg.OneCycleLR,
        )
    elif sfg.SCHEDULER_NAME == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sfg.ReduceLROnPlateau)
    elif sfg.SCHEDULER_NAME == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, last_epoch=last_epoch, **sfg.MultiStepLR)
    else:
        raise ValueError(f"Unknown lr_scheduler {sfg.SCHEDULER_NAME}")

    if sfg.WARMUP.ENABLED:
        warmup = WarmupScheduler(optimizer, sfg.WARMUP.warmup_steps, sfg.WARMUP.warmup_strategy)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[warmup, scheduler])

    return scheduler
