import argparse
import copy
import os
import sys

import matplotlib
import numpy as np
from fvcore.common.config import CfgNode
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from loguru import logger

matplotlib.use("Agg")

from mgmt.config import get_cfg
from mgmt.data.dataloader import DataModule
from mgmt.model.model_module import Classifier
from mgmt.utils.logger import setup_logger
from mgmt.utils.progress_bar import ProgressBar


def main(cfg):
    """
    Trainer
    - https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
    Tensorboard Logger
    - https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html#tensorboardlogger

    Weights and Biases Logging
    - https://docs.wandb.ai/guides/integrations/lightning
    """
    setup_logger(cfg)
    seed_everything(cfg.SEED_EVERYTHING, workers=True)

    datamodule = DataModule(cfg)
    steps_per_epoch = get_steps_per_epoch(cfg, datamodule)
    # TODO: make this work when you're resuming from a checkpoint
    max_steps = steps_per_epoch * cfg.TRAINER.max_epochs

    tb_logger = TensorBoardLogger(save_dir=cfg.OUTPUT_DIR, version="logs", name="")
    callbacks = get_callbacks(cfg, max_steps)
    trainer_kwargs = get_trainer_kwargs(cfg)
    trainer = Trainer(**trainer_kwargs, callbacks=callbacks, logger=tb_logger)

    model = Classifier(cfg, steps_per_epoch=steps_per_epoch)
    # Distributed is initialized in fit, not init
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.CHECKPOINT.PATH)


def get_trainer_kwargs(cfg: CfgNode):
    kwargs = copy.copy(cfg.TRAINER)
    if kwargs["profiler"] == "simple":
        kwargs["profiler"] = SimpleProfiler(**cfg.PROFILER.SIMPLE)
    elif kwargs["profiler"] == "advanced":
        kwargs["profiler"] = AdvancedProfiler(**cfg.PROFILER.ADVANCED)
    return kwargs


def get_steps_per_epoch(cfg: CfgNode, datamodule: DataModule) -> int:
    datamodule.prepare_data()
    num_subjects = len(datamodule.subjects)
    num_train = np.ceil(cfg.DATA.TRAIN_VAL_RATIO * num_subjects)
    steps_per_epoch = int(np.ceil(num_train / cfg.DATA.BATCH_SIZE))
    return steps_per_epoch


def get_callbacks(cfg: CfgNode, max_steps: int) -> list[Callback]:
    """

    LearningRateMonitor
        - https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html
    """
    # Maybe add EarlyStopping, lightning.pytorch.callbacks.EarlyStopping

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # maybe monitor the val accuracy rather than val loss
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(cfg.OUTPUT_DIR, "checkpoints"),
        filename="epoch={epoch}-step={step}-val_accuracy={val/accuracy:.2f}",
        monitor="val/accuracy",
        save_top_k=cfg.CHECKPOINT.save_top_k,
        mode="max",
        auto_insert_metric_name=False,
        save_last=True,
    )
    progress_bar = ProgressBar(max_steps=max_steps)
    # progress_bar = RichProgressBar()
    # maybe add lightning.pytorch.callbacks.EarlyStopping
    # TODO: add scheduler https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html

    return [lr_monitor, progress_bar, checkpoint]


def setup_config(args: argparse.Namespace):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())

    return cfg


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Glioblastoma MGMT Promoter Methylation classification trainer.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        "-c",
        default=[],
        metavar="FILE",
        action="append",
        help="Path to config file. Can provide multiple config files, "
        "which are combined together, overwriting the previous.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg = setup_config(args)
    sys.exit(main(cfg))
