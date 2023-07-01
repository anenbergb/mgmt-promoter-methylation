import argparse
import os
import sys
import logging
from loguru import logger

import lightning.pytorch as pl
from fvcore.common.config import CfgNode
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from mgmt.config import get_cfg
from mgmt.data.dataloader import DataModule
from mgmt.model.model_module import Classifier
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


    tb_logger = TensorBoardLogger(save_dir=cfg.OUTPUT_DIR, version="logs", name="")
    callbacks = get_callbacks(cfg)
    trainer = Trainer(**cfg.TRAINER, callbacks=callbacks, logger=tb_logger)

    model = Classifier(cfg)
    datamodule = DataModule(cfg)
    # Distributed is initialized in fit, not init
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.CHECKPOINT.PATH)


    # config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../configs")
    # default_train_config = os.path.join(config_dir, "train.yaml")
    # cli = CLI(
    #     model_class=Classifier,
    #     datamodule_class=CAIDMDataModule,
    #     seed_everything_default=True,
    #     trainer_defaults={
    #         # "logger": tb_logger,
    #         "callbacks": [
    #             lr_monitor,
    #             checkpoint,
    #             progress_bar,
    #         ],
    #     },
    #     save_config_kwargs={"overwrite": True},
    #     parser_kwargs={
    #         "fit": {"default_config_files": [default_train_config]},
    #     },
    # )


def setup_config(args: argparse.Namespace):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def setup_logger(cfg: CfgNode):
    """
    https://loguru.readthedocs.io/en/stable/resources/migration.html
    """
    # lit_logger = logging.getLogger("lightning.pytorch")
    # lit_handler = logging.StreamHandler(lit_logger)
    # import ipdb
    # ipdb.set_trace()
    # logger.add(lit_handler, level="INFO")

    # configure logging on module level, redirect to file
    lit_logger = logging.getLogger("lightning.pytorch")
    # lit_logger.addHandler(logging.FileHandler("lightning.log"))
    lit_logger.handlers = [InterceptHandler()]
    lit_logger.setLevel(logging.INFO)
    logger.add(f"{cfg.OUTPUT_DIR}/train.log", level="INFO")


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def get_callbacks(cfg: CfgNode) -> list[Callback]:
    # Maybe add EarlyStopping
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # maybe monitor the val accuracy rather than val loss
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(cfg.OUTPUT_DIR, "checkpoints"),
        filename="epoch={epoch}-step={step}-val_loss={val_loss:.2f}",
        monitor="val_loss",
        save_top_k=cfg.CHECKPOINT.save_top_k,
        mode="min",
        auto_insert_metric_name=False,
        save_last=True,
    )
    progress_bar = ProgressBar()
    # maybe add lightning.pytorch.callbacks.EarlyStopping
    # TODO: add scheduler https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html

    return [lr_monitor, progress_bar, checkpoint]


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
