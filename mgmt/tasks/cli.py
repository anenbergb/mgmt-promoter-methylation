import argparse
import logging
import os
import sys

import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

from mgmt.data.dataloader import CAIDMDataModule
from mgmt.model.model_module import Classifier
from mgmt.utils.logger import setup_logger
from mgmt.utils.progress_bar import ProgressBar


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """
        https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html#cli-link-arguments
        """
        parser.link_arguments("data.batch_size", "model.batch_size")


def cli_main():
    """
    LightningCLI https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html
        - https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html
    Trainer
    - https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
    """
    tb_logger = pl.loggers.TensorBoardLogger(save_dir="", version="logs", name="")
    # callbacks
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    # maybe monitor the val accuracy rather than val loss
    checkpoint = pl.callbacks.ModelCheckpoint(
        filename="epoch={epoch}-step={step}-val_loss={val_loss:.2f}",
        monitor="val_loss",
        save_top_k=10,
        mode="min",
        auto_insert_metric_name=False,
        save_last=True,
    )
    progress_bar = ProgressBar()
    # maybe add lightning.pytorch.callbacks.EarlyStopping
    # TODO: add scheduler https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html

    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../configs")
    default_train_config = os.path.join(config_dir, "train.yaml")
    cli = CLI(
        model_class=Classifier,
        datamodule_class=CAIDMDataModule,
        seed_everything_default=True,
        trainer_defaults={
            # "logger": tb_logger,
            "callbacks": [
                lr_monitor,
                checkpoint,
                progress_bar,
            ],
        },
        save_config_kwargs={"overwrite": True},
        parser_kwargs={
            "fit": {"default_config_files": [default_train_config]},
        },
    )


if __name__ == "__main__":
    cli_main()
