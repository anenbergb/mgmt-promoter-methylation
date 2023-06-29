import argparse
import logging
import os
import sys

import lightning.pytorch as pl
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    callbacks,
    cli_lightning_logo,
)
from lightning.pytorch.cli import LightningCLI

from mgmt.data.dataloader import CAIDMDataModule
from mgmt.model.model_module import Classifier
from mgmt.utils.logger import setup_logger
from mgmt.utils.progress_bar import ProgressBar

# def main():


#     log_dir = tb_logger.log_dir
#     rank = pl.utilities.distributed._get_rank()
#     if rank == 0:
#         os.makedirs(log_dir, exist_ok=True)
#     logger = logging.getLogger("lightning.pytorch")
#     logger.removeHandler(logger.handlers[0])
#     setup_logger(
#         output = log_dir,
#         distributed_rank=rank,
#         name="lightning.pytorch",
#         abbrev_name="pl",
#         level=logging.INFO
#     )
#     logger.info(f"Command Line Args:\n{args}")
#     logger.info(f"Saving logs and checkpoints to {log_dir}")

#     model = Model(args)
#     dm = DataModule(args)

#     # callbacks
#     lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
#     checkpoint = pl.callbacks.ModelCheckpoint(
#         filename="epoch={epoch}-step={step}-val-loss={val-loss/total:.2f}",
#         monitor="val-loss/total",
#         save_top_k=10,
#         mode="min",
#         auto_insert_metric_name=False,
#         save_last=True,
#     )
#     progress_bar = ProgressBar()
#     trainer = pl.Trainer.from_argparse_args(
#         args, callbacks = [lr_monitor, checkpoint, progress_bar],
#         logger=tb_logger,
#     )
#     # Distributed is initialized in fit, not init
#     trainer.fit(model, datamodule=dm, ckpt_path = ckpt_path)

#     cli = pl.cli.LightningCLI(DemoModel, BoringDataModule)


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
    # Distributed is initialized in fit, not init

    # cli = LightningCLI(
    #     model_class=Classifier,
    #     datamodule_class=CAIDMDataModule,
    #     seed_everything_default=1234,
    #     run=False,  # used to de-activate automatic fitting.
    #     trainer_defaults={"callbacks": ImageSampler(), "max_epochs": 10},
    #     save_config_kwargs={"overwrite": True},
    # )
    # cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    # predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    # print(predictions[0])


if __name__ == "__main__":
    cli_main()
