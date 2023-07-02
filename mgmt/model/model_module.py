import torch
import torchio
import torchmetrics
from fvcore.common.config import CfgNode
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from monai.networks.nets.resnet import resnet10
from torch.nn import BCEWithLogitsLoss

from mgmt.model import build_model


class Classifier(LightningModule):
    def __init__(
        self,
        cfg: CfgNode,
    ):
        """

        Resnet10
            - https://docs.monai.io/en/stable/_modules/monai/networks/nets/resnet.html#ResNet

        save_hyperparameters
            - https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#id1
            - store all of the __init__ args under self.hparams attribute

        torchmetrics
            - https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
            - https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html

        submodules
            - https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html
            - subclass_mode_model = True
        OptimizerCallable and LRSchedulerCallable
            - https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html
        """
        super().__init__()
        self.cfg = cfg

        # TODO: decide how checkpointing will work.... do I need to save params here
        # self.save_hyperparameters()

        # equivalent
        # self.save_hyperparameters("layer_1_dim", "learning_rate")
        # self.save_hyperparameters(ignore=[...])

        self.net = build_model(cfg)

        # TODO: consider adding additional losses
        self.criterion = BCEWithLogitsLoss()
        self.optimizer_class = torch.optim.AdamW

        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.train_auc = torchmetrics.classification.BinaryAUROC()
        self.val_auc = torchmetrics.classification.BinaryAUROC()

    # TODO: consider adding a from_config(cfg: cfgNode) constructor method

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.net.parameters(), lr=self.cfg.SOLVER.BASE_LR)
        return optimizer

    # Consider also configuring scheduler

    def prepare_batch(self, batch):
        return batch[self.cfg.DATA.MODALITY][torchio.DATA], batch["category_id"]

    def infer_batch(self, batch):
        x, target = self.prepare_batch(batch)
        preds = self.net(x).flatten()
        return preds, target

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop

        logging:
            - https://lightning.ai/docs/pytorch/stable/extensions/logging.html
            - https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/core/module.py#L344

            rank_zero_only: Whether the value will be logged only on rank 0.
                This will prevent synchronization which would produce a deadlock
                as not all processes would perform this log call.
        """
        preds, target = self.infer_batch(batch)
        loss = self.criterion(preds, target.to(torch.float))
        self.train_acc(preds, target)
        self.train_auc(preds, target)
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": self.train_acc,
                "train_auc": self.train_auc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.cfg.DATA.BATCH_SIZE,
        )
        return {"loss": loss, "preds": preds, "target": target}

    # def training_step_end(self, outputs)
    # I think train_step_end is only required if training with data parallel
    # https://github.com/Lightning-AI/lightning/issues/8105

    def validation_step(self, batch, batch_idx):
        preds, target = self.infer_batch(batch)
        loss = self.criterion(preds, target.to(torch.float))
        self.val_acc(preds, target)
        self.val_auc(preds, target)
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": self.val_acc,
                "val_auc": self.val_auc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.cfg.DATA.BATCH_SIZE,
        )
        return {"loss": loss, "preds": preds, "target": target}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, _ = self.infer_batch(batch)
        return y_hat
