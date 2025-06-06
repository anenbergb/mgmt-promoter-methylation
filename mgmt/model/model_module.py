from collections import defaultdict
from dataclasses import dataclass, field, fields

import numpy as np
import torch
import torchio
import torchmetrics
from fvcore.common.config import CfgNode
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from loguru import logger
from torch.nn import BCEWithLogitsLoss

from mgmt.data.subject_utils import get_subjects_from_batch
from mgmt.data_science.plot_center_mass import add_color_border
from mgmt.model import build_model
from mgmt.utils.lr_scheduler import build_lr_scheduler
from mgmt.utils.optimizer import build_optimizer
from mgmt.visualize.subject import plot_subject_with_label
from mgmt.visualize.visualize import plot_classification_grid


class Classifier(LightningModule):
    def __init__(
        self,
        cfg: CfgNode,
        steps_per_epoch=50,
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
        logger.info("Initializing LightningModule: Classifer")
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch

        # TODO: decide how checkpointing will work.... do I need to save params here
        # self.save_hyperparameters()

        # equivalent
        # self.save_hyperparameters("layer_1_dim", "learning_rate")
        # self.save_hyperparameters(ignore=[...])

        self.net = build_model(cfg)

        # TODO: consider adding additional losses
        self.criterion = BCEWithLogitsLoss()

        self.train_acc = torchmetrics.classification.BinaryAccuracy(threshold=cfg.METRICS.THRESHOLD)
        self.val_acc = torchmetrics.classification.BinaryAccuracy(threshold=cfg.METRICS.THRESHOLD)
        self.train_auc = torchmetrics.classification.BinaryAUROC()
        self.val_auc = torchmetrics.classification.BinaryAUROC()

        self.validation_step_outputs = []

    # TODO: consider adding a from_config(cfg: cfgNode) constructor method

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        """
        Pytorch Lightning documentation
            - https://lightning.ai/docs/pytorch/stable/common/optimization.html
            - https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers

        Other optimizers
            - https://github.com/huggingface/pytorch-image-models/blob/main/timm/scheduler/scheduler.py

        """
        optimizer = build_optimizer(self.net.parameters(), self.cfg)
        scheduler = build_lr_scheduler(optimizer, self.cfg, self.steps_per_epoch)
        interval = "step" if self.cfg.SOLVER.SCHEDULER_NAME == "OneCycleLR" else "epoch"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val/loss",
                "strict": True,
            },
        }

    # Consider also configuring scheduler

    def prepare_batch(self, batch):
        return batch[self.cfg.DATA.MODALITY][torchio.DATA], batch["category_id"]

    def infer_batch(self, batch):
        x, target = self.prepare_batch(batch)
        logits = self.net(x).flatten()
        preds = torch.sigmoid(logits)
        binary_preds = (preds > self.cfg.METRICS.THRESHOLD).to(torch.int64)
        return logits, preds, binary_preds, target

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop

        logging:
            - https://lightning.ai/docs/pytorch/stable/extensions/logging.html
            - https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/core/module.py#L344

            rank_zero_only: Whether the value will be logged only on rank 0.
                This will prevent synchronization which would produce a deadlock
                as not all processes would perform this log call.

        Under the hood, Lightning does the following (pseudocode):
        model.train()
        torch.set_grad_enabled(True)

        for batch_idx, batch in enumerate(train_dataloader):
            loss = training_step(batch, batch_idx)

            # clear gradients
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update parameters
            optimizer.step()


        Dictionary must include "loss" key
        """
        logits, preds, binary_preds, target = self.infer_batch(batch)
        loss = self.criterion(logits, target.to(torch.float))
        self.train_acc(binary_preds, target)
        self.train_auc(preds, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.cfg.DATA.BATCH_SIZE)
        self.log_dict(
            {
                "train/accuracy": self.train_acc,
                "train/auc": self.train_auc,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.cfg.DATA.BATCH_SIZE,
        )
        return {"loss": loss, "preds": preds, "target": target}

    # def training_step_end(self, outputs)
    # I think train_step_end is only required if training with data parallel
    # https://github.com/Lightning-AI/lightning/issues/8105

    # TODO: Not sure if I need to manually step
    # def on_train_epoch_end(self):
    #     sch = self.lr_schedulers()

    #     # If the selected scheduler is a ReduceLROnPlateau scheduler.
    #     if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #         sch.step(self.trainer.callback_metrics["val/loss"])

    def validation_step(self, batch, batch_idx):
        logits, preds, binary_preds, target = self.infer_batch(batch)
        loss = self.criterion(logits, target.to(torch.float))
        self.val_acc(binary_preds, target)
        self.val_auc(preds, target)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.cfg.DATA.BATCH_SIZE)
        self.log(
            "val/accuracy",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.cfg.DATA.BATCH_SIZE,
        )
        self.log(
            "val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.cfg.DATA.BATCH_SIZE
        )

        # only visualize first and final epoch
        # TODO: make sure this works with restart
        if self.current_epoch in (0, self.cfg.TRAINER.max_epochs - 1):
            self.visualize_predictions(batch, binary_preds, target)

        self.validation_step_outputs.append(
            {
                "preds": preds.cpu().numpy(),
                "target": target.cpu().numpy(),
                "patient_id": batch["patient_id"].cpu().numpy(),
            }
        )

        return {"loss": loss, "preds": preds, "target": target}

    def on_validation_epoch_end(self):
        val_output_keys = ("preds", "target", "patient_id")
        all_outputs = {key: np.concatenate([x[key] for x in self.validation_step_outputs]) for key in val_output_keys}
        self.plot_classification_grid(all_outputs["preds"], all_outputs["target"], all_outputs["patient_id"])
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        preds, _, _, _ = self.infer_batch(batch)
        return preds

    def visualize_predictions(self, batch, binary_preds, targets):
        """
        Tensorboard
            - https://pytorch.org/docs/stable/tensorboard.html
        """
        batch_subjects = get_subjects_from_batch(batch)
        for subject, pred, target in zip(batch_subjects, binary_preds, targets):
            # TODO: make sure this works for concat mode
            image = plot_subject_with_label(
                subject,
                show=False,
                return_fig=False,
                # figsize=(6.4, 1.6),
                add_metadata=True,
                add_tumor_legend=True,
            )
            color = "green" if pred == target else "red"
            image = add_color_border(image, color=color)
            tensor = torch.from_numpy(image)  # HWC
            self.logger.experiment.add_image(
                f"val_subject/{subject.patient_id}", tensor, global_step=self.global_step, dataformats="HWC"
            )

    def plot_classification_grid(self, preds, target, patient_id):
        grid = plot_classification_grid(preds, target, patient_id)
        tensor = torch.from_numpy(grid)  # HWC
        self.logger.experiment.add_image(
            f"val_classification_grid", tensor, global_step=self.global_step, dataformats="HWC"
        )


@dataclass
class PredictionsNumpy:
    logits: np.ndarray
    probs: np.ndarray
    binary_preds: np.ndarray

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, i):
        return {
            "logits": self.logits[i] if self.logits is not None else None,
            "probs": self.probs[i] if self.probs is not None else None,
            "binary_preds": self.binary_preds[i] if self.binary_preds is not None else None,
        }

    @classmethod
    def concatenate(cls, predictions_list):
        assert len(predictions_list) > 0
        attrs = defaultdict(list)

        is_none = set()
        for p in predictions_list:
            for name in ("logits", "probs", "binary_preds"):
                val = getattr(p, name)

                if name in is_none:
                    continue
                elif val is None:
                    is_none.add(name)
                    continue

                attrs[name].append(val)

        concat = {k: np.concatenate(v) for k, v in attrs.items()}
        for key in is_none:
            concat[key] = None
        return PredictionsNumpy(**concat)


@dataclass
class Predictions:
    # raw scores in range (-inf, +inf)
    logits: torch.tensor
    threshold: float = field(default=0.5)
    # score in range (0, 1.0)
    probs: torch.tensor = field(default=None)
    # binary prediction {0, 1}
    binary_preds: torch.tensor = field(init=False)

    def __post_init__(self):
        if self.probs is None:
            self.probs = torch.sigmoid(self.logits)
        self.binary_preds = (self.probs > self.threshold).to(torch.int64)

    def to_numpy(self):
        return PredictionsNumpy(
            logits=self.logits.cpu().numpy() if self.logits is not None else None,
            probs=self.probs.cpu().numpy() if self.probs is not None else None,
            binary_preds=self.binary_preds.cpu().numpy() if self.binary_preds is not None else None,
        )


class PredictionsMultiResolution:
    def __init__(self, logits: dict[str, torch.tensor], threshold: float = 0.5):
        self.threshold = threshold
        self.names = list(logits.keys())
        self.predictions = {}
        for name, logit_vec in logits.items():
            self.predictions[name] = Predictions(logit_vec, threshold)

        self.average_by_logits = self._average_by_logits()
        self.average_by_probs = self._average_by_probs()

    def _average_by_logits(self):
        logits = [x.logits for x in self.predictions.values()]
        logits_avg = torch.mean(torch.stack(logits), dim=0)
        return Predictions(logits_avg, self.threshold)

    def _average_by_probs(self):
        probs = [x.probs for x in self.predictions.values()]
        probs_avg = torch.mean(torch.stack(probs), dim=0)
        return Predictions(logits=None, threshold=self.threshold, probs=probs_avg)

    def __iter__(self):
        for name, p in self.predictions.items():
            yield (name, p)


@dataclass
class ValidationOutput:
    predictions: PredictionsNumpy
    targets: np.ndarray
    patient_ids: np.ndarray

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        ret = self.predictions[i]
        ret["targets"] = self.targets[i]
        ret["patient_ids"] = self.patient_ids[i]
        return ret

    @classmethod
    def concatenate(cls, val_list):
        assert len(val_list) > 0
        attrs = defaultdict(list)
        for v in val_list:
            for name in ("predictions", "targets", "patient_ids"):
                val = getattr(v, name)
                attrs[name].append(val)

        return ValidationOutput(
            PredictionsNumpy.concatenate(attrs["predictions"]),
            np.concatenate(attrs["targets"]),
            np.concatenate(attrs["patient_ids"]),
        )


class ClassifierMultiResolution(LightningModule):
    def __init__(
        self,
        cfg: CfgNode,
        steps_per_epoch=50,
    ):
        super().__init__()
        logger.info("Initializing LightningModule: ClassifierMultiResolution")
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch

        self.net = build_model(cfg)
        self.head_names = list(self.net.heads.keys())

        # verify that head_names make sense

        self.criterion = BCEWithLogitsLoss()
        self.add_metrics()
        self.validation_step_outputs = []

    def add_metrics(self):
        self.train_acc = {}
        self.val_acc = {}
        self.train_auc = {}
        self.val_auc = {}
        for train_val in ("train", "val"):
            for head_name in self.head_names:
                setattr(
                    self,
                    f"{train_val}_{head_name}_accuracy",
                    torchmetrics.classification.BinaryAccuracy(threshold=self.cfg.METRICS.THRESHOLD),
                )
                setattr(self, f"{train_val}_{head_name}_auc", torchmetrics.classification.BinaryAUROC())

    def get_metric(self, metric_name: str, head_name: str, is_train: bool = True):
        train_val = "train" if is_train else "val"
        key = f"{train_val}_{head_name}_{metric_name}"
        return getattr(self, key)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.net.parameters(), self.cfg)
        scheduler = build_lr_scheduler(optimizer, self.cfg, self.steps_per_epoch)
        interval = "step" if self.cfg.SOLVER.SCHEDULER_NAME == "OneCycleLR" else "epoch"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val/loss",
                "strict": True,
            },
        }

    def apply_criterion(self, predictions: PredictionsMultiResolution, target: torch.tensor):
        # CHECK: double check that I can use the same BCE instance
        target = target.to(torch.float)
        losses = {}
        for name, p in predictions:
            losses[name] = self.criterion(p.logits, target)
        return losses

    def infer_batch(self, batch):
        x = batch[self.cfg.DATA.MODALITY][torchio.DATA]
        target = batch["category_id"]
        tumor_mask = (batch["tumor"][torchio.DATA] > 0).type(torch.float)
        # CHECK: that backprop doesn't update tumor_mask
        logits_map = self.net(x, tumor_mask)
        predictions = PredictionsMultiResolution(logits_map, self.cfg.METRICS.THRESHOLD)
        return predictions, target

    def training_step(self, batch, batch_idx):
        predictions, target = self.infer_batch(batch)
        losses = self.apply_criterion(predictions, target)
        total_loss = 0
        for name, loss in losses.items():
            self.log(
                f"train-loss/{name}",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=self.cfg.DATA.BATCH_SIZE,
            )
            total_loss += loss
        self.log(
            f"train-loss/total",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.cfg.DATA.BATCH_SIZE,
        )

        self.log_train_metrics(predictions, target)
        return total_loss

    def log_train_metrics(self, predictions: PredictionsMultiResolution, target: torch.tensor):
        for name, p in predictions:
            self.get_metric("accuracy", name, True)(p.binary_preds, target)
            self.get_metric("auc", name, True)(p.probs, target)
            self.log_dict(
                {
                    f"train-accuracy/{name}": self.get_metric("accuracy", name, True),
                    f"train-auc/{name}": self.get_metric("auc", name, True),
                },
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=self.cfg.DATA.BATCH_SIZE,
            )

    def validation_step(self, batch, batch_idx):
        predictions, target = self.infer_batch(batch)
        losses = self.apply_criterion(predictions, target)
        total_loss = 0
        for name, loss in losses.items():
            self.log(
                f"val-loss/{name}",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=self.cfg.DATA.BATCH_SIZE,
            )
            total_loss += loss
        self.log(
            f"val-loss/total",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.cfg.DATA.BATCH_SIZE,
        )

        self.log_val_metrics(predictions, target)

        output = ValidationOutput(
            predictions=predictions.average_by_probs.to_numpy(),
            targets=target.cpu().numpy(),
            patient_ids=batch["patient_id"].cpu().numpy(),
        )

        # only visualize first and final epoch
        # TODO: make sure this works with restart
        if self.current_epoch in (0, self.cfg.TRAINER.max_epochs - 1):
            self.visualize_predictions(batch, output)

        self.validation_step_outputs.append(output)

    def log_val_metrics(self, predictions: PredictionsMultiResolution, target: torch.tensor):
        for name, p in predictions:
            self.get_metric("accuracy", name, False)(p.binary_preds, target)
            self.get_metric("auc", name, False)(p.probs, target)
            self.log(
                f"val-accuracy/{name}",
                self.get_metric("accuracy", name, False),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.cfg.DATA.BATCH_SIZE,
            )
            self.log(
                f"val-auc/{name}",
                self.get_metric("auc", name, False),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=self.cfg.DATA.BATCH_SIZE,
            )

    def on_validation_epoch_end(self):
        val_outputs = ValidationOutput.concatenate(self.validation_step_outputs)
        self.plot_classification_grid(val_outputs.predictions.probs, val_outputs.targets, val_outputs.patient_ids)
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predictions, target = self.infer_batch(batch)
        output = ValidationOutput(
            predictions=predictions.average_by_probs.to_numpy(),
            targets=target.cpu().numpy(),
            patient_ids=batch["patient_id"].cpu().numpy(),
        )
        return output

    def visualize_predictions(self, batch, val_output):
        """
        Tensorboard
            - https://pytorch.org/docs/stable/tensorboard.html
        """
        batch_subjects = get_subjects_from_batch(batch)
        for i, subject in enumerate(batch_subjects):
            # TODO: make sure this works for concat mode
            image = plot_subject_with_label(
                subject,
                show=False,
                return_fig=False,
                # figsize=(6.4, 1.6),
                add_metadata=True,
                add_tumor_legend=True,
            )
            target = val_output[i]["targets"]
            binary_pred = val_output[i]["binary_preds"]
            color = "green" if binary_pred == target else "red"
            image = add_color_border(image, color=color)
            tensor = torch.from_numpy(image)  # HWC
            self.logger.experiment.add_image(
                f"val_subject/{subject.patient_id}", tensor, global_step=self.global_step, dataformats="HWC"
            )

    def plot_classification_grid(self, preds, target, patient_id):
        grid = plot_classification_grid(preds, target, patient_id)
        tensor = torch.from_numpy(grid)  # HWC
        self.logger.experiment.add_image(
            f"val_classification_grid", tensor, global_step=self.global_step, dataformats="HWC"
        )
