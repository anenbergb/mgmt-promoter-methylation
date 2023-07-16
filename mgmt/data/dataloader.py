import csv
import math
from typing import Dict, Optional

import numpy as np
import torch
import torchio as tio
from fvcore.common.config import CfgNode
from lightning.pytorch import LightningDataModule
from loguru import logger
from torch.utils.data import DataLoader

from mgmt.data.nifti import load_subjects as nifti_load_subjects
from mgmt.data.numpy import load_subjects as numpy_load_subjects
from mgmt.data.subject_transforms import CropLargestTumor


def load_patient_exclusion(filename: str) -> np.ndarray:
    with open(filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        indices = [int(row[0]) for row in csv_reader]
    arr = np.array(indices, dtype=int)
    return arr


def make_concat_image(subject: tio.Subject, modality: list[str] = ["t2w"]) -> tio.ScalarImage:
    tensors = []
    for m in modality:
        assert m in subject
        tensors.append(subject[m].tensor)
    tensor = torch.cat(tensors, dim=0)
    return tio.ScalarImage(tensor=tensor)


def get_max_shape(subjects):
    dataset = tio.SubjectsDataset(subjects)
    shapes = np.array([s.spatial_shape for s in dataset])
    return shapes.max(axis=0)


def subjects_train_val_split(
    subjects: list[tio.Subject],
    train_val_ratio: float = 0.85,
    generator: Optional[torch.Generator] = torch.default_generator,
):
    """
    Splits the subject list into train and val set, and also ensures
    that the val set has equal balance between methylated and
    unmethylated subjects.
    """
    if train_val_ratio == 0:
        return [], subjects
    elif train_val_ratio == 1:
        return subjects, []
    else:
        assert train_val_ratio > 0 and train_val_ratio < 1

    indices = torch.randperm(len(subjects), generator=generator).tolist()
    num_val = math.floor(len(subjects) * (1 - train_val_ratio))
    num_methylated = 0
    val = []
    train = []
    for i in indices:
        is_methylated = subjects[i].category == "methylated"
        num_unmethylated = len(val) - num_methylated
        if len(val) < num_val and (
            (is_methylated and num_methylated < num_val / 2) or (not is_methylated and num_unmethylated < num_val / 2)
        ):
            val.append(subjects[i])
            num_methylated += int(is_methylated)
        else:
            train.append(subjects[i])

    logger.info(
        f"{len(subjects)} total subjects. {100*train_val_ratio:.2f}% train/val ratio ({len(train)} train, {len(val)} val)"
    )
    logger.info(f"val: {num_methylated} methylated, {len(val) - num_methylated} unmethylated")
    return train, val


# IDEA: Could define an argument class that is passed to init


class DataModule(LightningDataModule):
    """
    DataModule basics
    - prepare_data (how to download, tokenize, etc…)
    - setup (how to split, define dataset, etc…)
    - train_dataloader
    - val_dataloader
    - test_dataloader
    - predict_dataloader
    """

    def __init__(
        self,
        cfg: CfgNode,
        # option to add an transform to concat different combinations
        # of modalities into a 'combined' tensor
    ):
        """

        Args:
            crop_dim: spatial dimensions W, H, D
        """
        super().__init__()
        self.cfg = cfg

        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        """
        Step 1
        Load dataset and construct subject list.
        prepare_data is called within a single process on CPU.
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
        """
        if self.cfg.DATA.SOURCE == "numpy":
            subjects = numpy_load_subjects(self.cfg.NUMPY.FILEPATH_NPZ)
            exs = load_patient_exclusion(self.cfg.DATA.PATIENT_EXCLUSION_CSV)
            self.subjects = [s for s in subjects if s.patient_id not in exs]
        elif self.cfg.DATA.SOURCE == "nifti":
            self.subjects = nifti_load_subjects(
                self.cfg.DATA.NIFTI.FOLDER_PATH,
                self.cfg.DATA.NIFTI.TRAIN_LABELS,
                self.cfg.DATA.NIFTI.TEST_FOLDER_PREFIX,
            )
        self.add_combined_image()

    def add_combined_image(self):
        if self.cfg.DATA.MODALITY == "concat":
            for subject in self.subjects:
                image = make_concat_image(subject, self.cfg.DATA.MODALITY_CONCAT)
                subject.add_image(image, self.cfg.DATA.MODALITY)

    def setup(self, stage=None):
        """
        Step 2
        - data operations perform on every GPU.
        - perform train/val/test splits
        - create datasets
        - apply transforms
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup

        Args:
            stage: It is used to separate setup logic for trainer.{fit,validate,test,predict}.
        """
        generator = torch.Generator().manual_seed(self.cfg.DATA.TRAIN_VAL_MANUAL_SEED)

        # TODO: allow processing of test subjects
        subjects = [s for s in self.subjects if s.get("train_test_split", "train") == "train"]
        train_subjects, val_subjects = subjects_train_val_split(subjects, self.cfg.DATA.TRAIN_VAL_RATIO, generator)

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)

    def get_early_preprocessing_transform(self):
        # TODO: make this configurable
        transforms = []
        cfg = self.cfg.PREPROCESS
        if cfg.TO_CANONICAL.ENABLED:
            transforms.append(tio.ToCanonical())

        # Consider using percentiles (0.5, 99.5) to control for possible outliers
        tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(0, 100)),

        if cfg.CROP_LARGEST_TUMOR.ENABLED:
            transforms.append(CropLargestTumor(crop_dim=cfg.CROP_LARGEST_TUMOR.crop_dim))

        preprocess = tio.Compose(
            [
                tio.EnsureShapeMultiple(self.cfg.DATA.SHAPE_MULTIPLE),
            ]
        )
        # tio.OneHot() - one-hot encoding could be applied to the tumor label tensor
        return preprocess

    def get_augmentation_transform(self):
        """

        RandomAffine
        - random rotations of the cropped tumor should be OK since you don't have the spatial context.

        """
        # TODO: consider moving a lot of these augmentations before the crop
        augment = tio.Compose(
            [
                tio.RandomAffine(
                    # could consider slightly rescaling of (0.75, 1.25, 0.75, 1.25, 1, 1)
                    scales=(1, 1, 1, 1, 1, 1),
                    # only rotate about the z-axis (depth)
                    degrees=(0, 0, 0, 0, 0, 360),
                ),
                tio.RandomGamma(p=0.5),
                # https://torchio.readthedocs.io/transforms/augmentation.html#randomnoise
                # greater than 0.1 looks pretty grainy
                tio.RandomNoise(p=0.5, std=(0, 0.1)),
                tio.RandomMotion(p=0.1, translation=(-1, 1), degrees=(-1, 1)),
                tio.RandomBiasField(p=0.1, coefficients=(-0.1, 0.1)),
            ]
        )
        return augment

    def train_dataloader(self):
        """
        Trainer fit() method calls this.
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val-dataloader
        """
        return DataLoader(
            self.val_set, batch_size=self.cfg.DATA.BATCH_SIZE, num_workers=self.cfg.DATA.NUM_WORKERS, pin_memory=True
        )
