import copy
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
from mgmt.transforms.patch_sampler_probability_map import AddPatchSamplerProbabilityMap
from mgmt.transforms.rescale_intensity import RescaleIntensity
from mgmt.transforms.skull_crop import SkullCropTransform


def load_patient_exclusion(filename: str) -> np.ndarray:
    with open(filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        indices = [int(row[0]) for row in csv_reader]
    arr = np.array(indices, dtype=int)
    return arr


# TODO: probably want to perform concat in the module_module, so that rescale
# intensity can be applied independently to each modality
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


from torch.utils.data._utils.collate import collate, default_collate_fn_map


def default_collate_wrapper(batch):
    print("default collate!!!")
    import ipdb

    ipdb.set_trace()
    return collate(batch, collate_fn_map=default_collate_fn_map)


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
            modality = self.cfg.DATA.MODALITY
            if modality == "concat":
                modality = self.cfg.DATA.MODALITY_CONCAT
            else:
                modality = [modality]
            self.subjects = nifti_load_subjects(
                self.cfg.DATA.NIFTI.FOLDER_PATH,
                self.cfg.DATA.NIFTI.TRAIN_LABELS,
                modality,
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

        # double check whether this is duplicated for each GPU
        if not self.cfg.DATA.LAZY_LOAD_TRAIN:
            for s in train_subjects:
                s.load()

        val_transforms = self.get_transforms(train=False)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=val_transforms)

        if self.cfg.PATCH_BASED_TRAINER.ENABLED:
            train_transforms = self.get_transforms_patches()
            self.train_set = tio.SubjectsDataset(train_subjects, transform=train_transforms)

            sampler = tio.data.LabelSampler(**self.cfg.PATCH_BASED_TRAINER.LABEL_SAMPLER)
            self.train_patches_queue = tio.Queue(
                subjects_dataset=self.train_set,
                sampler=sampler,
                subject_sampler=None,
                num_workers=self.cfg.DATA.NUM_WORKERS,
                **self.cfg.PATCH_BASED_TRAINER.QUEUE,
            )
        else:
            train_transforms = self.get_transforms(train=True)
            self.train_set = tio.SubjectsDataset(train_subjects, transform=train_transforms)

    def get_transforms_patches(self):
        transforms = []

        if self.cfg.PREPROCESS.TO_CANONICAL_ENABLED:
            transforms.append(tio.ToCanonical())

        transforms.append(SkullCropTransform(**self.cfg.PREPROCESS.SKULL_CROP_TRANSFORM))
        transforms.append(
            AddPatchSamplerProbabilityMap(
                patch_size=self.cfg.PATCH_BASED_TRAINER.WEIGHTED_SAMPLER.patch_size,
                device="cuda",
                segmentation_mask_key="tumor",
                probability_map_name=self.cfg.PATCH_BASED_TRAINER.WEIGHTED_SAMPLER.probability_map,
            )
        )

        # # pad half the patch size in order to avoid sampling out of bounds
        # patch_size = self.cfg.PATCH_BASED_TRAINER.LABEL_SAMPLER.patch_size
        # padding = np.ceil((np.array(patch_size) / 2)).astype(np.int32).tolist()
        # transforms.append(tio.Pad(padding=padding))

        if self.cfg.AUGMENT.RANDOM_AFFINE_ENABLED:
            transforms.append(tio.RandomAffine(**self.cfg.AUGMENT.RANDOM_AFFINE))
        if self.cfg.AUGMENT.RANDOM_GAMMA_ENABLED:
            transforms.append(tio.RandomGamma(**self.cfg.AUGMENT.RANDOM_GAMMA))
        if self.cfg.AUGMENT.RANDOM_BIAS_FIELD:
            transforms.append(tio.RandomBiasField(**self.cfg.AUGMENT.RANDOM_BIAS_FIELD))
        if self.cfg.AUGMENT.RANDOM_MOTION_ENABLED:
            transforms.append(tio.RandomMotion(**self.cfg.AUGMENT.RANDOM_MOTION))

        int_trans = self.get_rescale_intensity_transforms(train=True)
        transforms.extend(int_trans)
        return tio.Compose(transforms)

    def get_transforms(self, train=True):
        transforms = []
        if self.cfg.PREPROCESS.TO_CANONICAL_ENABLED:
            transforms.append(tio.ToCanonical())

        if self.cfg.PREPROCESS.EARLY_CROP_LARGEST_TUMOR_ENABLED:
            transforms.append(CropLargestTumor(**self.cfg.PREPROCESS.EARLY_CROP_LARGEST_TUMOR))

        if train and self.cfg.AUGMENT.RANDOM_AFFINE_ENABLED:
            transforms.append(tio.RandomAffine(**self.cfg.AUGMENT.RANDOM_AFFINE))
        if train and self.cfg.AUGMENT.RANDOM_GAMMA_ENABLED:
            transforms.append(tio.RandomGamma(**self.cfg.AUGMENT.RANDOM_GAMMA))
        if train and self.cfg.AUGMENT.RANDOM_BIAS_FIELD:
            transforms.append(tio.RandomBiasField(**self.cfg.AUGMENT.RANDOM_BIAS_FIELD))
        if train and self.cfg.AUGMENT.RANDOM_MOTION_ENABLED:
            transforms.append(tio.RandomMotion(**self.cfg.AUGMENT.RANDOM_MOTION))

        if self.cfg.PREPROCESS.RESCALE_INTENSITY.BEFORE_CROP:
            int_trans = self.get_rescale_intensity_transforms(train=train)
            transforms.extend(int_trans)
        if self.cfg.PREPROCESS.CROP_LARGEST_TUMOR_ENABLED:
            transforms.append(CropLargestTumor(**self.cfg.PREPROCESS.CROP_LARGEST_TUMOR))
        if not self.cfg.PREPROCESS.RESCALE_INTENSITY.BEFORE_CROP:
            int_trans = self.get_rescale_intensity_transforms(train=train)
            transforms.extend(int_trans)

        if self.cfg.PREPROCESS.RESIZE_ENABLED:
            transforms.append(tio.Resize(**self.cfg.PREPROCESS.RESIZE))
        transforms.append(tio.EnsureShapeMultiple(**self.cfg.PREPROCESS.ENSURE_SHAPE_MULTIPLE))
        return tio.Compose(transforms)

    def get_rescale_intensity_transforms(self, train=True):
        transforms = []
        if self.cfg.PREPROCESS.RESCALE_INTENSITY_ENABLED:
            kwargs = copy.copy(self.cfg.PREPROCESS.RESCALE_INTENSITY)
            skull_mask = kwargs.pop("SKULL_MASK")
            kwargs.pop("BEFORE_CROP")
            transforms.append(RescaleIntensity(skull_mask=skull_mask, **kwargs))
        if train and self.cfg.AUGMENT.RANDOM_NOISE_ENABLED:
            transforms.append(tio.RandomNoise(**self.cfg.AUGMENT.RANDOM_NOISE))
            # rescale back to the target intensity scale range
            # do not need to apply percentile filtering or mask
            if self.cfg.PREPROCESS.RESCALE_INTENSITY_ENABLED:
                transforms.append(
                    RescaleIntensity(
                        out_min_max=self.cfg.PREPROCESS.RESCALE_INTENSITY.out_min_max,
                    )
                )
        return transforms

    def train_dataloader(self):
        """
        Trainer fit() method calls this.
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader
        """
        if self.cfg.PATCH_BASED_TRAINER.ENABLED:
            return DataLoader(
                self.train_patches_queue,
                batch_size=self.cfg.DATA.BATCH_SIZE,
                num_workers=0,  # this must be 0
            )

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
            self.val_set,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            # collate_fn=default_collate_wrapper,
        )
