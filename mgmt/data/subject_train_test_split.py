import math
from typing import Optional

import numpy as np
import torch
import torchio as tio
from fvcore.common.config import CfgNode
from loguru import logger
from sklearn.model_selection import StratifiedKFold


def subjects_kfold_train_val_split(
    subjects: list[tio.Subject], val_ratio: float = 0.20, fold_index: int = 0, random_seed: int = 1
):
    """
    Guarentes that different subjects are in the validation set for each split.

    StratifiedKFold sampler to ensure that the same methylated/non-methylated
    fraction is in the training and test set.
    The ratio on the full training dataset of 577 samples happens to be
    Methylated: 301 (52.17%), unmethylated: 276
    which is close enough to 50% that I'm not worried about dataset skew/bias.
    If it were a worse ratio I would like to control for the balance in the
    validation set to ensure 50% distribution there.

    https://scikit-learn.org/stable/modules/generated/
        sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    k_folds = math.floor(1 / val_ratio)
    if fold_index > k_folds:
        logger.warning(f"Kfold index({fold_index}) > # Folds ({k_folds})")
    fold_index = fold_index % k_folds
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    # sort the subjects first to ensure that all variability comes from the shuffle
    subjects = sorted(subjects, key=lambda x: x.patient_id)
    indices = np.arange(len(subjects))
    category_ids = np.array([x.category_id for x in subjects])
    skf_splits = list(skf.split(indices, category_ids))
    train_indices, val_indices = skf_splits[fold_index]
    train_subjects = [subjects[i] for i in train_indices]
    val_subjects = [subjects[i] for i in val_indices]
    return train_subjects, val_subjects


def subjects_train_val_split(
    subjects: list[tio.Subject],
    val_ratio: float = 0.15,
    random_seed: int = 1,
):
    """
    Splits the subject list into train and val set, and also ensures
    that the val set has equal balance between methylated and
    unmethylated subjects.

    train_val_ratio: train set percentage
    """

    if val_ratio == 0:
        return subjects, []
    elif val_ratio == 1:
        return [], subjects
    else:
        assert val_ratio > 0 and val_ratio < 1

    # sort the subjects first to ensure that all variability comes from the randperm
    subjects = sorted(subjects, key=lambda x: x.patient_id)

    generator = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(len(subjects), generator=generator).tolist()
    num_val = math.floor(len(subjects) * val_ratio)
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

    logger.info(f"{len(subjects)} total subjects. {100*val_ratio:.2f}% val ratio ({len(train)} train, {len(val)} val)")
    logger.info(f"val: {num_methylated} methylated, {len(val) - num_methylated} unmethylated")
    return train, val


def train_test_val_split(subjects: list[tio.Subject], cfg: CfgNode):
    """
    We don't have ground-truth labels for the "test" examples.
    Divide the "train" examples into train / test / val split
    """
    logger.info(f"{len(subjects)} before 'train' filter")
    subjects = [s for s in subjects if s.get("train_test_split", "train") == "train"]
    logger.info(f"{len(subjects)} after 'train' filter")
    split_cfg = cfg.DATA.SPLITS
    assert split_cfg.TEST_RATIO + split_cfg.VAL_RATIO < 1.0
    train_ratio = 1.0 - split_cfg.TEST_RATIO - split_cfg.VAL_RATIO

    train_val_subjects, test_subjects = subjects_train_val_split(
        subjects,
        split_cfg.TEST_RATIO,
        split_cfg.TEST_SEED,
    )
    rel_val_ratio = split_cfg.VAL_RATIO * len(subjects) / len(train_val_subjects)

    train_subjects, val_subjects = subjects_kfold_train_val_split(
        train_val_subjects,
        rel_val_ratio,
        split_cfg.FOLD_INDEX,
        split_cfg.VAL_SEED,
    )
    actual_train_per = 100 * len(train_subjects) / len(subjects)
    actual_test_per = 100 * len(test_subjects) / len(subjects)
    actual_val_per = 100 * len(val_subjects) / len(subjects)
    sum_train_test_val = len(train_subjects) + len(test_subjects) + len(val_subjects)
    if sum_train_test_val != len(subjects):
        logger.warning(f"{len(subjects) - sum_train_test_val} missing subjects after train/test/val split")
    logger.info(
        f"total {len(subjects)}. "
        f"train {len(train_subjects)} ({100*train_ratio:.1f}% -> {actual_train_per:.1f}%) "
        f"test {len(test_subjects)} ({100 * split_cfg.TEST_RATIO:.1f}% -> {actual_test_per:.1f}%) "
        f"val {len(val_subjects)} ({100 * split_cfg.VAL_RATIO:.1f}% -> {actual_val_per:.1f}%)"
    )
    return train_subjects, test_subjects, val_subjects
