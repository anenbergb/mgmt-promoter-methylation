import argparse
import copy
import os
import pickle
import sys
from collections import OrderedDict
from glob import glob

import nibabel as nib
from fvcore.common.config import CfgNode
from lightning.pytorch import seed_everything
from loguru import logger
from tqdm import tqdm

from mgmt.data.dataloader import DataModule
from mgmt.data.nifti import get_subject_folders, load_subjects
from mgmt.tasks.train import setup_config
from mgmt.transforms.skull_crop import SkullCropTransform
from mgmt.utils.logger import setup_logger


def get_image_modality(file_path: str) -> str:
    return os.path.basename(file_path).split(".")[0]


def save_transformed_subject(subject, extension=".skull-crop.nii"):
    for image in subject.get_images(intensity_only=False):
        dirname = os.path.dirname(image.path)
        modality = get_image_modality(image.path)
        save_path = f"{dirname}/{modality}{extension}"
        image.save(save_path)


def save_subject_nifti(subjects, transforms, extension):
    for subject in tqdm(subjects):
        trans_subject = transforms(subject)
        save_transformed_subject(trans_subject, extension)


def save_subject_pickle(subjects, transforms, pickle_save_path: str):
    os.makedirs(pickle_save_path, exist_ok=True)
    logger.info(f"Saving transformed subjects to {pickle_save_path}")
    for subject in tqdm(subjects):
        trans_subject = transforms(subject)
        save_path = os.path.join(pickle_save_path, f"{subject.patient_id_str}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(trans_subject, f)


def save_dataset_pickle(subjects, transforms, pickle_save_path: str):
    os.makedirs(os.path.dirname(pickle_save_path), exist_ok=True)
    logger.info(f"Saving transformed subjects to {pickle_save_path}")
    transformed_subjects = [transforms(subject) for subject in tqdm(subjects)]
    with open(pickle_save_path, "wb") as f:
        pickle.dump(transformed_subjects, f)


def main(
    cfg: CfgNode,
    extension: str | None,
    pickle_save_path: str,
    mode: str,
    modalities: list[str] = ["fla", "t1w", "t1c", "t2w"],
):
    setup_logger(cfg)
    seed_everything(cfg.SEED_EVERYTHING, workers=True)
    datamodule = DataModule(cfg)
    subjects = load_subjects(
        cfg.DATA.NIFTI.FOLDER_PATH,
        cfg.DATA.NIFTI.TRAIN_LABELS,
        modalities,
        cfg.DATA.NIFTI.TEST_FOLDER_PREFIX,
    )

    transforms = datamodule.get_transforms_patches(train=True)
    if mode == "pickle-subjects":
        save_subject_pickle(subjects, transforms, pickle_save_path)
    if mode == "pickle-dataset":
        save_dataset_pickle(subjects, transforms, pickle_save_path)
    elif mode == "nifti":
        save_subject_nifti(subjects, transforms, extension)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Preprocess the subjects and save outputs
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
        "--extension", default=".skull-crop.nii", type=str, help="Name of extension to save transformed image"
    )
    parser.add_argument(
        "--pickle-save-path",
        default="/home/bryan/data/brain_tumor/caidm_3d_240_pickle/preprocess1/dataset.pkl",
        type=str,
        help="Will save the pickled subjects here",
    )
    parser.add_argument(
        "--mode",
        choices=("pickle-subjects", "pickle-dataset", "nifti"),
        default="pickle-subjects",
        type=str,
        help="Mode for how to save the output. " "pickle saves the full dataset into a single pickle file",
    )
    parser.add_argument(
        "--modalities",
        default=["fla", "t1w", "t1c", "t2w"],
        type=str,
        nargs="+",
        help="List of MRI modalities to load",
    )
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
    sys.exit(main(cfg, args.extension, args.pickle_save_path, args.mode, args.modalities))
