import argparse
import copy
import os
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


def main(cfg: CfgNode, extension):
    setup_logger(cfg)
    seed_everything(cfg.SEED_EVERYTHING, workers=True)
    datamodule = DataModule(cfg)
    subjects = load_subjects(
        cfg.DATA.NIFTI.FOLDER_PATH,
        cfg.DATA.NIFTI.TRAIN_LABELS,
        ["fla", "t1w", "t1c", "t2w"],
        cfg.DATA.NIFTI.TEST_FOLDER_PREFIX,
    )

    transforms = datamodule.get_transforms_patches()
    for subject in tqdm(subjects):
        trans_subject = transforms(subject)
        save_transformed_subject(trans_subject, extension)


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
    sys.exit(main(cfg, args.extension))
