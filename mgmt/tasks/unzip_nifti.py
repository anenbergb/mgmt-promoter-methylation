import argparse
import copy
import os
import sys
from collections import OrderedDict
from glob import glob

import nibabel as nib
from loguru import logger
from tqdm import tqdm

from mgmt.data.nifti import get_subject_folders


def unzip_subject_and_save(subject_folder: str):
    gz_files = glob(f"{subject_folder}/*.gz")
    for gz_file in gz_files:
        image = nib.load(gz_file)
        without_suffix = gz_file.removesuffix(".gz")
        nib.save(image, without_suffix)


def unzip_nifti(data_root_directory: str):
    logger.info(f"Searching for .nii.gz images in {data_root_directory}")
    subject_folders = get_subject_folders(data_root_directory)
    logger.info(f"Found {len(subject_folders)} folders with .nii.gz files")
    for subject_folder in tqdm(subject_folders):
        unzip_subject_and_save(subject_folder)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Unzip the .nii.gz files to speed up loading times.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="/home/bryan/data/brain_tumor/caidm_3d_240",
        type=str,
        help="Path to root of directory containing .nii.gz files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(unzip_nifti(args.data))
