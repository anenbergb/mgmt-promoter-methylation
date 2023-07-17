import argparse
import os
import sys
from glob import glob

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from monai.visualize import blend_images
from PIL import Image
from tqdm import tqdm

from mgmt.data.constants import MODALITIES, TUMOR_LABELS_LONG
from mgmt.data.nifti import load_subjects
from mgmt.data_science.dataframe import tumor_dataframe
from mgmt.utils.crop import slide_box_within_border
from mgmt.utils.ffmpeg import FfmpegWriter
from mgmt.visualize.subject import plot_volume_with_label
from mgmt.visualize.visualize import segmentation_label_color

modality2name = {
    "fla": "FLAIR",  # "Fluid Attenuated\nInversion Recovery\n(FLAIR)",
    "t1w": "T1-Weighted Pre-contrast\n(T1w)",
    "t1c": "T1-Weighted Post-contrast\n(T1Gd)",
    "t2w": "T2-Weighted (T2)",
}


def plot_subject_grid_all_modalities(
    subjects,
    num_patients=None,
    patient_range=None,
    filename=None,
    dpi=100,
    size_factor=5,
):
    ncols = 4
    num_patients = len(subjects) if num_patients is None else num_patients
    if patient_range is None:
        patient_range = range(num_patients)
    else:
        num_patients = len(patient_range)

    nrows = num_patients
    fig, axes = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        figsize=(ncols * size_factor, nrows * size_factor),
        dpi=dpi,
    )

    fig.suptitle(
        "MRI with tumor segmentation",
        fontsize=24,
    )
    for row, patient_i in enumerate(patient_range):
        subject = subjects[patient_i]
        color = "green" if subject.category == "methylated" else "red"
        # these are test subjects
        color = None if subject.train_test_split == "test" else color
        for col, modality in enumerate(MODALITIES):
            ax = axes[row, col]
            plot_volume_with_label(
                image=subject[modality],
                label=subject.tumor,
                axes=(ax,),
                single_axis="axial",
                show=False,
                border_color=color,
                cmap="hsv",
            )
            ax.set_title("")
            ax.set_xlabel(f"{subject.patient_id_str}", fontsize=20)
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax = axes[row, col]
                modname = modality2name[modality]
                ax.set_title(modname, fontsize=22)

    patches = [
        mpatches.Patch(color="green", label="Methylated"),
        mpatches.Patch(color="red", label="Not Methylated"),
    ]
    mgmt_legend = fig.legend(
        handles=patches,
        loc="upper right",
        fontsize=20,
        title="MGMT Promoter Methylation",
        title_fontsize=20,
    )

    tumor_colors = segmentation_label_color([0, 1, 2, 4], "hsv", rgb_255=False)
    patches = [mpatches.Patch(color=c, label=TUMOR_LABELS_LONG[i]) for i, c in zip([1, 2, 4], tumor_colors[1:])]
    tumor_legend = fig.legend(
        handles=patches,
        loc="upper left",
        fontsize=20,
        title="Tumor Segmentation Labels",
        title_fontsize=20,
    )
    fig.add_artist(mgmt_legend)
    fig.add_artist(tumor_legend)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="png", dpi=dpi)
    return fig


def main_render_dataset_video(args):
    logger.info(f"Loading dataset from {args.data}")
    subjects = load_subjects(args.data, args.labels)
    num_patients = len(subjects)
    logger.info(f"Found {num_patients} patients")

    if args.num_patients is not None:
        num_patients = args.num_patients
    num_patients_per_image = 5
    size_factor = 5

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    image_height_pix = args.dpi * size_factor * num_patients_per_image
    image_width_pix = args.dpi * size_factor * 4
    video_writer = FfmpegWriter(
        args.output,
        image_width_pix,
        image_height_pix,
        framerate=args.framerate,
        vcodec="libx264",
        crf=20,
    )
    logger.info(f"Saving video to {args.output}")

    for pi in tqdm(range(0, num_patients, num_patients_per_image)):
        patient_range = list(range(pi, pi + num_patients_per_image))

        fig = plot_subject_grid_all_modalities(
            subjects,
            patient_range=patient_range,
            dpi=args.dpi,
            size_factor=size_factor,
        )
        if video_writer is not None:
            video_writer.write(fig)
        plt.close(fig)
    if video_writer is not None:
        video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render MRI images across all modalities with the tumor highlighted",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="/home/bryan/data/brain_tumor/caidm_3d_240",
        help="Path to directory of patient folders. Each folder containing "
        "fla.nii.gz  seg.nii.gz  t1c.nii.gz  t1w.nii.gz  t2w.nii.gz",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        default="/home/bryan/data/brain_tumor/classification/train_labels.csv",
        help="Path to MGMT promoter methylation status classification",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/home/bryan/gdrive/Radiology-Research/brain_tumor/caidm-240-data-vis/brain-tumor.mp4",
        help="Path to output .mp4 file",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="matplotlib figure dpi",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-patients",
        type=int,
    )
    parser.add_argument(
        "--draw-center-mass-box",
        action="store_true",
        help="Whether to draw a yellow box enscribing the tumor at the center of mass frame",
    )
    parser.add_argument(
        "--crop-box-width",
        type=int,
        help="If provided, will draw a green box centered at the center of (x,y) location. "
        "This box is useful to visualize the cropped image.",
    )
    args = parser.parse_args()
    sys.exit(main_render_dataset_video(args))
