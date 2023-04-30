import argparse
import os
import sys
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from PIL import Image

from monai.visualize import blend_images
from mgmt.data_science.dataframe import tumor_dataframe
from mgmt.utils.ffmpeg import FfmpegWriter

modality2name = {
    "fla": "FLAIR",  # "Fluid Attenuated\nInversion Recovery\n(FLAIR)",
    "t1w": "T1-Weighted Pre-contrast\n(T1w)",
    "t1c": "T1-Weighted Post-contrast\n(T1Gd)",
    "t2w": "T2-Weighted (T2)",
}


def convert_to_255(image):
    return np.array(image * 255.0, dtype=np.uint8)


def add_color_border(img, border_width=10, color="green"):
    """
    Assume img is shape (H,W,C)
    """
    assert isinstance(img, np.ndarray)
    height = img.shape[0]
    width = img.shape[1]
    frame_height = 2 * border_width + height
    frame_width = 2 * border_width + width
    framed_img = Image.new("RGB", (frame_height, frame_width), color)
    framed_img = np.array(framed_img)
    framed_img[border_width:-border_width, border_width:-border_width] = img
    return framed_img


def plot_center_of_mass(
    ax, df, data, patient_i, modality="t2w", xlabel=False, border_width=4
):
    row = df.iloc[patient_i]
    s = int(np.round(row["tumor_center_of_mass_slice"]))
    h = row["tumor_center_of_mass_H"]
    w = row["tumor_center_of_mass_W"]
    methylation = row["methylation"]

    image = blend_images(
        # convert from BDHWC to HWC CHW where C=1, and D=48 3D depth channels
        image=np.transpose(data[modality][patient_i, s, ...], axes=(2, 0, 1)),
        label=np.transpose(data["tum"][patient_i, s, ...], axes=(2, 0, 1)),
        alpha=0.25,
        cmap="hsv",
        rescale_arrays=True,
    )
    image = np.moveaxis(image, 0, -1)  # CHW -> HWC
    image255 = convert_to_255(image)
    color = "green" if methylation else "red"
    image_frame = add_color_border(image255, border_width=border_width, color=color)
    ax.scatter([w + border_width], [h + border_width], marker="o", s=50, c="yellow")
    ax.imshow(image_frame)
    title = f"Patient {patient_i} Slice {s}"
    if xlabel:
        ax.set_xlabel(title, fontsize=20)
    else:
        ax.set_title(title, fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_center_of_mass_grid_one_modality(
    df,
    data,
    modality="t2w",
    num_patients=None,
    patient_range=None,
    filename=None,
):
    ncols = 3
    num_patients = len(df)
    num_patients = len(df) if num_patients is None else num_patients
    if patient_range is None:
        patient_range = range(num_patients)
    else:
        num_patients = len(patient_range)

    nrows = int(np.ceil(num_patients / ncols))
    size_factor = 5
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False, figsize=(ncols * size_factor, nrows * size_factor)
    )

    modname = modality2name[modality]
    fig.suptitle(
        f"{modname} MRI with tumor segmentation and center point\n"
        "Slice selected by center of mass",
        fontsize=20,
    )
    for patient_i in patient_range:
        row = int(patient_i / ncols)
        col = patient_i % ncols
        plot_center_of_mass(axes[row, col], df, data, patient_i, modality)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="png")
    return fig


def plot_center_of_mass_grid_all_modalities(
    df,
    data,
    num_patients=None,
    patient_range=None,
    filename=None,
    dpi=100,
    size_factor=5,
):
    ncols = 4
    num_patients = len(df)
    num_patients = len(df) if num_patients is None else num_patients
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
        "MRI with tumor segmentation and center point\n"
        "Slice selected by center of mass",
        fontsize=24,
    )
    for row, patient_i in enumerate(patient_range):
        for col, modality in enumerate(modality2name.keys()):
            plot_center_of_mass(
                axes[row, col], df, data, patient_i, modality, xlabel=True
            )
            if row == 0:
                ax = axes[row, col]
                modname = modality2name[modality]
                ax.set_title(modname, fontsize=22)

    patches = [
        mpatches.Patch(color="green", label="Methylated"),
        mpatches.Patch(color="red", label="Not Methylated"),
    ]
    fig.legend(
        handles=patches,
        loc="upper right",
        fontsize=22,
        title="MGMT Promoter Methylation",
        title_fontsize=22,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="png", dpi=dpi)
    return fig


def plot_center_of_mass_patient(df, data, modality="t2w", patient=0, filename=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    modname = modality2name[modality]
    fig.suptitle(f"{modname} MRI", fontsize=14)
    plot_center_of_mass(ax, df, data, patient, modality)
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="png")
    return fig


def load_data(data_npz):
    data = np.load(data_npz)
    return {k: v for k, v in data.items()}


def make_filename(output_dir, patient_range):
    p_first = patient_range[0]
    p_last = patient_range[-1]
    file = f"cm-slice-patient-{p_first:04d}-{p_last:04d}.png"
    filename = os.path.join(output_dir, file)
    return filename


def main_plot_center_mass(args):
    logger.info(f"Loading dataset from {args.data}")
    data = load_data(args.data)
    logger.info("Successfully loaded dataset")
    tumor_df = tumor_dataframe(data)
    logger.info("Built tumor dataframe")

    num_patients = len(tumor_df)
    if args.num_patients is not None:
        num_patients = args.num_patients
    num_patients_per_image = 5
    size_factor = 5

    video_writer = None
    if args.output.endswith(".mp4"):
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
    else:
        logger.info(f"Saving images to {args.output}")

    for pi in tqdm(range(0, num_patients, num_patients_per_image)):
        patient_range = list(range(pi, pi + num_patients_per_image))
        filename = (
            make_filename(args.output, patient_range) if video_writer is None else None
        )
        fig = plot_center_of_mass_grid_all_modalities(
            tumor_df,
            data,
            patient_range=patient_range,
            filename=filename,
            dpi=args.dpi,
            size_factor=size_factor,
        )
        if video_writer is not None:
            video_writer.write(fig)
    if video_writer is not None:
        video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render MRI images across all modalities with the tumor highlighted "
        "in the slice that aligns with the tumor center of mass",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="/home/bryan/data/brain_tumor/caidm_3d_96/data.npz",
        help="Path to data.npz",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/home/bryan/gdrive/Radiology-Research/brain_tumor/tumor-statistic-report/tumor-center-mass.mp4",
        help="Path to output directory",
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
    args = parser.parse_args()
    sys.exit(main_plot_center_mass(args))
