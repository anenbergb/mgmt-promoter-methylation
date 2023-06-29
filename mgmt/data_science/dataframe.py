from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
from monai.apps.detection.transforms.box_ops import convert_mask_to_box
from scipy import ndimage


def tumor_dataframe(data):
    """
    data is dictionary of tensors

    key | value
    -----------
    t2w float32
    fla float32
    t1w float32
    t1c float32
    tum uint8
    lbl int64

    data["tum"].shape
    (565, 48, 96, 96, 1)
    N, D, H, W, C
        - N(0) : patients
        - D(1) : depth. # of slices
        - H(2): height
        - W(3) : width
        - C(4) : channels per slice
    """
    tumor_df = pd.DataFrame(
        {
            "patient": np.arange(len(data["lbl"])),
            "methylation": data["lbl"].flatten().astype(bool),
        }
    )
    # number of tumor pixels in 3d volume
    tumor_df["tumor_volume"] = np.count_nonzero(data["tum"], axis=(1, 2, 3, 4))
    # total pixels in the 3d volume
    total_pixels = np.prod(data["tum"].shape[1:])
    tumor_df["tumor_volume_rel_%"] = 100 * tumor_df["tumor_volume"] / total_pixels

    tumor_area = np.count_nonzero(data["tum"], axis=(2, 3, 4))  # yeilds NxD tensor
    # max slice 2D area per patient
    tumor_df["tumor_area_max"] = np.max(tumor_area, axis=1)
    tumor_df["tumor_area_max_slice"] = np.argmax(tumor_area, axis=1)
    logger.info("tumor_dataframe: adding center of mass")
    tumor_df = add_center_of_mass(tumor_df, data["tum"])
    logger.info("tumr_dataframe: adding tumor quadrant")
    tumor_df = add_tumor_quadrant(tumor_df)
    logger.info("tumor_dataframe: adding pixel statistics")
    tumor_df = add_pixel_stats(tumor_df, data)
    logger.info("tumor_dataframe: adding center of mass box")
    tumor_df = add_center_of_mass_box(tumor_df, data)
    return tumor_df


def add_center_of_mass(df, data_tumor):
    """
    data_tumor.shape
    (565, 48, 96, 96, 1)
    N, D, H, W, C
        - N(0) : patients
        - D(1) : depth. # of slices
        - H(2): height
        - W(3) : width
        - C(4) : channels per slice
    """

    slice_centers = []
    h_centers = []
    w_centers = []
    for patient_i in range(len(data_tumor)):
        if df["tumor_volume"][patient_i] == 0:
            slice_centers.append(0)
            h_centers.append(0)
            w_centers.append(0)
        else:
            slice_center, h_center, w_center = ndimage.center_of_mass(data_tumor[patient_i, ..., 0])
            slice_centers.append(slice_center)
            h_centers.append(h_center)
            w_centers.append(w_center)
    df["tumor_center_of_mass_slice"] = slice_centers
    df["tumor_center_of_mass_H"] = h_centers
    df["tumor_center_of_mass_W"] = w_centers
    return df


def add_tumor_quadrant(df, width=96):
    """
        II  |  I
        ---------
        III | IV

    Image origin (0,0) is in top-left corner
    """
    quadrants = []
    for i in range(len(df)):
        h_center = df["tumor_center_of_mass_H"][i]
        w_center = df["tumor_center_of_mass_W"][i]
        if h_center < width / 2 and w_center < width / 2:
            quadrants.append(2)
        elif h_center < width / 2 and w_center >= width / 2:
            quadrants.append(1)
        elif h_center >= width / 2 and w_center < width / 2:
            quadrants.append(3)
        elif h_center >= width / 2 and w_center >= width / 2:
            quadrants.append(4)
        else:
            print(i, h_center, w_center)
    df["tumor_quadrant"] = quadrants
    return df


def mask_to_bbox(mask):
    bboxes, bbox_classifications = convert_mask_to_box(mask, bg_label=0)
    # bboxes are y1,x1,y2,x2
    # convert to x1,y1,x2,y2
    bboxes_xy = np.zeros_like(bboxes)
    bboxes_xy[:, 0] = bboxes[:, 1]
    bboxes_xy[:, 1] = bboxes[:, 0]
    bboxes_xy[:, 2] = bboxes[:, 3]
    bboxes_xy[:, 3] = bboxes[:, 2]
    bboxes_xy = bboxes_xy.astype(int)
    if len(bboxes_xy) == 0:
        return np.array([0, 0, 0, 0], dtype=int)
    return bboxes_xy[0]  # single box


def add_center_of_mass_box(df, data):
    cols = defaultdict(list)
    for patient_i in range(data["tum"].shape[0]):
        slice_i = int(np.round(df.tumor_center_of_mass_slice[patient_i]))
        tum = data["tum"][patient_i, slice_i, ..., 0][None, :]
        bbox = mask_to_bbox(tum)
        cols["tumor_center_of_mass_x_min"].append(bbox[0])
        cols["tumor_center_of_mass_y_min"].append(bbox[1])
        cols["tumor_center_of_mass_x_max"].append(bbox[2])
        cols["tumor_center_of_mass_y_max"].append(bbox[3])
        cols["tumor_center_of_mass_width"].append(bbox[2] - bbox[0])
        cols["tumor_center_of_mass_height"].append(bbox[3] - bbox[1])

    for k, v in cols.items():
        df[k] = v
    return df


def compute_pixel_stats(pixels):
    is_empty = len(pixels) == 0
    return {
        "mean": None if is_empty else np.average(pixels),
        "median": None if is_empty else np.median(pixels),
        "min": None if is_empty else np.min(pixels),
        "max": None if is_empty else np.max(pixels),
    }


def add_pixel_stats(df, data):
    """
    data_tumor.shape
    (565, 48, 96, 96, 1)
    N, D, H, W, C
        - N(0) : patients
        - D(1) : depth. # of slices
        - H(2): height
        - W(3) : width
        - C(4) : channels per slice
    """
    cols_to_add = defaultdict(list)
    for modality in ("t2w", "fla", "t1w", "t1c"):
        for patient_i in range(data["tum"].shape[0]):
            mri = data[modality][patient_i, ..., 0]
            tumor_mask = data["tum"][patient_i, ..., 0].astype(bool)
            brain_mask = mri != mri.min()
            brain_non_tumor_mask = brain_mask & (~tumor_mask)

            # mirror image of tumor mask about vertical axis
            flip_tumor_mask = tumor_mask[..., ::-1]

            tumor_pixels = mri[tumor_mask]
            non_tumor_pixels = mri[brain_non_tumor_mask]
            flip_tumor_pixels = mri[flip_tumor_mask]

            for k, v in compute_pixel_stats(tumor_pixels).items():
                cols_to_add[f"{modality}_tumor_{k}"].append(v)
            for k, v in compute_pixel_stats(non_tumor_pixels).items():
                cols_to_add[f"{modality}_non_tumor_{k}"].append(v)
            for k, v in compute_pixel_stats(flip_tumor_pixels).items():
                cols_to_add[f"{modality}_flip_tumor_{k}"].append(v)

    for k, v in cols_to_add.items():
        df[k] = v
    return df


def add_pixel_stats_per_slice(dfs, data):
    """
    data_tumor.shape
    (565, 48, 96, 96, 1)
    N, D, H, W, C
        - N(0) : patients
        - D(1) : depth. # of slices
        - H(2): height
        - W(3) : width
        - C(4) : channels per slice
    """
    for modality in ("t2w", "fla", "t1w", "t1c"):
        for patient_i in range(data["tum"].shape[0]):
            mri = data[modality][patient_i, ..., 0]
            tumor_mask = data["tum"][patient_i, ..., 0].astype(bool)
            brain_mask = mri != mri.min()
            brain_non_tumor_mask = brain_mask & (~tumor_mask)

            # mirror image of tumor mask about vertical axis
            flip_tumor_mask = tumor_mask[..., ::-1]

            cols_to_add = defaultdict(list)

            for slice_i in range(data["tum"].shape[1]):
                slice_mri = mri[slice_i]
                slice_tumor_mask = tumor_mask[slice_i]
                slice_non_tumor_mask = brain_non_tumor_mask[slice_i]
                slice_flip_tumor_mask = flip_tumor_mask[slice_i]

                slice_tumor_pixels = slice_mri[slice_tumor_mask]
                slice_non_tumor_pixels = slice_mri[slice_non_tumor_mask]
                slice_flip_tumor_pixels = slice_mri[slice_flip_tumor_mask]

                for k, v in compute_pixel_stats(slice_tumor_pixels).items():
                    cols_to_add[f"{modality}_tumor_{k}"].append(v)
                for k, v in compute_pixel_stats(slice_non_tumor_pixels).items():
                    cols_to_add[f"{modality}_non_tumor_{k}"].append(v)
                for k, v in compute_pixel_stats(slice_flip_tumor_pixels).items():
                    cols_to_add[f"{modality}_flip_tumor_{k}"].append(v)

            for k, v in cols_to_add.items():
                dfs[patient_i][k] = v
    return dfs


def add_tumor_bbox_per_slice(dfs, data):
    tumor_area = np.count_nonzero(data["tum"], axis=(2, 3, 4))
    for patient_i in range(data["tum"].shape[0]):
        cols_to_add = defaultdict(list)
        for slice_i in range(data["tum"].shape[1]):
            tum = data["tum"][patient_i, slice_i, ..., 0]
            if tumor_area[patient_i, slice_i] == 0:
                x_min, x_max, y_min, y_max = (0, 0, 0, 0)
            else:
                x_min, x_max = np.flatnonzero(np.max(tum, axis=0))[[0, -1]]
                y_min, y_max = np.flatnonzero(np.max(tum, axis=1))[[0, -1]]
            h = y_max - y_min
            w = x_max - x_min

            cols_to_add["tumor_area"].append(tumor_area[patient_i, slice_i])
            cols_to_add["tumor_x_min"].append(x_min)
            cols_to_add["tumor_x_max"].append(x_max)
            cols_to_add["tumor_y_min"].append(y_min)
            cols_to_add["tumor_y_max"].append(y_max)
            cols_to_add["tumor_height"].append(h)
            cols_to_add["tumor_width"].append(w)

        for k, v in cols_to_add.items():
            dfs[patient_i][k] = v
    return dfs


def tumor_slice_dataframe(data):
    """
    data is dictionary of tensors

    key | value
    -----------
    t2w float32
    fla float32
    t1w float32
    t1c float32
    tum uint8
    lbl int64

    data["tum"].shape
    (565, 48, 96, 96, 1)
    N, D, H, W, C
        - N(0) : patients
        - D(1) : depth. # of slices
        - H(2): height
        - W(3) : width
        - C(4) : channels per slice
    """
    dfs = {}
    for patient_i in range(data["tum"].shape[0]):
        dfs[patient_i] = pd.DataFrame(
            {
                "slice": np.arange(data["tum"].shape[1]),
            }
        )
    dfs = add_tumor_bbox_per_slice(dfs, data)
    dfs = add_pixel_stats_per_slice(dfs, data)
    return dfs
