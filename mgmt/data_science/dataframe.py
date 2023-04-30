import numpy as np
import pandas as pd
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
    tumor_df = add_center_of_mass(tumor_df, data["tum"])

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
            slice_center, h_center, w_center = ndimage.center_of_mass(
                data_tumor[patient_i, ..., 0]
            )
            slice_centers.append(slice_center)
            h_centers.append(h_center)
            w_centers.append(w_center)
    df["tumor_center_of_mass_slice"] = slice_centers
    df["tumor_center_of_mass_H"] = h_centers
    df["tumor_center_of_mass_W"] = w_centers
    return df
