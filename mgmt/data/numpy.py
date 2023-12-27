# Utilities for loading subjects from a numpy npz

import numpy as np
import torchio as tio

from mgmt.data.constants import MODALITIES, MODALITY2NAME


def load_data(file_name: str) -> dict[str, np.ndarray]:
    data = np.load(file_name)
    return {k: v for k, v in data.items()}


def make_scalar_image(data: dict[str, np.ndarray], patient_index: int = 0, modality: str = "t2w") -> tio.ScalarImage:
    # (D,H,W,C) where D=48, C=1
    assert modality in data
    tensor = data[modality][patient_index]
    # convert to (C,W,H,D)
    tensor = np.transpose(tensor, axes=(3, 2, 1, 0))
    name = MODALITY2NAME.get(modality)
    return tio.ScalarImage(tensor=tensor, name=name)


def make_segmentation_image(
    data: dict[str, np.ndarray],
    patient_index: int = 0,
) -> tio.LabelMap:
    # (D,H,W,C) where D=48, C=1
    tensor = data["tum"][patient_index]
    # convert to (C,W,H,D)
    tensor = np.transpose(tensor, axes=(3, 2, 1, 0))
    return tio.LabelMap(tensor=tensor, name="Tumor Segmentation")


def make_subject(data: dict[str, np.ndarray], patient_index: int = 0) -> tio.Subject:
    category_id = data["lbl"][patient_index].item()
    category = "methylated" if category_id == 1 else "unmethylated"
    mri_images = {modality: make_scalar_image(data, patient_index, modality) for modality in MODALITIES}
    subject = tio.Subject(
        tumor=make_segmentation_image(data, patient_index),
        category_id=category_id,
        category=category,
        patient_id=patient_index,
        **mri_images,
    )
    return subject


def load_subjects(filepath_npz: str) -> list[tio.Subject]:
    data = load_data(filepath_npz)
    subjects = [make_subject(data, i) for i in range(len(data))]
    return subjects
