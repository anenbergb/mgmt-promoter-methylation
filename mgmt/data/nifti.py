import csv
import os

import torchio as tio

from mgmt.data.constants import MODALITIES, MODALITY2NAME


def make_subject(folder_path, category_id=0) -> tio.Subject:
    category = "methylated" if category_id == 1 else "unmethylated"
    mri_images = {}
    for modality, name in MODALITY2NAME.items():
        file_path = os.path.join(folder_path, f"{modality}.nii.gz")
        mri_images[modality] = tio.ScalarImage(path=file_path, name=name)
    tumor = tio.LabelMap(path=os.path.join(folder_path, "seg.nii.gz"), name="Tumor Segmentation")
    # patient_id_str is formatted P-00000 or MGMT-025579
    patient_id_str = os.path.basename(folder_path)
    prefix, patient_id = patient_id_str.split("-")
    train_test_split = "test" if prefix == "MGMT" else "train"
    subject = tio.Subject(
        tumor=tumor,
        category_id=category_id,
        category=category,
        patient_id=patient_id,
        patient_id_str=patient_id_str,
        train_test_split=train_test_split,
        **mri_images,
    )
    return subject


def load_label_csv(csv_path: str = "/home/bryan/data/brain_tumor/classification/train_labels.csv"):
    with open(csv_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        rows = [row for row in csv_reader]
        # first row is just headers: BraTS21ID,MGMT_value
        label_map = {str(row[0]): int(row[1]) for row in rows[1:]}
    return label_map


def get_subject_folders(dataset_folder):
    patient_folders = []
    for file_name in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file_name)
        if os.path.isdir(file_path):
            patient_files = os.listdir(file_path)
            if (
                "fla.nii.gz" in patient_files
                and "seg.nii.gz" in patient_files
                and "t1c.nii.gz" in patient_files
                and "t1w.nii.gz" in patient_files
                and "t2w.nii.gz" in patient_files
            ):
                patient_folders.append(file_path)
        os.path.join(dataset_folder)
    return patient_folders


def load_subjects(dataset_folder, label_csv_path):
    label_map = load_label_csv(label_csv_path)
    subject_folders = get_subject_folders(dataset_folder)
    subjects = []
    for subject_folder in subject_folders:
        # folder_name should be formatted as P-00000 or MGMT-025579
        folder_name = os.path.basename(subject_folder)
        folder_id = folder_name.split("-")[-1]
        category_id = label_map.get(folder_id, 0)
        subjects.append(make_subject(subject_folder, category_id))
    return subjects
