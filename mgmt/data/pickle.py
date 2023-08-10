import os
import pickle
from glob import glob

import torchio as tio
from tqdm import tqdm


def load_subject_pickles(folder_path: str, filter_file_prefix: str | None = "P-") -> list[tio.Subject]:
    subject_pickles = glob(os.path.join(folder_path, "*.pkl"))
    subjects = []
    for subject_pickle in tqdm(subject_pickles, "loading subjects from pickles"):
        basename = os.path.basename(subject_pickle)
        if filter_file_prefix is not None and not basename.startswith(filter_file_prefix):
            continue
        with open(subject_pickle, "rb") as f:
            subject = pickle.load(f)
        subjects.append(subject)
    return subjects
