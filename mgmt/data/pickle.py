import os
import pickle
from glob import glob

import torchio as tio
from tqdm import tqdm
from mgmt.data.cacheable_subject import CacheableSubject


def load_subject_pickles(
    folder_path: str,
    filter_file_prefix: str | None = "P-",
    remove_paths: bool = True,
    cache_dir: str = "",
) -> list[tio.Subject]:
    subject_pickles = glob(os.path.join(folder_path, "*.pkl"))
    subjects = []
    for subject_pickle in tqdm(subject_pickles[:50], "loading subjects from pickles"):
        basename = os.path.basename(subject_pickle)
        if filter_file_prefix is not None and not basename.startswith(filter_file_prefix):
            continue
        with open(subject_pickle, "rb") as f:
            subject = pickle.load(f)
        subject = subject_remove_paths(subject)
        cache_subject = CacheableSubject.from_subject(subject, cache_dir=cache_dir)
        cache_subject.cache()
        subjects.append(cache_subject)
    return subjects


def subject_remove_paths(subject: tio.Subject) -> tio.Subject:
    for image in subject.get_images(intensity_only=False):
        image.path = None
    return subject


def count_subject_pickles(folder_path: str, filter_file_prefix: str | None = "P-", **kwargs) -> int:
    subject_pickles = glob(os.path.join(folder_path, "*.pkl"))
    subjects = []
    counter = 0
    for subject_pickle in subject_pickles[:50]:
        basename = os.path.basename(subject_pickle)
        if filter_file_prefix is not None and not basename.startswith(filter_file_prefix):
            continue
        counter += 1
    return counter
