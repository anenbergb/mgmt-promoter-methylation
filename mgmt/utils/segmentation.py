import copy

import numpy as np
import scipy


def make_label_mask(bool_mask: np.ndarray, gaussian_sigma: float = 0.0) -> tuple[np.ndarray, int]:
    """
    Constructs a mask with integer labels (1, 2, 3, etc)
    for each disjoint connected component in the bool mask.
    The largest object (by pixel volume) is 1. Next largested is 2.

    If gaussian_sigma > 0, then apply a guassian filter to the bool mask,
    which will filter out small objects. A greater gaussian_sigma will
    filter larger objects.
    """
    if gaussian_sigma > 0:
        bool_mask = scipy.ndimage.gaussian_filter(bool_mask, gaussian_sigma)
    # background has value = 0
    label_mask, num_labels = scipy.ndimage.label(bool_mask)
    lbl_val_to_count = {}
    for lbl_val in range(1, num_labels + 1):
        lbl_val_to_count[lbl_val] = (label_mask == lbl_val).sum()
    # sort the (lbl_val, lbl_count) in order by decreasing lbl_count
    lbl_val_sorted_by_count = sorted(lbl_val_to_count.items(), key=lambda x: x[1], reverse=True)
    lbl_reassignments = {x[0]: index + 1 for index, x in enumerate(lbl_val_sorted_by_count)}
    new_label_mask = copy.deepcopy(label_mask)
    for lbl_val, lbl_new_val in lbl_reassignments.items():
        if lbl_val != lbl_new_val:
            new_label_mask[label_mask == lbl_val] = lbl_new_val
    return new_label_mask, num_labels


def center_of_mass_per_label(label_mask: np.ndarray) -> list[tuple[float, ...]]:
    num_labels = label_mask.max()
    out = [scipy.ndimage.center_of_mass(label_mask == lbl_val) for lbl_val in range(1, num_labels + 1)]
    return out


def center_to_crop_slice(center: int | float, crop_width: int, width: int):
    crop_width_half = crop_width / 2

    # move the center point so that the crop does not go out of bounds
    max_x = width - crop_width_half
    min_x = crop_width_half
    center = max(center, min_x)
    center = min(center, max_x)

    crop_min = int(max(0, np.floor(center - crop_width_half)))
    crop_max = int(min(width, np.floor(center + crop_width_half)))
    return slice(crop_min, crop_max)


# TODO: try tight crop + fixed padding


def find_objects_fixed_crop(label_mask: np.ndarray, crop_dims: tuple[int, ...]) -> list[slice]:
    """
    Same output structure as scipy.ndimage.find_objects
    e.g.
    [(slice(0, 1, None),
    slice(56, 90, None),
    slice(33, 80, None),
    slice(17, 33, None)),
    (slice(0, 1, None),
    slice(22, 34, None),
    slice(56, 70, None),
    slice(19, 32, None))]
    """
    assert label_mask.ndim == len(crop_dims)
    centers_of_mass = center_of_mass_per_label(label_mask)

    def ndim_center_to_crop_slice(com: tuple[float, ...]):
        return tuple(
            [
                center_to_crop_slice(c, crop_width, width)
                for c, crop_width, width in zip(com, crop_dims, label_mask.shape)
            ]
        )

    objects = [ndim_center_to_crop_slice(com) for com in centers_of_mass]
    return objects
