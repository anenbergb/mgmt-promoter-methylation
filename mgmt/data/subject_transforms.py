import functools
from typing import Tuple

import scipy
import torchio
from torchio import Subject
from torchio.transforms import Crop, Pad, SpatialTransform
from torchio.transforms.transform import TypeSixBounds

from mgmt.utils.crop import slide_box_within_border
from mgmt.utils.segmentation import find_objects_fixed_crop, make_label_mask


def center_of_mass(subject: Subject) -> Tuple[float, float, float]:
    assert "tumor" in subject
    _, x, y, d = scipy.ndimage.center_of_mass(subject["tumor"].numpy())
    return (x, y, d)


def object_slice_to_crop_bounds(object_slice: tuple[slice, slice, slice], spatial_shape: tuple[int, int, int]):
    """
    Converts the object per-dim slice to a crop bounds tuple compatible with
    https://torchio.readthedocs.io/transforms/preprocessing.html#crop

    Returns tuple :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`
        defining the number of values cropped from the edges of each axis.
    """
    border_crop = []
    for slc, spatial_dim in zip(object_slice, spatial_shape):
        border_crop.append(slc.start)
        border_crop.append(spatial_dim - slc.stop)
    return tuple(border_crop)


def largest_tumor_crop_bounds(subject: Subject, crop_dim: tuple[int, int, int] | None = None) -> TypeSixBounds:
    assert "tumor" in subject
    tumor = subject.tumor.numpy()
    assert tumor.shape[0] == 1
    label_mask, num_labels = make_label_mask(tumor[0])
    assert num_labels > 0
    if crop_dim is None:
        objects = scipy.ndimage.find_objects(label_mask)
    else:
        objects = find_objects_fixed_crop(label_mask, crop_dim)

    largest_object = objects[0]
    crop_bounds = object_slice_to_crop_bounds(largest_object, subject.spatial_shape)
    return crop_bounds


def object_slice_to_dimensions(object_slice: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    dimensions = []
    for slc in object_slice:
        dimensions.append(slc.stop - slc.start)
    return tuple(dimensions)


def tumor_crop_dimensions(subject: Subject) -> list[tuple[int, int, int]]:
    assert "tumor" in subject
    tumor = subject.tumor.numpy()
    assert tumor.shape[0] == 1
    label_mask, num_labels = make_label_mask(tumor[0])
    assert num_labels > 0
    objects = scipy.ndimage.find_objects(label_mask)
    object_dims = [object_slice_to_dimensions(o) for o in objects]
    return object_dims


class CropLargestTumor(SpatialTransform):
    def __init__(self, crop_dim: tuple[int, int, int] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.crop_dim = crop_dim
        self.args_names = ["crop_dim"]
        if "include" in kwargs:
            self.include = ["tumor", *kwargs["include"]]
            kwargs.pop("include")
        self.kwargs = kwargs

    def apply_transform(self, subject: Subject) -> Subject:
        crop_bounds = largest_tumor_crop_bounds(subject, self.crop_dim)
        subject = Crop(cropping=crop_bounds, include=self.include, **self.kwargs).apply_transform(subject)
        return subject

    def is_invertible(self):
        return False


# TODO: If it's necessary to compute inverse of CropLargestTumor, then store
# the patient_id_crop_map in a GLOBAL variable and make that available to
# PadGivenMap
# class PadGivenMap(SpatialTransform):
#     def __init__(
#         self,
#         patient_id_crop_map: dict[int, TypeSixBounds],
#         crop_dim: tuple[int, int, int] | None = None,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.patient_id_crop_map = patient_id_crop_map
#         self.crop_dim = crop_dim
#         self.kwargs = kwargs
#         self.args_names = ["patient_id_crop_map", "crop_dim"]

#     def apply_transform(self, subject: Subject) -> Subject:
#         crop_bounds = self.patient_id_crop_map[subject.patient_id]
#         subject = Pad(crop_bounds, **self.kwargs).apply_transform(subject)
#         return subject

#     def is_invertible(self):
#         return True

#     def inverse(self):
#         return CropLargestTumor(self.crop_dim)
