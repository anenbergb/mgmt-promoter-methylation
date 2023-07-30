import numpy as np
import torch
import torchio as tio
from torchio.transforms.transform import TypeMaskingMethod


class SkullCropTransform(tio.SpatialTransform):
    r"""Apply a crop to the subject using the bounds of the skull + padding
    Args:
    padding: Tuple
        If only three values :math:`(w, h, d)` are provided, then
        :math:`w_{ini} = w_{fin} = w`,
        :math:`h_{ini} = h_{fin} = h` and
        :math:`d_{ini} = d_{fin} = d`.
    """

    def __init__(
        self,
        masking_method: TypeMaskingMethod = lambda x: x > 0.0,
        mask_image_name: str = "t1c",
        padding: tuple[int, int, int] = (0, 0, 0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.masking_method = masking_method
        self.mask_image_name = mask_image_name
        self.padding = padding
        if "include" in kwargs:
            self.include = ["tumor", *kwargs["include"]]
            kwargs.pop("include")
        self.kwargs = kwargs

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        assert self.mask_image_name in subject

        mask = self.get_mask_from_masking_method(
            self.masking_method,
            subject,
            subject[self.mask_image_name].data,
        )
        crop_bounds = mask_to_crop_bounds(mask, self.padding)
        subject = tio.Crop(cropping=crop_bounds, include=self.include, **self.kwargs).apply_transform(subject)
        return subject

    def is_invertible(self):
        return False


def find_bounds(mask: np.ndarray):
    """
    Find the bounds of the 3D bounding box that will enclose the object in a numpy array.
    :param mask: A boolean mask of an object stored in a numpy array of shape (240, 240, 155).
    :return: A tuple containing the minimum and maximum coordinates of the bounding box along each axis.
    """
    # Find the indices of the non-zero elements in the mask
    nz = np.nonzero(mask)

    # Find the minimum and maximum coordinates along each axis
    min_coords = np.min(nz, axis=1)
    max_coords = np.max(nz, axis=1)

    # Return the bounds as a tuple
    return tuple(zip(min_coords, max_coords))


def range_bounds_to_crop_bounds(
    range_bounds: list[tuple[int, int]],
    spatial_shape: tuple[int, int, int],
    padding: tuple[int, int, int],
):
    border_crop = []
    for (start, stop), spatial_dim, pad in zip(range_bounds, spatial_shape, padding):
        start = max(0, start - pad)
        border_crop.append(start)
        stop = max(0, spatial_dim - stop - pad)
        border_crop.append(stop)
    return border_crop


def mask_to_crop_bounds(mask: torch.Tensor, padding: tuple[int, int, int] = (0, 0, 0)):
    """
    Returns tuple :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`
            defining the number of values cropped from the edges of each axis.
    """
    mask_np = mask.numpy()[0]  # first dimension is channels
    spatial_shape = tuple(mask.shape[1:])
    range_bounds = find_bounds(mask_np)
    crop_bounds = range_bounds_to_crop_bounds(range_bounds, spatial_shape, padding)
    return crop_bounds
