from typing import Union

import torch
import torchio as tio


def get_min_padding(
    shape: tuple[int, int, int] = (64, 64, 64),
    min_shape: tuple[int, int, int] = (64, 64, 64),
):
    padding = []
    for m, sub in zip(min_shape, shape):
        pad = [0, 0]
        if sub <= m:
            diff = m - sub
            pad = [diff // 2, diff // 2 + diff % 2]
        padding.extend(pad)
    return padding


class PadToMinShape(tio.SpatialTransform):
    r"""
    Args:
    min_shape: Tuple
        Minimum shape of the output subject. :math:`W \times H \times D`.
    """

    def __init__(
        self,
        min_shape: tuple[int, int, int] = (64, 64, 64),
        padding_mode: Union[str, float] = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_shape = min_shape
        tio.Pad.check_padding_mode(padding_mode)
        self.padding_mode = padding_mode
        if "include" in kwargs:
            self.include = ["tumor", *kwargs["include"]]
            kwargs.pop("include")
        self.kwargs = kwargs
        self.args_names = ["min_shape", "padding_mode"]

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        padding = get_min_padding(subject.spatial_shape, self.min_shape)
        subject = tio.Pad(padding=padding, include=self.include, **self.kwargs).apply_transform(subject)
        return subject

    def is_invertible(self):
        return False
