import torch
import torchio as tio
from torchio import RescaleIntensity as _RescaleIntensity


class RescaleIntensity(_RescaleIntensity):
    """Rescale intensity values to a certain range.

    Equivalent to torchio.RescaleIntensity, but rescale each image
    in the subject independently.
    """

    def __init__(self, skull_mask = False, **kwargs):
        if skull_mask:
            kwargs["masking_method"] = lambda x: x > 0.0
        super().__init__(**kwargs)
        self.original_in_min_max = self.in_min_max
        # reset self.args_names because masking_method lambda can't be pickled
        self.args_names = []

    def apply_normalization(
        self,
        subject: tio.Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        if self.original_in_min_max is None:
            self.in_min_max = None
            self.in_min = None
            self.in_max = None
        image = subject[image_name]
        image.set_data(self.rescale(image.data, mask, image_name))
