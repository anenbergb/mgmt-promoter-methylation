import torch
import torchio as tio
from torchio.transforms import Transform


def segmentation_to_probability_map(seg_mask, patch_size=[64, 64, 64], device=torch.device("cuda")):
    pz = patch_size
    seg_mask = (seg_mask > 0).type(torch.float32)
    seg_mask = seg_mask[None, ...].to(device=device)
    weight = torch.ones([1, 1, *patch_size], dtype=torch.float32, device=device)
    conv = torch.nn.functional.conv3d(
        seg_mask,
        weight,
        bias=None,
    )
    log = torch.log(conv)
    log[log < 0] = 0.0

    # padding is applied to the dimensions in the reverse order
    padding = []
    for pz in patch_size[::-1]:
        padding.append(pz // 2)
        padding.append(pz // 2 - 1)
    padding += [0, 0]
    padded = torch.nn.functional.pad(log[0], padding)
    return padded.cpu()


class AddPatchSamplerProbabilityMap(Transform):
    def __init__(
        self,
        patch_size: int = 64,
        device: str = "cuda",
        segmentation_mask_key: str = "tumor",
        probability_map_name: str = "probability_map",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.patch_size = patch_size
        self.device = torch.device(device)
        self.segmentation_mask_key = segmentation_mask_key
        self.probability_map_name = probability_map_name
        self.args_names = ["patch_size", "device", "segmentation_mask_key", "probability_map_name"]

    def apply_transform(self, subject):
        seg_mask = subject[self.segmentation_mask_key]
        prob_mask = segmentation_to_probability_map(seg_mask.data, self.patch_size, device=self.device)
        subject.add_image(
            tio.LabelMap(tensor=prob_mask, affine=seg_mask.affine),
            image_name=self.probability_map_name,
        )
        return subject
