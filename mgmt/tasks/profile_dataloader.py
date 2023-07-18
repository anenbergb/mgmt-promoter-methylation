import argparse
import copy
import os
import sys
from collections import OrderedDict

import torchio as tio
from fvcore.common.config import CfgNode
from lightning.pytorch import seed_everything
from lightning.pytorch.profilers import SimpleProfiler
from tqdm import tqdm

from mgmt.config import get_cfg
from mgmt.data.dataloader import DataModule
from mgmt.data.subject_transforms import CropLargestTumor
from mgmt.tasks.train import setup_config
from mgmt.transforms.rescale_intensity import RescaleIntensity
from mgmt.utils.logger import setup_logger


def main(cfg: CfgNode, iterations: int = 100):
    setup_logger(cfg)
    seed_everything(cfg.SEED_EVERYTHING, workers=True)
    profiler = SimpleProfiler(dirpath=cfg.OUTPUT_DIR, filename="simple-profiler.txt", extended=True)

    datamodule = DataModule(cfg)
    datamodule.prepare_data()
    transforms = make_transforms(cfg)

    for i in tqdm(range(iterations)):
        i = i % len(datamodule.subjects)
        subject = datamodule.subjects[i]
        profiler.start("process_subject")
        process_subject(subject, transforms, profiler)
        profiler.stop("process_subject")

    profiler.describe()


def process_subject(subject, transforms, profiler):
    with profiler.profile("subject.copy"):
        subject = copy.deepcopy(subject)
    with profiler.profile("subject.load"):
        subject.load()

    for name, transform in transforms.items():
        with profiler.profile(name):
            subject = transform(subject)


def make_transforms(cfg: CfgNode):
    include = None
    if cfg.DATA.MODALITY != "concat":
        include = [cfg.DATA.MODALITY]
    include = None
    transforms = OrderedDict()

    if cfg.PREPROCESS.TO_CANONICAL_ENABLED:
        transforms["ToCanonical"] = tio.ToCanonical(include=include)

    def rescale_intensity():
        if cfg.PREPROCESS.RESCALE_INTENSITY_ENABLED:
            kwargs = copy.copy(cfg.PREPROCESS.RESCALE_INTENSITY)
            skull_mask = kwargs.pop("SKULL_MASK")
            kwargs.pop("BEFORE_CROP")
            masking_method = None
            if skull_mask:
                masking_method = lambda x: x > 0.0
            transforms["RescaleIntensity1"] = RescaleIntensity(masking_method=masking_method, include=include, **kwargs)
        if cfg.AUGMENT.RANDOM_NOISE_ENABLED:
            transforms["RandomNoise"] = tio.RandomNoise(include=include, **cfg.AUGMENT.RANDOM_NOISE)
            # rescale back to the target intensity scale range
            # do not need to apply percentile filtering or mask
            if cfg.PREPROCESS.RESCALE_INTENSITY_ENABLED:
                transforms["RescaleIntensity2"] = RescaleIntensity(
                    include=include,
                    out_min_max=cfg.PREPROCESS.RESCALE_INTENSITY.out_min_max,
                )

    if cfg.AUGMENT.RANDOM_AFFINE_ENABLED:
        transforms["RandomAffine"] = tio.RandomAffine(include=include, **cfg.AUGMENT.RANDOM_AFFINE)
    if cfg.AUGMENT.RANDOM_GAMMA_ENABLED:
        transforms["RandomGamma"] = tio.RandomGamma(include=include, **cfg.AUGMENT.RANDOM_GAMMA)
    if cfg.AUGMENT.RANDOM_BIAS_FIELD:
        transforms["RandomBiasField"] = tio.RandomBiasField(include=include, **cfg.AUGMENT.RANDOM_BIAS_FIELD)
    if cfg.AUGMENT.RANDOM_MOTION_ENABLED:
        transforms["RandomMotion"] = tio.RandomMotion(include=include, **cfg.AUGMENT.RANDOM_MOTION)

    if cfg.PREPROCESS.RESCALE_INTENSITY.BEFORE_CROP:
        rescale_intensity()
    if cfg.PREPROCESS.CROP_LARGEST_TUMOR_ENABLED:
        transforms["CropLargestTumor"] = CropLargestTumor(include=include, **cfg.PREPROCESS.CROP_LARGEST_TUMOR)
    if not cfg.PREPROCESS.RESCALE_INTENSITY.BEFORE_CROP:
        rescale_intensity()

    if cfg.PREPROCESS.RESIZE_ENABLED:
        transforms["Resize"] = tio.Resize(include=include, **cfg.PREPROCESS.RESIZE)
    transforms["EnsuareShapeMultiple"] = tio.EnsureShapeMultiple(
        include=include, **cfg.PREPROCESS.ENSURE_SHAPE_MULTIPLE
    )
    return transforms


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Measure the runtime of data preprocess and augmentation routines.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        "-c",
        default=[],
        metavar="FILE",
        action="append",
        help="Path to config file. Can provide multiple config files, "
        "which are combined together, overwriting the previous.",
    )
    parser.add_argument("--iterations", default=100, type=int, help="Number of iterations to profile.")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg = setup_config(args)
    sys.exit(main(cfg, args.iterations))
