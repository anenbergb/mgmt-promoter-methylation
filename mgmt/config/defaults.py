# from yacs.config import CfgNode as CN

from fvcore.common.config import CfgNode as CN

_C = CN()
_C.SEED_EVERYTHING = 42
_C.OUTPUT_DIR = "output"
_C.CHECKPOINT = CN()
_C.CHECKPOINT.PATH = None  # "best", "last", "hpc", or path to checkpoint
_C.CHECKPOINT.save_top_k = 3

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 8

_C.TRAINER = CN()
_C.TRAINER.accelerator = "auto"
_C.TRAINER.strategy = "auto"
_C.TRAINER.devices = "auto"
_C.TRAINER.num_nodes = 1
_C.TRAINER.precision = "32-true"  # "16-mixed"  # "32-true"
_C.TRAINER.max_epochs = 10
# Useful when debugging to only train on portion of dataset
_C.TRAINER.limit_train_batches = 1.0
_C.TRAINER.limit_val_batches = 1.0
# Useful when debugging to overfit on purpose
_C.TRAINER.overfit_batches = 0.0
_C.TRAINER.check_val_every_n_epoch = 1
_C.TRAINER.log_every_n_steps = 50
_C.TRAINER.accumulate_grad_batches = 1
# TODO: consider adding gradient clipping
#   gradient_clip_val: null
#   gradient_clip_algorithm: null
_C.TRAINER.deterministic = False
_C.TRAINER.benchmark = False
# Whether to use torch.inference_mode() or torch.no_grad()
# mode during evaluation (validate/test/predict)
_C.TRAINER.inference_mode = True
_C.TRAINER.use_distributed_sampler = True
_C.TRAINER.profiler = None  # ("simple", "advanced")
_C.TRAINER.detect_anomaly = False
_C.TRAINER.barebones = False
_C.TRAINER.sync_batchnorm = False
_C.TRAINER.reload_dataloaders_every_n_epochs = 0

_C.PROFILER = CN()
_C.PROFILER.SIMPLE = CN()
_C.PROFILER.SIMPLE.filename = "simple-profiler.txt"
_C.PROFILER.SIMPLE.extended = True
_C.PROFILER.ADVANCED = CN()
_C.PROFILER.ADVANCED.filename = "advanced-profiler.txt"
# his can be used to limit the number of functions reported for each action.
# either an integer (to select a count of lines),
# or a decimal fraction between 0.0 and 1.0 inclusive (to select a percentage of lines)
_C.PROFILER.ADVANCED.line_count_restriction = 1.0

_C.DATA = CN()

_C.DATA.SOURCE = "nifti"  # nifti, numpy, pickle-subjects
_C.DATA.NIFTI = CN()
_C.DATA.NIFTI.FOLDER_PATH = "/home/bryan/data/brain_tumor/caidm_3d_240"
_C.DATA.NIFTI.TRAIN_LABELS = "/home/bryan/data/brain_tumor/classification/train_labels.csv"
# folders with test data are titled "MGMT"
_C.DATA.NIFTI.TEST_FOLDER_PREFIX = "MGMT"
_C.DATA.LAZY_LOAD_TRAIN = True

_C.DATA.NUMPY = CN()
_C.DATA.NUMPY.FILEPATH_NPZ = "/home/bryan/data/brain_tumor/caidm_3d_96/data.npz"
# /Users/bryan/gdrive/Radiology-Research/brain_tumor/data/caidm_3d_96/data.npz"
_C.DATA.NUMPY.PATIENT_EXCLUSION_CSV = "/home/bryan/src/mgmt-promoter-methylation/mgmt/data/patient_exclusion.csv"

_C.DATA.PICKLE_SUBJECTS = CN()
_C.DATA.PICKLE_SUBJECTS.folder_path = "/home/bryan/expr/brain_tumor/2023-08-08/preprocess-subjects-crop-64-t1c"
_C.DATA.PICKLE_SUBJECTS.filter_file_prefix = "P-"

_C.DATA.TRAIN_VAL_RATIO = 0.85
_C.DATA.TRAIN_VAL_MANUAL_SEED = 10
_C.DATA.BATCH_SIZE = 16
_C.DATA.NUM_WORKERS = 4
# 'fla', 't1w', 't1c', 't2w', 'concat'
_C.DATA.MODALITY = "t1c"  # "concat"
_C.DATA.MODALITY_CONCAT = ["fla", "t1w", "t1c", "t2w"]

# https://torchio.readthedocs.io/patches/patch_training.html
_C.PATCH_BASED_TRAINER = CN()
_C.PATCH_BASED_TRAINER.ENABLED = False
# skip transforms in the case where they were already applied saved into the subject pickle
_C.PATCH_BASED_TRAINER.TRANSFORMS_ENABLED = False

_C.PATCH_BASED_TRAINER.LABEL_SAMPLER = CN()
_C.PATCH_BASED_TRAINER.LABEL_SAMPLER.patch_size = [64, 64, 64]
_C.PATCH_BASED_TRAINER.LABEL_SAMPLER.label_name = "tumor"
_C.PATCH_BASED_TRAINER.LABEL_SAMPLER.label_probabilities = None

_C.PATCH_BASED_TRAINER.WEIGHTED_SAMPLER = CN()
_C.PATCH_BASED_TRAINER.WEIGHTED_SAMPLER.patch_size = [64, 64, 64]
_C.PATCH_BASED_TRAINER.WEIGHTED_SAMPLER.probability_map = "patch_sampling_probability_map"

_C.PATCH_BASED_TRAINER.QUEUE = CN()
# Maximum number of patches that can be stored in the queue.
# 491 * 25 / 3
_C.PATCH_BASED_TRAINER.QUEUE.max_length = 4100
# Default number of patches to extract from each volume.
_C.PATCH_BASED_TRAINER.QUEUE.samples_per_volume = 25
_C.PATCH_BASED_TRAINER.QUEUE.shuffle_subjects = True
_C.PATCH_BASED_TRAINER.QUEUE.shuffle_patches = True
_C.PATCH_BASED_TRAINER.QUEUE.start_background = True
_C.PATCH_BASED_TRAINER.QUEUE.verbose = False

_C.PREPROCESS = CN()
_C.PREPROCESS.TO_CANONICAL_ENABLED = True

_C.PREPROCESS.SKULL_CROP_TRANSFORM_ENABLED = True
_C.PREPROCESS.SKULL_CROP_TRANSFORM = CN()
_C.PREPROCESS.SKULL_CROP_TRANSFORM.mask_image_name = "t1c"
_C.PREPROCESS.SKULL_CROP_TRANSFORM.padding = [0, 0, 0]

_C.PREPROCESS.ADD_PATCH_SAMPLER_PROB_MAP_ENABLED = True

_C.PREPROCESS.RESCALE_INTENSITY_ENABLED = True
_C.PREPROCESS.RESCALE_INTENSITY = CN()
_C.PREPROCESS.RESCALE_INTENSITY.out_min_max = [-1, 1]
_C.PREPROCESS.RESCALE_INTENSITY.percentiles = [0.5, 99.5]  # (0.5, 99.5) to control for possible outliers
_C.PREPROCESS.RESCALE_INTENSITY.SKULL_MASK = True  # apply mask: lambda x: x > 0.0
_C.PREPROCESS.RESCALE_INTENSITY.BEFORE_CROP = True

_C.PREPROCESS.CROP_LARGEST_TUMOR_ENABLED = True
_C.PREPROCESS.CROP_LARGEST_TUMOR = CN()
_C.PREPROCESS.CROP_LARGEST_TUMOR.crop_dim = [96, 96, 96]  # or None

# Crop the volume to the largest tumor as the first step in the
# data transform process to speed up compute downstream
_C.PREPROCESS.EARLY_CROP_LARGEST_TUMOR_ENABLED = True
_C.PREPROCESS.EARLY_CROP_LARGEST_TUMOR = CN()
_C.PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim = [96, 96, 96]

_C.PREPROCESS.RESIZE_ENABLED = True
_C.PREPROCESS.RESIZE = CN()
_C.PREPROCESS.RESIZE.target_shape = [64, 64, 32]
_C.PREPROCESS.RESIZE.image_interpolation = "linear"

_C.PREPROCESS.ENSURE_SHAPE_MULTIPLE = CN()
_C.PREPROCESS.ENSURE_SHAPE_MULTIPLE.target_multiple = [8, 8, 8]
_C.PREPROCESS.ENSURE_SHAPE_MULTIPLE.method = "pad"  # 'crop', 'pad'

_C.PREPROCESS.RESAMPLE_ENABLED = False
_C.PREPROCESS.RESAMPLE = CN()
# target is the output spacing. 2.0 means divide size by factor of 2.0
_C.PREPROCESS.RESAMPLE.target = 2.0
_C.PREPROCESS.RESAMPLE.image_interpolation = "linear"
_C.PREPROCESS.RESAMPLE.label_interpolation = "nearest"

_C.AUGMENT = CN()
_C.AUGMENT.RANDOM_AFFINE_ENABLED = True
_C.AUGMENT.RANDOM_AFFINE = CN()
_C.AUGMENT.RANDOM_AFFINE.p = 1.0
# could consider slightly rescaling of (0.75, 1.25, 0.75, 1.25, 1, 1)
_C.AUGMENT.RANDOM_AFFINE.scales = [1, 1, 1, 1, 1, 1]
# only rotate about the z-axis (depth)
_C.AUGMENT.RANDOM_AFFINE.degrees = [0, 0, 0, 0, 0, 360]

_C.AUGMENT.RANDOM_GAMMA_ENABLED = True
_C.AUGMENT.RANDOM_GAMMA = CN()
_C.AUGMENT.RANDOM_GAMMA.p = 0.5
_C.AUGMENT.RANDOM_GAMMA.log_gamma = [-0.3, 0.3]

# https://torchio.readthedocs.io/transforms/augmentation.html#randomnoise
_C.AUGMENT.RANDOM_NOISE_ENABLED = True
_C.AUGMENT.RANDOM_NOISE = CN()
_C.AUGMENT.RANDOM_NOISE.p = 0.5
_C.AUGMENT.RANDOM_NOISE.mean = [0.0, 0.0]
_C.AUGMENT.RANDOM_NOISE.std = [0, 0.1]  # greater than 0.1 looks pretty grainy

_C.AUGMENT.RANDOM_BIAS_FIELD_ENABLED = True
_C.AUGMENT.RANDOM_BIAS_FIELD = CN()
_C.AUGMENT.RANDOM_BIAS_FIELD.p = 0.5
_C.AUGMENT.RANDOM_BIAS_FIELD.coefficients = [-0.1, 0.1]
_C.AUGMENT.RANDOM_BIAS_FIELD.order = 3

_C.AUGMENT.RANDOM_MOTION_ENABLED = False
_C.AUGMENT.RANDOM_MOTION = CN()
_C.AUGMENT.RANDOM_MOTION.p = 0.5
_C.AUGMENT.RANDOM_MOTION.degrees = [10.0, 10.0]
_C.AUGMENT.RANDOM_MOTION.translation = [-10.0, 10.0]
_C.AUGMENT.RANDOM_MOTION.num_transforms = 1
_C.AUGMENT.RANDOM_MOTION.image_interpolation = "linear"

_C.VAL_INFERENCE = CN()
_C.VAL_INFERENCE.MODE = "CropLargestTumor"
_C.VAL_INFERENCE.CROP_LARGEST_TUMOR = CN()
_C.VAL_INFERENCE.CROP_LARGEST_TUMOR.crop_dim = [96, 96, 96]

_C.SOLVER = CN()
# Adam, AdamW
_C.SOLVER.OPTIMIZER_NAME = "AdamW"
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.WEIGHT_DECAY = 0.001  # optimizer weight decay 5e-4

_C.SOLVER.ADAM = CN()
_C.SOLVER.ADAM.betas = [0.9, 0.999]
_C.SOLVER.NADAM = CN()
_C.SOLVER.NADAM.momentum_decay = 4e-3

_C.SOLVER.SGD = CN()
_C.SOLVER.SGD.momentum = 0.937
_C.SOLVER.SGD.dampening = 0.0
_C.SOLVER.SGD.nesterov = False

_C.SOLVER.SCHEDULER_NAME = "OneCycleLR"
# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
_C.SOLVER.OneCycleLR = CN()
_C.SOLVER.OneCycleLR.max_lr = 0.001
_C.SOLVER.OneCycleLR.pct_start = 0.3
_C.SOLVER.OneCycleLR.anneal_strategy = "cos"
_C.SOLVER.OneCycleLR.cycle_momentum = True
_C.SOLVER.OneCycleLR.base_momentum = 0.85
_C.SOLVER.OneCycleLR.max_momentum = 0.95
# Determines the initial learning rate via initial_lr = max_lr/div_factor
_C.SOLVER.OneCycleLR.div_factor = 25.0
# Determines the minimum learning rate via min_lr = initial_lr/final_div_factor
_C.SOLVER.OneCycleLR.final_div_factor = 10000.0
_C.SOLVER.OneCycleLR.three_phase = False
# The index of the last batch. This parameter is used when resuming a training job.
# Since step() should be invoked after each batch instead of after each epoch,
# this number represents the total number of batches computed, not the total number of epochs computed.
# When last_epoch=-1, the schedule is started from the beginning. Default: -1

_C.SOLVER.ReduceLROnPlateau = CN()
_C.SOLVER.ReduceLROnPlateau.mode = "min"
# Factor by which the learning rate will be reduced. new_lr = lr * factor.
_C.SOLVER.ReduceLROnPlateau.factor = 0.1
# Number of epochs with no improvement after which learning rate will be reduced.
# For example, if patience = 2, then we will ignore the first 2 epochs with no improvement,
# and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then.
_C.SOLVER.ReduceLROnPlateau.patience = 10
# One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or
#  best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold
# in max mode or best - threshold in min mode.
_C.SOLVER.ReduceLROnPlateau.threshold_mode = "rel"
_C.SOLVER.ReduceLROnPlateau.cooldown = 0
_C.SOLVER.ReduceLROnPlateau.min_lr = 0
_C.SOLVER.ReduceLROnPlateau.eps = 1e-08

_C.SOLVER.StepLR = CN()
_C.SOLVER.StepLR.step_size = 10
_C.SOLVER.StepLR.gamma = 0.1
_C.SOLVER.StepLR.last_epoch = -1
_C.SOLVER.StepLR.verbose = False


_C.SOLVER.MultiStepLR = CN()
# List of epoch indices. Must be increasing.
_C.SOLVER.MultiStepLR.milestones = [5, 10]
# Multiplicative factor of learning rate decay. Default: 0.1.
_C.SOLVER.MultiStepLR.gamma = 0.1

_C.SOLVER.WARMUP = CN()
_C.SOLVER.WARMUP.ENABLED = False
_C.SOLVER.WARMUP.warmup_steps = 3
_C.SOLVER.WARMUP.warmup_strategy = "linear"

_C.MODEL = CN()
_C.MODEL.NAME = "resnet10"
# Monai resnet https://docs.monai.io/en/stable/_modules/monai/networks/nets/resnet.html
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.pretrained = False
_C.MODEL.RESNET.progress = False
_C.MODEL.RESNET.spatial_dims = 3
# _C.MODEL.RESNET.n_input_channels = 1
_C.MODEL.RESNET.conv1_t_size = 7
_C.MODEL.RESNET.conv1_t_stride = 1
_C.MODEL.RESNET.no_max_pool = False
_C.MODEL.RESNET.shortcut_type = "B"
_C.MODEL.RESNET.widen_factor = 1.0
_C.MODEL.RESNET.num_classes = 1

_C.MODEL.ResNet = CN()
_C.MODEL.ResNet.block = "basic"
_C.MODEL.ResNet.layers = [1, 1, 1, 1]
_C.MODEL.ResNet.block_inplanes = [16, 32, 64, 128]
_C.MODEL.ResNet.spatial_dims = 3
_C.MODEL.ResNet.conv1_t_size = 3
_C.MODEL.ResNet.conv1_t_stride = 1
_C.MODEL.ResNet.no_max_pool = False
_C.MODEL.ResNet.shortcut_type = "B"
_C.MODEL.ResNet.widen_factor = 1.0
_C.MODEL.ResNet.num_classes = 1

# TODO: add support for efficient net
_C.MODEL.EfficientNet = CN()
# string = (
#     f"r{self.num_repeat}_k{self.kernel_size}_s{self.stride}{self.stride}"
#     f"_e{self.expand_ratio}_i{self.input_filters}_o{self.output_filters}"
#     f"_se{self.se_ratio}"
# )
# if not self.id_skip:
#     string += "_noskip"
_C.MODEL.EfficientNet.blocks_args_str = [
    "r1_k3_s11_e1_i32_o16_se0.25",
    "r2_k3_s22_e6_i16_o24_se0.25",
    "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s22_e6_i40_o80_se0.25",
    "r3_k5_s11_e6_i80_o112_se0.25",
    "r4_k5_s22_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]
_C.MODEL.EfficientNet.spatial_dims = 3
_C.MODEL.EfficientNet.num_classes = 1
_C.MODEL.EfficientNet.width_coefficient = 1.0
_C.MODEL.EfficientNet.depth_coefficient = 1.0
_C.MODEL.EfficientNet.dropout_rate = 0.2
_C.MODEL.EfficientNet.image_size = 64
_C.MODEL.EfficientNet.norm = ["batch", {"eps": 1e-3, "momentum": 0.01}]
# ["layer", {"eps": 1e-3, "normalized_shape": (10, 10, 10)}]
_C.MODEL.EfficientNet.drop_connect_rate = 0.2
_C.MODEL.EfficientNet.depth_divisor = 8
_C.MODEL.EfficientNet.stem_kernel_size = 3
_C.MODEL.EfficientNet.stem_stride = 2
_C.MODEL.EfficientNet.head_output_filters = 256

_C.MODEL.MultiResolutionWithMask = CN()
_C.MODEL.MultiResolutionWithMask.pool = "adaptiveavg"
_C.MODEL.MultiResolutionWithMask.num_classes = 1


_C.BACKBONE = CN()
_C.BACKBONE.NAME = "BasicBackbone"
_C.BACKBONE.BasicBackbone = CN()
_C.BACKBONE.BasicBackbone.block_num_convs = [4, 4, 3, 2, 2]
_C.BACKBONE.BasicBackbone.block_out_channels = [16, 40, 64, 88, 112]
_C.BACKBONE.BasicBackbone.act = "relu"
_C.BACKBONE.BasicBackbone.norm = ["group", {"eps": 1e-5, "num_groups": 8}]
_C.BACKBONE.BasicBackbone.dropout = 0.1
_C.BACKBONE.BasicBackbone.dropout_dim = 1
_C.BACKBONE.BasicBackbone.groups = 1

_C.METRICS = CN()
_C.METRICS.THRESHOLD = 0.5
