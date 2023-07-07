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
_C.TRAINER.precision = "32-true"
_C.TRAINER.max_epochs = 10
# Useful when debugging to only train on portion of dataset
_C.TRAINER.limit_train_batches = 1.0
_C.TRAINER.limit_val_batches = 1.0
# Useful when debugging to overfit on purpose
_C.TRAINER.overfit_batches = 0.0
_C.TRAINER.check_val_every_n_epoch = 1
_C.TRAINER.log_every_n_steps = 10
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


_C.DATA = CN()
_C.DATA.FILEPATH_NPZ = "/home/bryan/data/brain_tumor/caidm_3d_96/data.npz"
# /Users/bryan/gdrive/Radiology-Research/brain_tumor/data/caidm_3d_96/data.npz"
_C.DATA.PATIENT_EXCLUSION_CSV = "/home/bryan/src/mgmt-promoter-methylation/mgmt/data/patient_exclusion.csv"
# /Users/bryan/src/mgmt-promoter-methylation/mgmt/data/patient_exclusion.csv"
_C.DATA.TRAIN_VAL_RATIO = 0.85
_C.DATA.BATCH_SIZE = 16
_C.DATA.CROP_DIM = [40, 40, 8]
_C.DATA.SHAPE_MULTIPLE = 8
_C.DATA.NUM_WORKERS = 12
_C.DATA.MODALITY = "fla"

_C.SOLVER = CN()
# Adam, AdamW
_C.SOLVER.OPTIMIZER_NAME = "AdamW"
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.WEIGHT_DECAY = 0.0005  # optimizer weight decay 5e-4

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
_C.MODEL.RESNET.n_input_channels = 1
_C.MODEL.RESNET.conv1_t_size = 7
_C.MODEL.RESNET.conv1_t_stride = 1
_C.MODEL.RESNET.no_max_pool = False
_C.MODEL.RESNET.shortcut_type = "B"
_C.MODEL.RESNET.widen_factor = 1.0
_C.MODEL.RESNET.num_classes = 1

_C.METRICS = CN()
_C.METRICS.THRESHOLD = 0.5
