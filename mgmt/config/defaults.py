# from yacs.config import CfgNode as CN

from fvcore.common.config import CfgNode as CN

_C = CN()
_C.SEED_EVERYTHING = 42
_C.OUTPUT_DIR = "output"
_C.CHECKPOINT = CN()
_C.CHECKPOINT.PATH = "None" # "best", "last", "hpc", or path to checkpoint
_C.CHECKPOINT.save_top_k = 10

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 8

_C.TRAINER = CN()
_C.TRAINER.accelerator = "auto"
_C.TRAINER.strategy = "auto"
_C.TRAINER.devices = "auto"
_C.TRAINER.num_nodes = 1
_C.TRAINER.precision = "32-true"
_C.TRAINER.max_epochs = 2
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
_C.TRAINER.profiler = "None"  # ("simple", "advanced")
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
_C.SOLVER.BASE_LR = 0.0001

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
