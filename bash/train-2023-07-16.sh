#!/bin/bash

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-16/try2


# python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run1 \
# TRAINER.max_epochs 2 \
# DATA.MODALITY t1w \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# TRAINER.profiler simple

# python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run2 \
# TRAINER.max_epochs 3 \
# DATA.MODALITY t1w \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# TRAINER.profiler advanced

# python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run3 \
# TRAINER.max_epochs 3 \
# DATA.MODALITY concat \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# TRAINER.profiler advanced

# python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run4 \
# TRAINER.max_epochs 3 \
# DATA.MODALITY t1w \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# TRAINER.profiler advanced \
# DATA.BATCH_SIZE 8 \
# PREPROCESS.RESIZE.target_shape "[64, 64, 64]"

# python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run5 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1w \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# TRAINER.profiler simple

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-16/debug

# python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run1 \
# TRAINER.max_epochs 2 \
# DATA.MODALITY t1w \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# TRAINER.profiler simple \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_GAMMA_ENABLED False \
# AUGMENT.RANDOM_NOISE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# AUGMENT.RANDOM_MOTION_ENABLED False

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-16/try3

# python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run1 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# TRAINER.profiler simple \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_GAMMA_ENABLED False \
# AUGMENT.RANDOM_NOISE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# AUGMENT.RANDOM_MOTION_ENABLED False

# python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run2 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False

# python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run3 \
# TRAINER.max_epochs 50 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False

# python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run4 \
# TRAINER.max_epochs 50 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim "[32, 32, 32]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-16/try4

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run1 \
TRAINER.max_epochs 50 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
MODEL.RESNET.conv1_t_size 3

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-16/resnet-custom-try1

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run1 \
TRAINER.max_epochs 50 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
MODEL.NAME ResNet

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run2 \
TRAINER.max_epochs 50 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
MODEL.NAME ResNet \
MODEL.ResNet.block bottleneck

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run3 \
TRAINER.max_epochs 50 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
MODEL.NAME ResNet \
PREPROCESS.RESIZE.target_shape "[64, 64, 32]"