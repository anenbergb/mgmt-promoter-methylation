#!/bin/bash

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-19/efficient/try1

# python mgmt/tasks/train.py \
# OUTPUT_DIR $EXPR_DIR/run1 \
# TRAINER.max_epochs 50 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[64, 64, 32]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.NAME EfficientNet

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-a.yml \
# OUTPUT_DIR $EXPR_DIR/run2 \
# TRAINER.max_epochs 50 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[64, 64, 32]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-19/resnet/try1

# python mgmt/tasks/train.py \
# OUTPUT_DIR $EXPR_DIR/run1 \
# TRAINER.max_epochs 50 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.NAME ResNet \
# PREPROCESS.RESIZE.target_shape "[64, 64, 64]"

# python mgmt/tasks/train.py \
# OUTPUT_DIR $EXPR_DIR/run2 \
# TRAINER.max_epochs 50 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.NAME ResNet \
# PREPROCESS.RESIZE.target_shape "[80, 80, 64]"

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-19/efficient/try2

python mgmt/tasks/train.py \
-c architectures/EfficientNet-b.yml \
OUTPUT_DIR $EXPR_DIR/run1 \
TRAINER.max_epochs 50 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
PREPROCESS.RESIZE.target_shape "[64, 64, 32]" \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False

python mgmt/tasks/train.py \
-c architectures/EfficientNet-b.yml \
OUTPUT_DIR $EXPR_DIR/run2 \
TRAINER.max_epochs 50 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
PREPROCESS.RESIZE.target_shape "[64, 64, 64]" \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False

python mgmt/tasks/train.py \
-c architectures/EfficientNet-b.yml \
OUTPUT_DIR $EXPR_DIR/run3 \
TRAINER.max_epochs 50 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
PREPROCESS.RESIZE.target_shape "[80, 80, 80]" \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
MODEL.EfficientNet.image_size 80

python mgmt/tasks/train.py \
-c architectures/EfficientNet-b.yml \
OUTPUT_DIR $EXPR_DIR/run4 \
TRAINER.max_epochs 50 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
PREPROCESS.RESIZE.target_shape "[80, 80, 80]" \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
MODEL.EfficientNet.image_size 80

python mgmt/tasks/train.py \
-c architectures/EfficientNet-b.yml \
OUTPUT_DIR $EXPR_DIR/run5 \
TRAINER.max_epochs 50 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
MODEL.EfficientNet.image_size 96