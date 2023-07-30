#!/bin/bash

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-26/try1

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-c.yml \
# OUTPUT_DIR $EXPR_DIR/efficient-c-96-run1 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# MODEL.EfficientNet.image_size 96 \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-c-group.yml \
# OUTPUT_DIR $EXPR_DIR/efficient-c-group-96-run2 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# MODEL.EfficientNet.image_size 96 \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-26/try2

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-d-group.yml \
# OUTPUT_DIR $EXPR_DIR/efficient-d-group-96-run1 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# MODEL.EfficientNet.image_size 96 \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-d-group.yml \
# OUTPUT_DIR $EXPR_DIR/efficient-d-group-96-run2 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# MODEL.EfficientNet.image_size 96 \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.dropout_rate 0.5 \
# MODEL.EfficientNet.drop_connect_rate 0.3

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-26/try3

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-d.yml \
# OUTPUT_DIR $EXPR_DIR/efficient-d-64-run1 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[64, 64, 64]" \
# MODEL.EfficientNet.image_size 64

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-d.yml \
# OUTPUT_DIR $EXPR_DIR/efficient-d-64-run2 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[64, 64, 64]" \
# MODEL.EfficientNet.image_size 64

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-d.yml \
# OUTPUT_DIR $EXPR_DIR/efficient-d-64-run3 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim "[64, 64, 64]" \
# PREPROCESS.RESIZE_ENABLED False \
# MODEL.EfficientNet.image_size 64

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-d.yml \
# OUTPUT_DIR $EXPR_DIR/efficient-d-64-run4 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t2w \
# PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim "[64, 64, 64]" \
# PREPROCESS.RESIZE_ENABLED False \
# MODEL.EfficientNet.image_size 64

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-26/try4

python mgmt/tasks/train.py \
-c architectures/EfficientNet-d.yml \
OUTPUT_DIR $EXPR_DIR/efficient-d-64-run1 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1c \
PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim "[64, 64, 64]" \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
PREPROCESS.RESIZE_ENABLED False \
MODEL.EfficientNet.image_size 64

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-26/try5

python mgmt/tasks/train.py \
-c architectures/EfficientNet-d.yml \
OUTPUT_DIR $EXPR_DIR/efficient-d-64-run1 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1c \
PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim "[64, 64, 64]" \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED True \
PREPROCESS.RESIZE_ENABLED False \
MODEL.EfficientNet.image_size 64