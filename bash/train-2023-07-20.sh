#!/bin/bash

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-20/efficient/try1

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run1 \
# TRAINER.max_epochs 200 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 96

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run2 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 96 \
# SOLVER.WEIGHT_DECAY 0.005

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run3 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 96 \
# SOLVER.OneCycleLR.max_lr 0.005

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run4 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 96 \
# SOLVER.OneCycleLR.max_lr 0.01

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-a.yml \
# OUTPUT_DIR $EXPR_DIR/run5 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 96

# python mgmt/tasks/train.py \
# OUTPUT_DIR $EXPR_DIR/run6 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.NAME ResNet


# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run7 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# MODEL.EfficientNet.image_size 96


# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run8 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
# MODEL.EfficientNet.image_size 96 \
# AUGMENT.RANDOM_GAMMA.p 1.0 \
# AUGMENT.RANDOM_NOISE.p 1.0 \
# AUGMENT.RANDOM_MOTION.p 1.0


# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run9 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim "[96, 96, 96]" \
# PREPROCESS.RESIZE_ENABLED False \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 96


# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run10 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[112, 112, 112]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 112 \
# SOLVER.WEIGHT_DECAY 0.005

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run11 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[112, 112, 112]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 112


# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-a.yml \
# OUTPUT_DIR $EXPR_DIR/run12 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[112, 112, 112]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.NAME ResNet

# python mgmt/tasks/train.py \
# OUTPUT_DIR $EXPR_DIR/run13 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
# PREPROCESS.RESIZE.target_shape "[112, 112, 112]" \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 112

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run14 \
# TRAINER.max_epochs 100 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim "[96, 96, 96]" \
# PREPROCESS.RESIZE_ENABLED False \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 96

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-b.yml \
# OUTPUT_DIR $EXPR_DIR/run15 \
# TRAINER.max_epochs 200 \
# DATA.MODALITY t1c \
# PREPROCESS.CROP_LARGEST_TUMOR.crop_dim "[112, 112, 112]" \
# PREPROCESS.RESIZE_ENABLED False \
# AUGMENT.RANDOM_AFFINE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# MODEL.EfficientNet.image_size 112


EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-20/try2

python mgmt/tasks/train.py \
OUTPUT_DIR $EXPR_DIR/resnet-64-run1 \
TRAINER.max_epochs 200 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
PREPROCESS.RESIZE.target_shape "[64, 64, 64]" \
AUGMENT.RANDOM_GAMMA.p 1.0 \
AUGMENT.RANDOM_NOISE.p 1.0 \
AUGMENT.RANDOM_MOTION.p 1.0 \
MODEL.NAME ResNet

python mgmt/tasks/train.py \
-c architectures/EfficientNet-b.yml \
OUTPUT_DIR $EXPR_DIR/efficient-b-64-run1 \
TRAINER.max_epochs 200 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
PREPROCESS.RESIZE.target_shape "[64, 64, 64]" \
MODEL.EfficientNet.image_size 64 \
AUGMENT.RANDOM_GAMMA.p 1.0 \
AUGMENT.RANDOM_NOISE.p 1.0 \
AUGMENT.RANDOM_MOTION.p 1.0