#!/bin/bash

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-26/try2

python mgmt/tasks/train.py \
-c architectures/EfficientNet-d-group.yml \
OUTPUT_DIR $EXPR_DIR/efficient-d-group-96-run1 \
TRAINER.max_epochs 100 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
MODEL.EfficientNet.image_size 96 \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False

python mgmt/tasks/train.py \
-c architectures/EfficientNet-d-group.yml \
OUTPUT_DIR $EXPR_DIR/efficient-d-group-96-run2 \
TRAINER.max_epochs 100 \
DATA.MODALITY t1c \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
PREPROCESS.RESIZE.target_shape "[96, 96, 96]" \
MODEL.EfficientNet.image_size 96 \
AUGMENT.RANDOM_AFFINE_ENABLED False \
AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
MODEL.EfficientNet.dropout_rate 0.5 \
MODEL.EfficientNet.drop_connect_rate 0.3