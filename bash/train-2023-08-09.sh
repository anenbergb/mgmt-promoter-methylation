#!/bin/bash

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/debug1

python mgmt/tasks/train.py \
-c architectures/EfficientNet-d.yml \
-c train-configs/patch-based-trainer-1.yml \
OUTPUT_DIR "${EXPR_DIR}/run1" \
SEED_EVERYTHING 1 \
MODEL.EfficientNet.image_size 64 \
TRAINER.max_epochs 10