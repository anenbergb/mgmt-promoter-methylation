#!/bin/bash

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-09-05/debug1/resnet-concat-200

python mgmt/tasks/train.py \
-c architectures/ResNet-a.yml \
-c train-configs/patch-based-trainer-concat-crop-64.yml \
OUTPUT_DIR "${EXPR_DIR}/seed-1" \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 200