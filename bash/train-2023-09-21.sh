#!/bin/bash

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-09-21/debug/try1

python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
OUTPUT_DIR $EXPR_DIR \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 200 \
DATA.BATCH_SIZE 4