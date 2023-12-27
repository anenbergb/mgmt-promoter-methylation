#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# EXPR_DIR=/home/banenber/expr/brain_tumor/2023-12-19/debug-001

# python mgmt/tasks/train.py \
# -c architectures/MultiRes-Basic.yml \
# -c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
# OUTPUT_DIR $EXPR_DIR \
# SEED_EVERYTHING 1 \
# TRAINER.max_epochs 10 \
# DATA.BATCH_SIZE 4 \
# TRAINER.limit_train_batches 20 \
# TRAINER.limit_val_batches 3
# CHECKPOINT.PATH last


# EXPR_DIR=/home/banenber/expr/brain_tumor/2023-12-19/debug-009

# python mgmt/tasks/train.py \
# -c architectures/MultiRes-Basic.yml \
# -c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
# OUTPUT_DIR $EXPR_DIR \
# SEED_EVERYTHING 1 \
# TRAINER.max_epochs 100 \
# DATA.BATCH_SIZE 12 \
# DATA.NUM_WORKERS 1

# python mgmt/tasks/train.py \
# -c architectures/MultiRes-Basic.yml \
# -c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
# OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-21/debug-001 \
# SEED_EVERYTHING 1 \
# TRAINER.max_epochs 100 \
# DATA.BATCH_SIZE 12 \
# DATA.NUM_WORKERS 1 \
# PATCH_BASED_TRAINER.QUEUE.samples_per_volume 5 \
# CHECKPOINT.PATH last

# TRAINER.limit_train_batches 25.0

python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-21/train-002 \
SEED_EVERYTHING 2 \
TRAINER.max_epochs 100 \
DATA.BATCH_SIZE 12 \
DATA.NUM_WORKERS 1 \
PATCH_BASED_TRAINER.QUEUE.max_length 200

python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-22/debug-002 \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 100 \
DATA.BATCH_SIZE 12 \
DATA.NUM_WORKERS 1 \
PATCH_BASED_TRAINER.QUEUE.samples_per_volume 5 \
PATCH_BASED_TRAINER.QUEUE.max_length 1000