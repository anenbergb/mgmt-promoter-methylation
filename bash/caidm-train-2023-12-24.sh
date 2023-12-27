#!/bin/bash



export CUDA_VISIBLE_DEVICES=1
# python mgmt/tasks/train.py \
# -c architectures/MultiRes-Basic.yml \
# -c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
# OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-24/train-001-fold-01 \
# SEED_EVERYTHING 1 \
# TRAINER.max_epochs 100 \
# DATA.BATCH_SIZE 12 \
# DATA.NUM_WORKERS 1 \
# DATA.SPLITS.FOLD_INDEX 1

# export CUDA_VISIBLE_DEVICES=2
# python mgmt/tasks/train.py \
# -c architectures/MultiRes-Basic.yml \
# -c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
# OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-22/train-002-fold-02 \
# SEED_EVERYTHING 1 \
# TRAINER.max_epochs 100 \
# DATA.BATCH_SIZE 12 \
# DATA.NUM_WORKERS 1 \
# DATA.SPLITS.FOLD_INDEX 2

export CUDA_VISIBLE_DEVICES=1

python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-24/train-000-fold-00 \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 100 \
DATA.BATCH_SIZE 12 \
DATA.NUM_WORKERS 1 \
DATA.SPLITS.FOLD_INDEX 0

python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-24/train-003-fold-03 \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 100 \
DATA.BATCH_SIZE 12 \
DATA.NUM_WORKERS 1 \
DATA.SPLITS.FOLD_INDEX 3