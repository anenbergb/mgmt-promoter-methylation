#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
DATA.PICKLE_SUBJECTS.cache_dir /home/banenber/temp/cache_dir \
OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-28/train-lr-003-fold-00 \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 250 \
DATA.BATCH_SIZE 40 \
DATA.NUM_WORKERS 0 \
DATA.VAL_NUM_WORKERS 0 \
DATA.SPLITS.FOLD_INDEX 0 \
PATCH_BASED_TRAINER.QUEUE.max_length 40 \
PATCH_BASED_TRAINER.QUEUE.samples_per_volume 2 \
TRAINER.check_val_every_n_epoch 10 \
SOLVER.OneCycleLR.max_lr 0.003

python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
DATA.PICKLE_SUBJECTS.cache_dir /home/banenber/temp/cache_dir \
OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-28/train-lr-003-fold-01 \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 250 \
DATA.BATCH_SIZE 40 \
DATA.NUM_WORKERS 0 \
DATA.VAL_NUM_WORKERS 0 \
DATA.SPLITS.FOLD_INDEX 1 \
PATCH_BASED_TRAINER.QUEUE.max_length 40 \
PATCH_BASED_TRAINER.QUEUE.samples_per_volume 2 \
TRAINER.check_val_every_n_epoch 10 \
SOLVER.OneCycleLR.max_lr 0.003

python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
DATA.PICKLE_SUBJECTS.cache_dir /home/banenber/temp/cache_dir \
OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-28/train-lr-003-fold-02 \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 250 \
DATA.BATCH_SIZE 40 \
DATA.NUM_WORKERS 0 \
DATA.VAL_NUM_WORKERS 0 \
DATA.SPLITS.FOLD_INDEX 2 \
PATCH_BASED_TRAINER.QUEUE.max_length 40 \
PATCH_BASED_TRAINER.QUEUE.samples_per_volume 2 \
TRAINER.check_val_every_n_epoch 10 \
SOLVER.OneCycleLR.max_lr 0.003

python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
DATA.PICKLE_SUBJECTS.cache_dir /home/banenber/temp/cache_dir \
OUTPUT_DIR /home/banenber/expr/brain_tumor/2023-12-28/train-lr-003-fold-03 \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 250 \
DATA.BATCH_SIZE 40 \
DATA.NUM_WORKERS 0 \
DATA.VAL_NUM_WORKERS 0 \
DATA.SPLITS.FOLD_INDEX 3 \
PATCH_BASED_TRAINER.QUEUE.max_length 40 \
PATCH_BASED_TRAINER.QUEUE.samples_per_volume 2 \
TRAINER.check_val_every_n_epoch 10 \
SOLVER.OneCycleLR.max_lr 0.003

# PATCH_BASED_TRAINER.QUEUE.verbose True \
# PATCH_BASED_TRAINER.QUEUE.shuffle_subjects False
# TRAINER.profiler simple