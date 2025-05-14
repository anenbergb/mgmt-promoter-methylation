#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
DATA.PICKLE_SUBJECTS.folder_path /home/ubuntu/storage/radiology-research-west-2/expr/brain_tumor/preprocess-subjects-v2/resample-2.0-crop-64 \
DATA.PICKLE_SUBJECTS.cache_dir /home/ubuntu/storage/cache_dir \
OUTPUT_DIR /home/ubuntu/expr/brain_tumor/2023-12-28-g5/debug-001 \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 500 \
DATA.BATCH_SIZE 40 \
DATA.NUM_WORKERS 0 \
DATA.VAL_NUM_WORKERS 0 \
DATA.SPLITS.FOLD_INDEX 0 \
PATCH_BASED_TRAINER.QUEUE.max_length 40 \
PATCH_BASED_TRAINER.QUEUE.samples_per_volume 1 \
TRAINER.check_val_every_n_epoch 10 \
SOLVER.OneCycleLR.max_lr 0.004 \
DATA.PICKLE_SUBJECTS.cache True