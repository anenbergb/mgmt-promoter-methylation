#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python mgmt/tasks/train.py \
-c architectures/MultiRes-Basic.yml \
-c train-configs/patch-based-trainer-v2-concat-resample-2.0-crop-64.yml \
DATA.PICKLE_SUBJECTS.folder_path /home/ubuntu/storage/radiology-research-west-2/expr/brain_tumor/preprocess-subjects-v2/resample-2.0-crop-64 \
DATA.PICKLE_SUBJECTS.cache_dir /home/ubuntu/storage/cache_dir \
OUTPUT_DIR /home/ubuntu/expr/brain_tumor/2023-12-26/debug-001 \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 100 \
DATA.BATCH_SIZE 50 \
DATA.NUM_WORKERS 1 \
DATA.SPLITS.FOLD_INDEX 0 \
PATCH_BASED_TRAINER.QUEUE.max_length 200 \
PATCH_BASED_TRAINER.QUEUE.samples_per_volume 5  \
PATCH_BASED_TRAINER.QUEUE.verbose True \
TRAINER.limit_train_batches 0.2