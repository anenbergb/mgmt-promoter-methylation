#!/bin/bash

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-16/try2


python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run1 \
TRAINER.max_epochs 2 \
DATA.MODALITY t1w \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
TRAINER.profiler simple

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run2 \
TRAINER.max_epochs 3 \
DATA.MODALITY t1w \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
TRAINER.profiler advanced

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run3 \
TRAINER.max_epochs 3 \
DATA.MODALITY concat \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
TRAINER.profiler advanced

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run4 \
TRAINER.max_epochs 3 \
DATA.MODALITY t1w \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
TRAINER.profiler advanced \
DATA.BATCH_SIZE 8 \
PREPROCESS.RESIZE.target_shape "[64, 64, 64]"

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run5 \
TRAINER.max_epochs 100 \
DATA.MODALITY t1w \
PREPROCESS.CROP_LARGEST_TUMOR.crop_dim None \
TRAINER.profiler simple
