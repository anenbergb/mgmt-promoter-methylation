#!/bin/bash

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-07/try1


python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run1 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[40, 40, 4]"

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run2 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[48, 48, 8]"

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run3 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[56, 56, 8]"

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run4 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[64, 64, 8]"

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run5 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[48, 48, 16]"

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run6 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[56, 56, 16]"

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run7 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[64, 64, 16]"

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-07/try2


python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run9 \
TRAINER.max_epochs 500 \
DATA.MODALITY concat \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 10

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run10 \
TRAINER.max_epochs 500 \
DATA.MODALITY concat \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 10

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run11 \
TRAINER.max_epochs 500 \
DATA.MODALITY concat \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 10

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run12 \
TRAINER.max_epochs 500 \
DATA.MODALITY concat \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 10

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run1 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 10

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run2 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 10

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run3 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 10

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run4 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 10

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run5 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 11

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run6 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 11

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run7 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 11

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run8 \
TRAINER.max_epochs 500 \
DATA.MODALITY t1w \
DATA.CROP_DIM "[40, 40, 8]" \
DATA.TRAIN_VAL_MANUAL_SEED 11