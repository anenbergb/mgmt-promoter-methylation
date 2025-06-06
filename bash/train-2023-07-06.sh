#!/bin/bash

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-07-06/try2

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run1 \
TRAINER.max_epochs 500 \
SOLVER.SCHEDULER_NAME OneCycleLR \
SOLVER.OPTIMIZER_NAME AdamW \
DATA.MODALITY fla 

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run2 \
TRAINER.max_epochs 500 \
SOLVER.SCHEDULER_NAME OneCycleLR \
SOLVER.OPTIMIZER_NAME AdamW \
DATA.MODALITY t1w

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run3 \
TRAINER.max_epochs 500 \
SOLVER.SCHEDULER_NAME OneCycleLR \
SOLVER.OPTIMIZER_NAME AdamW \
DATA.MODALITY t1c

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run4  \
TRAINER.max_epochs 500 \
SOLVER.SCHEDULER_NAME OneCycleLR \
SOLVER.OPTIMIZER_NAME AdamW \
DATA.MODALITY t2w

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run5  \
TRAINER.max_epochs 500 \
SOLVER.SCHEDULER_NAME OneCycleLR \
SOLVER.OPTIMIZER_NAME Adam \
DATA.MODALITY t2w

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run6  \
TRAINER.max_epochs 500 \
SOLVER.SCHEDULER_NAME OneCycleLR \
SOLVER.OPTIMIZER_NAME NAdam \
DATA.MODALITY t2w

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run7  \
TRAINER.max_epochs 500 \
SOLVER.SCHEDULER_NAME OneCycleLR \
SOLVER.OPTIMIZER_NAME SGD \
DATA.MODALITY t2w

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run8  \
TRAINER.max_epochs 500 \
SOLVER.SCHEDULER_NAME ReduceLROnPlateau \
SOLVER.OPTIMIZER_NAME AdamW \
DATA.MODALITY t2w

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run8  \
TRAINER.max_epochs 500 \
SOLVER.SCHEDULER_NAME OneCycleLR \
SOLVER.OPTIMIZER_NAME AdamW \
DATA.MODALITY t2w \
SEED_EVERYTHING 43

python mgmt/tasks/train.py OUTPUT_DIR $EXPR_DIR/run9  \
TRAINER.max_epochs 500 \
SOLVER.SCHEDULER_NAME OneCycleLR \
SOLVER.OPTIMIZER_NAME AdamW \
DATA.MODALITY t2w \
SEED_EVERYTHING 43