#!/bin/bash

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/debug1

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-d.yml \
# -c train-configs/patch-based-trainer-1.yml \
# OUTPUT_DIR "${EXPR_DIR}/run1" \
# SEED_EVERYTHING 1 \
# MODEL.EfficientNet.image_size 64 \
# TRAINER.max_epochs 10

# python mgmt/tasks/train.py \
# -c architectures/ResNet-a.yml \
# -c train-configs/patch-based-trainer-concat-1.yml \
# OUTPUT_DIR /home/bryan/expr/brain_tumor/2023-08-09/debug/concat-run3 \
# SEED_EVERYTHING 1 \
# TRAINER.max_epochs 2

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/resnet-t1c-200

# for i in {1..5}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/ResNet-a.yml \
#     -c train-configs/patch-based-trainer-1.yml \
#     OUTPUT_DIR "${EXPR_DIR}/run-${i}" \
#     SEED_EVERYTHING $i \
#     TRAINER.max_epochs 200
# done

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/resnet-t1c-1000

# for i in {1..5}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/ResNet-a.yml \
#     -c train-configs/patch-based-trainer-1.yml \
#     OUTPUT_DIR "${EXPR_DIR}/run-${i}" \
#     SEED_EVERYTHING $i \
#     TRAINER.max_epochs 1000
# done


# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/resnet-concat-100

# for i in {1..3}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/ResNet-a.yml \
#     -c train-configs/patch-based-trainer-concat-1.yml \
#     OUTPUT_DIR "${EXPR_DIR}/run-${i}" \
#     SEED_EVERYTHING $i \
#     TRAINER.max_epochs 100
# done

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/resnet-t1c-100

# for i in {1..2}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/ResNet-a.yml \
#     -c train-configs/patch-based-trainer-1.yml \
#     OUTPUT_DIR "${EXPR_DIR}/run-${i}" \
#     SEED_EVERYTHING $i \
#     TRAINER.max_epochs 100
# done


# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/resnet-concat-200

# python mgmt/tasks/train.py \
# -c architectures/ResNet-a.yml \
# -c train-configs/patch-based-trainer-concat-1.yml \
# OUTPUT_DIR "${EXPR_DIR}/run-10" \
# SEED_EVERYTHING 10 \
# TRAINER.max_epochs 200

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/resnet-fla-100

# for i in {1..2}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/ResNet-a.yml \
#     -c train-configs/patch-based-trainer-fla.yml \
#     OUTPUT_DIR "${EXPR_DIR}/run-${i}" \
#     SEED_EVERYTHING $i \
#     TRAINER.max_epochs 100
# done

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/resnet-t1w-100

# for i in {1..2}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/ResNet-a.yml \
#     -c train-configs/patch-based-trainer-t1w.yml \
#     OUTPUT_DIR "${EXPR_DIR}/run-${i}" \
#     SEED_EVERYTHING $i \
#     TRAINER.max_epochs 100
# done

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/resnet-t2w-100

# for i in {1..2}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/ResNet-a.yml \
#     -c train-configs/patch-based-trainer-t2w.yml \
#     OUTPUT_DIR "${EXPR_DIR}/run-${i}" \
#     SEED_EVERYTHING $i \
#     TRAINER.max_epochs 100
# done


# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/try2/resnet-concat-200

# python mgmt/tasks/train.py \
# -c architectures/ResNet-a.yml \
# -c train-configs/patch-based-trainer-concat-1.yml \
# OUTPUT_DIR "${EXPR_DIR}/seed-1" \
# SEED_EVERYTHING 1 \
# TRAINER.max_epochs 200

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/try2/resnet-concat-100-no-resample

for i in {1..2}
do
    python mgmt/tasks/train.py \
    -c architectures/ResNet-a.yml \
    -c train-configs/patch-based-trainer-concat-crop-64.yml \
    OUTPUT_DIR "${EXPR_DIR}/run-${i}" \
    SEED_EVERYTHING $i \
    TRAINER.max_epochs 100
done

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/try2/resnet-t1c-100-no-resample

# for i in {1..2}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/ResNet-a.yml \
#     -c train-configs/patch-based-trainer-t1c-crop-64.yml \
#     OUTPUT_DIR "${EXPR_DIR}/run-${i}" \
#     SEED_EVERYTHING $i \
#     TRAINER.max_epochs 100
# done

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-09/try3/resnet-concat-200

python mgmt/tasks/train.py \
-c architectures/ResNet-a.yml \
-c train-configs/patch-based-trainer-concat-1.yml \
OUTPUT_DIR "${EXPR_DIR}/seed-1" \
SEED_EVERYTHING 1 \
TRAINER.max_epochs 200