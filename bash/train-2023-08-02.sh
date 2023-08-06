#!/bin/bash

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-02/try2

# for i in {1..20}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/EfficientNet-d.yml \
#     OUTPUT_DIR "${EXPR_DIR}/efficient-d-64-seed-${i}" \
#     SEED_EVERYTHING $i \
#     MODEL.EfficientNet.image_size 64 \
#     TRAINER.max_epochs 10 \
#     DATA.MODALITY t1c \
#     PATCH_BASED_TRAINER.ENABLED True \
#     PATCH_BASED_TRAINER.LABEL_SAMPLER.patch_size "[64, 64, 64]" \
#     PREPROCESS.EARLY_CROP_LARGEST_TUMOR_ENABLED True \
#     PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
#     PREPROCESS.CROP_LARGEST_TUMOR_ENABLED False \
#     PREPROCESS.RESIZE_ENABLED False \
#     AUGMENT.RANDOM_GAMMA_ENABLED False \
#     AUGMENT.RANDOM_NOISE_ENABLED False \
#     AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
#     AUGMENT.RANDOM_MOTION_ENABLED False
# done


# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-02/try3

# for i in {1..5}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/EfficientNet-d.yml \
#     OUTPUT_DIR "${EXPR_DIR}/efficient-d-64-seed-${i}" \
#     SEED_EVERYTHING $i \
#     MODEL.EfficientNet.image_size 64 \
#     TRAINER.max_epochs 10 \
#     DATA.MODALITY t1c \
#     PATCH_BASED_TRAINER.ENABLED True \
#     PATCH_BASED_TRAINER.LABEL_SAMPLER.patch_size "[64, 64, 64]" \
#     PATCH_BASED_TRAINER.QUEUE.samples_per_volume 75 \
#     PREPROCESS.EARLY_CROP_LARGEST_TUMOR_ENABLED True \
#     PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
#     PREPROCESS.CROP_LARGEST_TUMOR_ENABLED False \
#     PREPROCESS.RESIZE_ENABLED False \
#     AUGMENT.RANDOM_GAMMA_ENABLED False \
#     AUGMENT.RANDOM_NOISE_ENABLED False \
#     AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
#     AUGMENT.RANDOM_MOTION_ENABLED False
# done

# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-02/try4-multistep

# for i in {1..5}
# do
#     python mgmt/tasks/train.py \
#     -c architectures/EfficientNet-d.yml \
#     OUTPUT_DIR "${EXPR_DIR}/efficient-d-64-seed-${i}" \
#     SEED_EVERYTHING $i \
#     MODEL.EfficientNet.image_size 64 \
#     TRAINER.max_epochs 30 \
#     DATA.MODALITY t1c \
#     PATCH_BASED_TRAINER.ENABLED True \
#     PATCH_BASED_TRAINER.LABEL_SAMPLER.patch_size "[64, 64, 64]" \
#     PATCH_BASED_TRAINER.QUEUE.samples_per_volume 25 \
#     PATCH_BASED_TRAINER.QUEUE.max_length 3100 \
#     PREPROCESS.EARLY_CROP_LARGEST_TUMOR_ENABLED True \
#     PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
#     PREPROCESS.CROP_LARGEST_TUMOR_ENABLED False \
#     PREPROCESS.RESIZE_ENABLED False \
#     AUGMENT.RANDOM_GAMMA_ENABLED False \
#     AUGMENT.RANDOM_NOISE_ENABLED False \
#     AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
#     AUGMENT.RANDOM_MOTION_ENABLED False \
#     SOLVER.SCHEDULER_NAME MultiStepLR \
#     SOLVER.MultiStepLR.milestones "[10, 20, 25]" \
#     SOLVER.WARMUP.ENABLED True \
#     SOLVER.WARMUP.warmup_steps 3
# done


# EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-02/debug-1

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-d.yml \
# OUTPUT_DIR "${EXPR_DIR}/run1" \
# SEED_EVERYTHING 10 \
# MODEL.EfficientNet.image_size 64 \
# TRAINER.max_epochs 30 \
# DATA.MODALITY t1c \
# PATCH_BASED_TRAINER.ENABLED False \
# PATCH_BASED_TRAINER.LABEL_SAMPLER.patch_size "[64, 64, 64]" \
# PATCH_BASED_TRAINER.QUEUE.samples_per_volume 25 \
# PATCH_BASED_TRAINER.QUEUE.max_length 3100 \
# PREPROCESS.EARLY_CROP_LARGEST_TUMOR_ENABLED True \
# PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
# PREPROCESS.CROP_LARGEST_TUMOR_ENABLED False \
# PREPROCESS.RESIZE_ENABLED False \
# AUGMENT.RANDOM_GAMMA_ENABLED False \
# AUGMENT.RANDOM_NOISE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# AUGMENT.RANDOM_MOTION_ENABLED False \
# SOLVER.SCHEDULER_NAME MultiStepLR \
# SOLVER.MultiStepLR.milestones "[10, 20, 25]" \
# SOLVER.WARMUP.ENABLED True \
# SOLVER.WARMUP.warmup_steps 3 \
# DATA.BATCH_SIZE 2 \
# TRAINER.limit_train_batches 10.0 \
# TRAINER.limit_val_batches 2.0

# python mgmt/tasks/train.py \
# -c architectures/EfficientNet-d.yml \
# OUTPUT_DIR "${EXPR_DIR}/run2" \
# SEED_EVERYTHING 10 \
# MODEL.EfficientNet.image_size 64 \
# TRAINER.max_epochs 30 \
# DATA.MODALITY t1c \
# PATCH_BASED_TRAINER.ENABLED False \
# PATCH_BASED_TRAINER.LABEL_SAMPLER.patch_size "[64, 64, 64]" \
# PATCH_BASED_TRAINER.QUEUE.samples_per_volume 25 \
# PATCH_BASED_TRAINER.QUEUE.max_length 3100 \
# PREPROCESS.EARLY_CROP_LARGEST_TUMOR_ENABLED True \
# PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
# PREPROCESS.CROP_LARGEST_TUMOR_ENABLED False \
# PREPROCESS.RESIZE_ENABLED False \
# AUGMENT.RANDOM_GAMMA_ENABLED False \
# AUGMENT.RANDOM_NOISE_ENABLED False \
# AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
# AUGMENT.RANDOM_MOTION_ENABLED False \
# SOLVER.SCHEDULER_NAME StepLR \
# SOLVER.WARMUP.ENABLED True \
# SOLVER.WARMUP.warmup_steps 3 \
# DATA.BATCH_SIZE 2 \
# TRAINER.limit_train_batches 10.0 \
# TRAINER.limit_val_batches 2.0


EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-02/try5

for i in {5..10}
do
    python mgmt/tasks/train.py \
    -c architectures/EfficientNet-d.yml \
    OUTPUT_DIR "${EXPR_DIR}/one-cycle-linear-seed-${i}" \
    SEED_EVERYTHING $i \
    MODEL.EfficientNet.image_size 64 \
    TRAINER.max_epochs 10 \
    DATA.MODALITY t1c \
    PATCH_BASED_TRAINER.ENABLED True \
    PATCH_BASED_TRAINER.LABEL_SAMPLER.patch_size "[64, 64, 64]" \
    PATCH_BASED_TRAINER.QUEUE.samples_per_volume 75 \
    PATCH_BASED_TRAINER.QUEUE.max_length 4100 \
    PREPROCESS.EARLY_CROP_LARGEST_TUMOR_ENABLED True \
    PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
    PREPROCESS.CROP_LARGEST_TUMOR_ENABLED False \
    PREPROCESS.RESIZE_ENABLED False \
    AUGMENT.RANDOM_GAMMA_ENABLED False \
    AUGMENT.RANDOM_NOISE_ENABLED False \
    AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
    AUGMENT.RANDOM_MOTION_ENABLED False \
    SOLVER.OneCycleLR.anneal_strategy linear
done

EXPR_DIR=/home/bryan/expr/brain_tumor/2023-08-02/try6

for i in {10..15}
do
    python mgmt/tasks/train.py \
    -c architectures/EfficientNet-d.yml \
    OUTPUT_DIR "${EXPR_DIR}/one-cycle-cosine-seed-${i}" \
    SEED_EVERYTHING $i \
    MODEL.EfficientNet.image_size 64 \
    TRAINER.max_epochs 10 \
    DATA.MODALITY t1c \
    PATCH_BASED_TRAINER.ENABLED True \
    PATCH_BASED_TRAINER.LABEL_SAMPLER.patch_size "[64, 64, 64]" \
    PATCH_BASED_TRAINER.QUEUE.samples_per_volume 75 \
    PATCH_BASED_TRAINER.QUEUE.max_length 2500 \
    PREPROCESS.EARLY_CROP_LARGEST_TUMOR_ENABLED True \
    PREPROCESS.EARLY_CROP_LARGEST_TUMOR.crop_dim "[72, 72, 72]" \
    PREPROCESS.CROP_LARGEST_TUMOR_ENABLED False \
    PREPROCESS.RESIZE_ENABLED False \
    AUGMENT.RANDOM_GAMMA_ENABLED False \
    AUGMENT.RANDOM_NOISE_ENABLED False \
    AUGMENT.RANDOM_BIAS_FIELD_ENABLED False \
    AUGMENT.RANDOM_MOTION_ENABLED False
done