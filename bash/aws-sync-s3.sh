#!/bin/bash

aws s3 sync \
s3://radiology-research-west-2/expr/brain_tumor/preprocess-subjects-v2/resample-2.0-crop-64/ \
/home/ubuntu/storage/radiology-research-west-2/expr/brain_tumor/preprocess-subjects-v2/resample-2.0-crop-64