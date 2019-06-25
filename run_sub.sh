#!/bin/bash

NUM_PROCESSES=10
DEVICE_TYPE='cpu'
NUM_EPOCHS=10
HEATMAP_BATCH_SIZE=100
GPU_NUMBER=0

DATA_FOLDER='sample_data/images'
INITIAL_EXAM_LIST_PATH='sample_data/exam_list_before_cropping.pkl'
PATCH_MODEL_PATH='models/sample_patch_model.p'
IMAGE_MODEL_PATH='models/sample_image_model.p'
IMAGEHEATMAPS_MODEL_PATH='models/sample_imageheatmaps_model.p'

CROPPED_IMAGE_PATH='sample_output/cropped_images'
CROPPED_EXAM_LIST_PATH='sample_output/cropped_images/cropped_exam_list.pkl'
EXAM_LIST_PATH='sample_output/data.pkl'
HEATMAPS_PATH='sample_output/heatmaps'
IMAGE_PREDICTIONS_PATH='sample_output/image_predictions.csv'
IMAGEHEATMAPS_PREDICTIONS_PATH='sample_output/imageheatmaps_predictions.csv'
export PYTHONPATH=$(pwd):$PYTHONPATH


echo 'Run CC ViewResNetV2 (Image)'
python3 src/modeling/run_sub_model.py \
    --model-path $IMAGE_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGE_PREDICTIONS_PATH \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --batch-size 2
    # --use-augmentation \