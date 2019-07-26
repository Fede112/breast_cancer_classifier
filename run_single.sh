#!/bin/bash

NUM_PROCESSES=3
DEVICE_TYPE='cpu'
NUM_EPOCHS=4
HEATMAP_BATCH_SIZE=100
GPU_NUMBER=0

PATCH_MODEL_PATH='models/sample_patch_model.p'
IMAGE_MODEL_PATH='models/ImageOnly__ModeImage_weights.p'
IMAGEHEATMAPS_MODEL_PATH='models/ImageHeatmaps__ModeImage_weights.p'

# SAMPLE_SINGLE_OUTPUT_PATH='sample_single_output'

# initial files
INITIAL_EXAM_LIST_PATH='sample_data/exam_single_list_before_cropping.pkl'
DATA_FOLDER='sample_data/images'

#cropped files
CROPPED_EXAM_LIST_PATH='sample_single_output/cropped_images/cropped_exam_single_list.pkl'
CROPPED_IMAGE_PATH='sample_single_output/cropped_images'
EXAM_LIST_PATH='sample_single_output/data.pkl'

# output files
IMAGE_PREDICTIONS_PATH='sample_single_output/image_predictions.csv'
# IMAGEHEATMAPS_PREDICTIONS_PATH='sample_output/imageheatmaps_predictions.csv'




export PYTHONPATH=$(pwd):$PYTHONPATH


echo 'Stage 1: Crop Mammograms full'
python3 src/cropping/crop_single.py \
    --input-data-folder $DATA_FOLDER \
    --exam-list-path $INITIAL_EXAM_LIST_PATH  \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH  \
    --output-data-folder $CROPPED_IMAGE_PATH \
    --num-processes $NUM_PROCESSES

echo 'Stage 2: Extract Centers'
python3 src/optimal_centers/get_optimal_center_single.py \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
    --cropped-image-path $CROPPED_IMAGE_PATH \
    --output-exam-list-path $EXAM_LIST_PATH \
    --num-processes $NUM_PROCESSES


# echo 'Stage 3: Generate Heatmaps'
# python3 src/heatmaps/run_producer_single.py \
#     --model-path ${PATCH_MODEL_PATH} \
#     --cropped-mammogram-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped.png \
#     --metadata-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped_metadata.pkl \
#     --batch-size ${HEATMAP_BATCH_SIZE} \
#     --heatmap-path-malignant ${SAMPLE_SINGLE_OUTPUT_PATH}/malignant_heatmap.hdf5 \
#     --heatmap-path-benign ${SAMPLE_SINGLE_OUTPUT_PATH}/benign_heatmap.hdf5\
#     --device-type ${DEVICE_TYPE} \
#     --gpu-number ${GPU_NUMBER}

# echo 'Stage 4a: Run Classifier (Image)'
# python3 src/modeling/run_model_single.py \
#     --view $2 \
#     --model-path ${IMAGE_MODEL_PATH} \
#     --cropped-mammogram-path sample_single_output/cropped.png \
#     --metadata-path sample_single_output/cropped_metadata.pkl \
#     --use-augmentation \
#     --num-epochs ${NUM_EPOCHS} \
#     --device-type ${DEVICE_TYPE} \
#     --gpu-number ${GPU_NUMBER} \
#     --batch-size 2

echo 'Stage 4a: Run Classifier (Image)'
python3 src/modeling/run_model_single.py \
	--view 'L-CC' \
    --model-path $IMAGE_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --cropped-mammogram-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGE_PREDICTIONS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --batch-size 2

# echo 'Stage 4b: Run Classifier (Image+Heatmaps)'
# python3 src/modeling/run_model_single.py \
#     --view $2 \
#     --model-path ${IMAGEHEATMAPS_MODEL_PATH} \
#     --cropped-mammogram-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped.png \
#     --metadata-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped_metadata.pkl \
#     --use-heatmaps \
#     --heatmap-path-malignant ${SAMPLE_SINGLE_OUTPUT_PATH}/malignant_heatmap.hdf5 \
#     --heatmap-path-benign ${SAMPLE_SINGLE_OUTPUT_PATH}/benign_heatmap.hdf5\
#     --use-augmentation \
#     --num-epochs ${NUM_EPOCHS} \
#     --device-type ${DEVICE_TYPE} \
#     --gpu-number ${GPU_NUMBER}
