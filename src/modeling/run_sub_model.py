# Added extra code to run only one column (view) of the model 
# For the moment without data augmentation





"""
# Resnet Network for CC view right (its the same right or left):

# Resnet 22: 5 resnet layers with two blocks each.
# 16 -> 32 -> 64 -> 128 -> 256
"""

import argparse
import collections as col
import numpy as np
import os
import pandas as pd
import torch
import tqdm

import src.utilities.pickling as pickling
import src.utilities.tools as tools
import src.modeling.models as models
import src.modeling.layers as layers
import src.data_loading.loading as loading
from src.constants import VIEWS, VIEWANGLES, LABELS




def load_run_save(model_path, data_path, output_path, parameters):
    """
    Outputs the last layer of a ResNet column
    """

    # input_channels = 3 if parameters["use_heatmaps"] else 1
    input_channels = 1
    model = models.ViewResNetV2(
                input_channels=input_channels, 
                num_filters=16,
                first_layer_kernel_size=7, 
                first_layer_conv_stride=2,
                first_layer_padding=0,
                first_pool_size=3, 
                first_pool_stride=2, 
                first_pool_padding=0,
                blocks_per_layer_list=[2, 2, 2, 2, 2], 
                block_strides_list=[1, 2, 2, 2, 2], 
                block_fn=layers.BasicBlockV2,
                growth_factor=2
            )

    # load full model weights
    trained_model = torch.load(model_path)["model"]
    # sub model keys
    submodel_dict = submodel.state_dict()


    # From pytorch discuss: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/16
    # 1. filter out unnecessary keys
    trained_submodel = {k: v for k, v in pretrained_dict.items() if k in submodel_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(trained_submodel) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # exam_list = pickling.unpickle_from_file(data_path)
    predictions = run_model(model, exam_list, parameters)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)






def main():
    parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--num-epochs', default=1, type=int)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--use-hdf5', action="store_true")
    # parser.add_argument('--use-heatmaps', action="store_true")
    # parser.add_argument('--heatmaps-path')
    # parser.add_argument('--use-augmentation', action="store_true")
    args = parser.parse_args()

    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "image_path": args.image_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        # "augmentation": args.use_augmentation,
        # "max_crop_noise": (100, 100),
        # "max_crop_size_noise": 100,
        # "use_heatmaps": args.use_heatmaps,
        # "heatmaps_path": args.heatmaps_path,
        # "use_hdf5": args.use_hdf5
    }

    load_run_save(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        parameters=parameters,
    )


if __name__ == "__main__":
    print("Aca!")
    # main()
