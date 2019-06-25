# Added extra code to run only one column (view) of the model 
# For the moment without data augmentation





"""
# Resnet Network for CC view (right or left share weights)
# To change the subnet check subnet_prefix variable

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




def run_sub_model(model, exam_list, parameters):
    """
    Returns predictions of image only model or image+heatmaps model. 
    Prediction for each exam is averaged for a given number of epochs.
    """
    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device(f"cuda:{parameters["gpu_number"]}")
    else:
        device = torch.device("cpu")
    sub_model = sub_model.to(device)
    # F: sets model in evaluation mode. It has an effect in certain modules: e.g. Dropout or BatchNorm Layers
    sub_model.eval()

    random_number_generator = np.random.RandomState(parameters["seed"])

    image_extension = ".hdf5" if parameters["use_hdf5"] else ".png"

    with torch.no_grad():
        predictions_ls = []
        for datum in tqdm.tqdm(exam_list):
            predictions_for_datum = []
            loaded_image_dict = {view: [] for view in VIEWS.LIST}
            loaded_heatmaps_dict = {view: [] for view in VIEWS.LIST}
            for view in VIEWS.LIST:
                # F: for one exam, all images of a specific view
                for short_file_path in datum[view]:
                    loaded_image = loading.load_image(
                        image_path=os.path.join(parameters["image_path"], short_file_path + image_extension),
                        view=view,
                        horizontal_flip=datum["horizontal_flip"],
                    )
                 
                    loaded_image_dict[view].append(loaded_image)
            print(f"length loaded_image: {len(loaded_image_dict)}")
            for data_batch in tools.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"]):
                print(f"num_epochs: {parameters['num_epochs']}")
                print(f"batch_size: {parameters['batch_size']}")
                tmp = tools.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"])
                print(f"partition_batch: {tmp}")
                batch_dict = {view: [] for view in VIEWS.LIST}
                for _ in data_batch:
                    for view in VIEWS.LIST:
                        image_index = 0
                        # F: they use different augmentation for each view
                        if parameters["augmentation"]:
                            image_index = random_number_generator.randint(low=0, high=len(datum[view]))
                        cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
                            image=loaded_image_dict[view][image_index], 
                            auxiliary_image=loaded_heatmaps_dict[view][image_index],
                            view=view,
                            best_center=datum["best_center"][view][image_index],
                            random_number_generator=random_number_generator,
                            augmentation=parameters["augmentation"],
                            max_crop_noise=parameters["max_crop_noise"],
                            max_crop_size_noise=parameters["max_crop_size_noise"],
                        )
                        # print(f"cropped_image: {image_index} of m in minibatch: {_} size: {cropped_image.shape}")

                        else:
                            # F: e.g. batch_dict[view][:,:,1] is the first heatmap 
                            batch_dict[view].append(np.concatenate([
                                cropped_image[:, :, np.newaxis],
                                cropped_heatmaps,
                            ], axis=2))

                        # print(f"batch_dict_view: {len(batch_dict[view])}")
                        # print(f"batch_img_size: {batch_dict[view][_].shape}")


                tensor_batch = {
                    # F: result of np.stack has one more dimension:
                    # F: 4 dimensions: batch_data_i, y_pixels, x_pixels, channels 
                    view: torch.tensor(np.stack(batch_dict[view])).permute(0, 3, 1, 2).to(device)
                    for view in VIEWS.LIST
                }


                # print(f"layer_names: {model.state_dict().keys()}")
                # Print model's state_dict
                output = model(tensor_batch)
                batch_predictions = compute_batch_predictions(output)
                print(f"batch_predictions: \n {batch_predictions}")
                print(len(batch_predictions.keys()))
                # F: they pick value 1, disregarding value 0 which is the complement of that (prob = 1) 
                pred_df = pd.DataFrame({k: v[:, 1] for k, v in batch_predictions.items()})
                pred_df.columns.names = ["label", "view_angle"]
                # print(f"pred_df.head: {pred_df.head()}")
                # F: complicated way of grouping by label and calculating the mean                
                predictions = pred_df.T.reset_index().groupby("label").mean().T[LABELS.LIST].values
                predictions_for_datum.append(predictions)
                print(f"predictions: {predictions}")
                exit()
            predictions_ls.append(np.mean(np.concatenate(predictions_for_datum, axis=0), axis=0))

    return np.array(predictions_ls) 



def load_run_save(model_path, data_path, output_path, parameters):
    """
    Outputs the last layer of a ResNet column
    """

    ## Definde sub_model
    # input_channels = 3 if parameters["use_heatmaps"] else 1
    input_channels = 1
    sub_model = models.ViewResNetV2(
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


    ## Load subset of pretrained model
    # sub_model keys
    sub_model_dict = sub_model.state_dict()
    # load full model pretrained dict
    pretrained_model_dict = torch.load(model_path)["model"]
    # subnet I want to extract from the full model
    subnet_prefix = 'four_view_resnet.cc.'

    # From pytorch discuss: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/16
    # 1. filter out unnecessary keys
    pretrained_model_dict = {k.replace(subnet_prefix,''): v for k, v in pretrained_model_dict.items() if k.replace(subnet_prefix,'') in sub_model_dict}
    # 2. overwrite entries in the existing state dict
    sub_model_dict.update(pretrained_model_dict) 
    # 3. load the new state dict
    sub_model.load_state_dict(sub_model_dict)
    

    ## Run sub_model   
    # exam_list = pickling.unpickle_from_file(data_path)
    predictions = run_sub_model(sub_model, exam_list, parameters)
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)


    print()
    print(f"parameters: {parameters}")
    exit()



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
    parser.add_argument('--use-hdf5', action="store_true")
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
        "use_hdf5": args.use_hdf5
        # "augmentation": args.use_augmentation,
        # "max_crop_noise": (100, 100),
        # "max_crop_size_noise": 100,
        # "use_heatmaps": args.use_heatmaps,
        # "heatmaps_path": args.heatmaps_path,
    }

    load_run_save(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        parameters=parameters,
    )


if __name__ == "__main__":
    # print("Aca!")
    main()
