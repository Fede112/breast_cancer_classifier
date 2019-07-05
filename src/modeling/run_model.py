# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
"""
Runs the image only model and image+heatmaps model for breast cancer prediction.
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
import src.data_loading.loading as loading
from src.constants import VIEWS, VIEWANGLES, LABELS


def run_model(model, exam_list, parameters):
    """
    Returns predictions of image only model or image+heatmaps model. 
    Prediction for each exam is averaged for a given number of epochs.
    """
    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    # F: sets model in evaluation mode. It has an effect in certain modules: e.g. Dropout or BatchNorm Layers
    model.eval()

    random_number_generator = np.random.RandomState(parameters["seed"])

    image_extension = ".hdf5" if parameters["use_hdf5"] else ".png"

    with torch.no_grad():
        predictions_ls = []
        for datum in tqdm.tqdm(exam_list):
            predictions_for_datum = []
            # F: VIEWS is an adhoc class
            # F: VIEWS.LIST : list of views as string
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
                    if parameters["use_heatmaps"]:
                        loaded_heatmaps = loading.load_heatmaps(
                            benign_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_benign",
                                                             short_file_path + ".hdf5"),
                            malignant_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_malignant",
                                                                short_file_path + ".hdf5"),
                            view=view,
                            horizontal_flip=datum["horizontal_flip"],
                        )
                    else:
                        loaded_heatmaps = None

                    loaded_image_dict[view].append(loaded_image)
                    loaded_heatmaps_dict[view].append(loaded_heatmaps)
            # print(f"length loaded_image: {len(loaded_image_dict)}")
            for data_batch in tools.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"]):
                # print(f"num_epochs: {parameters['num_epochs']}")
                # print(f"batch_size: {parameters['batch_size']}")
                tmp = tools.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"])
                # print(f"partition_batch: {tmp}")
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


                        if loaded_heatmaps_dict[view][image_index] is None:
                            batch_dict[view].append(cropped_image[:, :, np.newaxis])
                            # F: e.g. batch_dict[view][_].shape = (2974, 1748, 1)

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
                # print("Model's state_dict:")
                # for param_tensor in model.state_dict():
                    # print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                output = model(tensor_batch)
                batch_predictions = compute_batch_predictions(output)
                # print(f"batch_predictions: \n {batch_predictions}")
                # print(len(batch_predictions.keys()))
                # F: they pick value 1, disregarding value 0 which is the complement of that (prob = 1) 
                pred_df = pd.DataFrame({k: v[:, 1] for k, v in batch_predictions.items()})
                pred_df.columns.names = ["label", "view_angle"]
                # print(f"pred_df.head: {pred_df.head()}")
                # F: complicated way of grouping by label and calculating the mean                
                predictions = pred_df.T.reset_index().groupby("label").mean().T[LABELS.LIST].values
                predictions_for_datum.append(predictions)
                # print(f"predictions: {predictions}")
            predictions_ls.append(np.mean(np.concatenate(predictions_for_datum, axis=0), axis=0))

    return np.array(predictions_ls) 


def compute_batch_predictions(y_hat):
    """
    Format predictions from different heads
    """
    batch_prediction_dict = col.OrderedDict([
        ( (label_name, view_angle), np.exp(y_hat[view_angle][:, i].cpu().detach().numpy()) )
        for i, label_name in enumerate(LABELS.LIST)
        for view_angle in VIEWANGLES.LIST
    ])
    return batch_prediction_dict


def load_run_save(model_path, data_path, output_path, parameters):
    """
    Outputs the predictions as csv file
    """
    input_channels = 3 if parameters["use_heatmaps"] else 1
    model = models.SplitBreastModel(input_channels)
    model.load_state_dict(torch.load(model_path)["model"])
    exam_list = pickling.unpickle_from_file(data_path)
    predictions = run_model(model, exam_list, parameters)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Take the positive prediction
    # F: actually we took it already
    df = pd.DataFrame(predictions, columns=LABELS.LIST)
    df.to_csv(output_path, index=False, float_format='%.4f')


def main():
    parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use-heatmaps', action="store_true")
    parser.add_argument('--heatmaps-path')
    parser.add_argument('--use-augmentation', action="store_true")
    parser.add_argument('--use-hdf5', action="store_true")
    parser.add_argument('--num-epochs', default=1, type=int)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    args = parser.parse_args()

    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "augmentation": args.use_augmentation,
        "num_epochs": args.num_epochs,
        "use_heatmaps": args.use_heatmaps,
        "heatmaps_path": args.heatmaps_path,
        "use_hdf5": args.use_hdf5
    }

    exam_list = pickling.unpickle_from_file(args.data_path)

    load_run_save(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        parameters=parameters,
    )


if __name__ == "__main__":
    main()
