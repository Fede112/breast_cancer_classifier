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

import os
from multiprocessing import Pool
import argparse
from functools import partial

import src.cropping.crop_mammogram as crop_mammogram
import src.utilities.pickling as pickling


def crop_single_mammogram(metadata_dict, mammogram_path, cropped_mammogram_path, 
                          num_iterations, buffer_size):
    """
    Crop a single mammogram image.

    metadata dict structure:
    metadata_dict = dict(
        short_file_path=None,
        horizontal_flip=horizontal_flip,
        full_view=view,
        side=view[0],
        view=view[2:],
    )
    """
    mammogram_path = os.path.join(mammogram_path, metadata_dict['short_file_path'] + '.png')
    cropped_mammogram_path = os.path.join(cropped_mammogram_path, metadata_dict['short_file_path'] + '.png')
    cropped_image_info = crop_mammogram.crop_mammogram_one_image(
        scan=metadata_dict,
        input_file_path=mammogram_path,
        output_file_path=cropped_mammogram_path,
        num_iterations=num_iterations,
        buffer_size=buffer_size,
    )
    metadata_dict["window_location"] = cropped_image_info[0]
    metadata_dict["rightmost_points"] = cropped_image_info[1]
    metadata_dict["bottommost_points"] = cropped_image_info[2]
    metadata_dict["distance_from_starting_side"] = cropped_image_info[3]
    return metadata_dict


def crop_all_single_mammogram(input_data_folder, exam_list_path, cropped_exam_list_path, output_data_folder,
                num_processes, num_iterations, buffer_size):
    
    # list of exams (one dictionary per exam)
    exam_single_list = pickling.unpickle_from_file(exam_list_path)

    if os.path.exists(output_data_folder):
        # Prevent overwriting to an existing directory
        print("Error: the directory to save cropped images already exists.")
        return
    else:
        os.makedirs(output_data_folder)

    crop_single_mammogram_func = partial(
        crop_single_mammogram,
        mammogram_path=input_data_folder,
        cropped_mammogram_path=output_data_folder,
        num_iterations=num_iterations,
        buffer_size=buffer_size,
    )

    with Pool(num_processes) as pool:
        exam_single_list = pool.map(crop_single_mammogram_func, exam_single_list)

    pickling.pickle_to_file(cropped_exam_list_path, exam_single_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove background of image and save cropped files for single view model')
    parser.add_argument('--input-data-folder', required=True)
    parser.add_argument('--exam-list-path', required=True)
    parser.add_argument('--cropped-exam-list-path', required=True)
    parser.add_argument('--output-data-folder', required=True)
    parser.add_argument('--num-processes', default=10, type=int)
    parser.add_argument('--num-iterations', default=100, type=int)
    parser.add_argument('--buffer-size', default=50, type=int)
    args = parser.parse_args()
    
    crop_all_single_mammogram(
        input_data_folder=args.input_data_folder, 
        exam_list_path=args.exam_list_path, 
        cropped_exam_list_path=args.cropped_exam_list_path, 
        output_data_folder=args.output_data_folder, 
        num_processes=args.num_processes,
        num_iterations=args.num_iterations,
        buffer_size=args.buffer_size,
    )