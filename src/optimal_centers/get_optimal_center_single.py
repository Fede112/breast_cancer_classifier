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
Runs search_windows_and_centers.py and extract_centers.py in the same directory
"""
import os
from multiprocessing import Pool
import argparse
from functools import partial
import numpy as np

import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
import src.optimal_centers.get_optimal_centers as get_optimal_centers


def get_optimal_center_single(metadata, cropped_mammogram_path):
    """
    Get optimal center for single example
    """
    cropped_mammogram_path = os.path.join(cropped_mammogram_path, metadata['short_file_path'] + '.png')
    image = reading_images.read_image_png(cropped_mammogram_path)
    optimal_center = get_optimal_centers.extract_center(metadata, image)
    metadata["best_center"] = optimal_center
    # pickling.pickle_to_file(metadata_path, metadata)
    return metadata


def get_all_optimal_center_single(cropped_exam_list_path, cropped_image_path, output_exam_list_path, num_processes=1):
    exam_single_list = pickling.unpickle_from_file(cropped_exam_list_path)

    get_optimal_center_single_func = partial(
        get_optimal_center_single,
        cropped_mammogram_path=cropped_image_path,
    )

    with Pool(num_processes) as pool:
        exam_single_list = pool.map(get_optimal_center_single_func, exam_single_list)

    os.makedirs(os.path.dirname(output_exam_list_path), exist_ok=True)
    pickling.pickle_to_file(output_exam_list_path, exam_single_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute and Extract Optimal Centers for single view model')
    parser.add_argument('--cropped-exam-list-path')
    parser.add_argument('--cropped-image-path')
    parser.add_argument('--output-exam-list-path', required=True)
    parser.add_argument('--num-processes', default=10, type=int)
    args = parser.parse_args()

    get_all_optimal_center_single(
        cropped_exam_list_path=args.cropped_exam_list_path,
        cropped_image_path=args.cropped_image_path,
        output_exam_list_path=args.output_exam_list_path,
        num_processes=args.num_processes
    )



# def main(cropped_exam_list_path, data_prefix, output_exam_list_path, num_processes=1):
#     exam_list = pickling.unpickle_from_file(cropped_exam_list_path)
#     data_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)
#     optimal_centers = get_optimal_centers(
#         data_list=data_list,
#         data_prefix=data_prefix,
#         num_processes=num_processes
#     )
#     data_handling.add_metadata(exam_list, "best_center", optimal_centers)
#     os.makedirs(os.path.dirname(output_exam_list_path), exist_ok=True)
#     pickling.pickle_to_file(output_exam_list_path, exam_list)
