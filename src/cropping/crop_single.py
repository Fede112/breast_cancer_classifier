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

import argparse

import src.cropping.crop_mammogram as crop_mammogram
import src.utilities.pickling as pickling



def crop_mammogram(input_data_folder, exam_list_path, cropped_exam_list_path, output_data_folder,
                   num_processes, num_iterations, buffer_size):
    """
    In parallel, crops mammograms in DICOM format found in input_data_folder and save as png format in
    output_data_folder and saves new image list in cropped_image_list_path
    """

    # list of exams (one dictionary per exam)
    exam_list = pickling.unpickle_from_file(exam_list_path)
    
    # list per image (one dictionary per image). It contains same information than in list of exams + cropped information if present.
    image_list = data_handling.unpack_exam_into_images(exam_list)
    

    if os.path.exists(output_data_folder):
        # Prevent overwriting to an existing directory
        print("Error: the directory to save cropped images already exists.")
        return
    else:
        os.makedirs(output_data_folder)

    crop_mammogram_one_image_func = partial(
        crop_mammogram_one_image_short_path,
        input_data_folder=input_data_folder, 
        output_data_folder=output_data_folder,
        num_iterations=num_iterations,
        buffer_size=buffer_size,
    )
    with Pool(num_processes) as pool:
        cropped_image_info = pool.map(crop_mammogram_one_image_func, image_list)
    # F: cropped image info returns a list. Each entry is the return of a single execution
    # F: of crop_mammogram_one_image_func.
    
    # F: each dict defined contains the information of all images
    window_location_dict = dict([x[0] for x in cropped_image_info])
    rightmost_points_dict = dict([x[1] for x in cropped_image_info])
    bottommost_points_dict = dict([x[2] for x in cropped_image_info])
    distance_from_starting_side_dict = dict([x[3] for x in cropped_image_info])

    data_handling.add_metadata(exam_list, "window_location", window_location_dict)
    data_handling.add_metadata(exam_list, "rightmost_points", rightmost_points_dict)
    data_handling.add_metadata(exam_list, "bottommost_points", bottommost_points_dict)
    data_handling.add_metadata(exam_list, "distance_from_starting_side", distance_from_starting_side_dict)
    
    pickling.pickle_to_file(cropped_exam_list_path, exam_list)


    


def crop_single_mammogram(mammogram_path, horizontal_flip, view,
                          cropped_mammogram_path, metadata_path,
                          num_iterations, buffer_size):
    """
    Crop a single mammogram image
    """
    metadata_dict = dict(
        short_file_path=None,
        horizontal_flip=horizontal_flip,
        full_view=view,
        side=view[0],
        view=view[2:],
    )
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
    pickling.pickle_to_file(metadata_path, metadata_dict)


def main():
    parser = argparse.ArgumentParser(description='Remove background of image and save cropped files')
    parser.add_argument('--mammogram-path', required=True)
    parser.add_argument('--view', required=True)
    parser.add_argument('--horizontal-flip', default="NO", type=str)
    parser.add_argument('--cropped-mammogram-path', required=True)
    parser.add_argument('--metadata-path', required=True)
    parser.add_argument('--num-iterations', default=100, type=int)
    parser.add_argument('--buffer-size', default=50, type=int)
    args = parser.parse_args()


    crop_single_mammogram(
        mammogram_path=args.mammogram_path,
        view=args.view,
        horizontal_flip=args.horizontal_flip,
        cropped_mammogram_path=args.cropped_mammogram_path,
        metadata_path=args.metadata_path,
        num_iterations=args.num_iterations,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()
