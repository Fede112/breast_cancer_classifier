# packages: pip install pypng pydicom
# sometimes you need gdcm to uncompress some specific dicom format:
# conda install -c conda-forge gdcm 

# ds = pydicom.dcmread(dicom_filename)


import png
import pydicom
import os
import argparse
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


import sys

####
## Temporary addition to find src file (if called from bash: export PYTHONPATH=$(pwd):$PYTHONPATH)
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(sys.path)
# appending parent dir of current_dir to sys.path
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
# print(sys.path)
####

import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
import src.utilities.saving_images as saving_images
import src.utilities.data_handling as data_handling



def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth=12):
    """
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param dicom_filename: path to input dicom file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
    """
    image = pydicom.read_file(dicom_filename).pixel_array
    with open(png_filename, 'wb') as f:
        writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
        writer.write(f, image.tolist())
  

def dicom_to_png(dcm_full_path, output_directory, bitdepth = 12):
    ds = pydicom.dcmread(dcm_full_path)
    image = ds.pixel_array
    exam_id = ds.PatientID
    laterality = ds.ImageLaterality
    view_angle= ds.ViewPosition
    label = exam_id + '_' + laterality + '_' + view_angle
    full_png_path = os.path.join(output_directory, label + '.png')


    with open(full_png_path, 'wb') as f:
        writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
        writer.write(f, image.tolist())



def get_dicom_files(dir_name):
    """
    returns a list with the full path of each dcm file in the directory tree 
    """
    # create a list of file and sub directories in the given directory 
    file_list = os.listdir(dir_name)
    dcm_files = []
    # Iterate over the entries
    for entry in file_list:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(full_path):
            dcm_files = dcm_files + get_dicom_files(full_path)
        # If entry is a file that endswith .dcm, append to dcm_list
        elif entry.endswith(".dcm"):
            dcm_files.append(full_path)
                
    return dcm_files


def add_dcm_extension(dir_name):
    """
    Auxiliary funciton. Adds '.dcm' extension to all files inside dir_name.
    """
    file_list = os.listdir(dir_name)
    for entry in file_list:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            add_dcm_extension(full_path)
        elif not entry.endswith(".dcm"):
            os.rename(full_path, full_path + ".dcm")


def exams_info(exam_list):
    # check exam_err_list
    views = {'L-CC': 0, 'R-CC': 0, 'L-MLO': 0, 'R-MLO': 0}
    for exam_dict in exam_list:
        for k,v in exam_dict.items():
            if k in views:
                views[k] += len(v) 
    return views








def cro_dicom_scrapper(input_directory, output_directory, bitdepth = 12, generate_png = False, num_processes = 2):
    """
    It finds all the .dcm images inside the input_directory and creates the corresponding metadata and .png. 
    The structure of this directory should be as follows:
    input_directory/exam1/ ; input_directory/exam2/ ; input_directory/exam3/ 
    Each subdirectory corresponds to a single CRO exam.
    It converts dcm_to_png in parallel
    :param input_directory: path to the directory containig CRO exams
    :param output_directory: path to the directory where to put the images and metadata

    """
    # Create folder were to store .pngs and metadata
    if generate_png:
        # Check if folder exists
        if os.path.exists(output_directory):
            # Prevent overwriting to an existing directory
            print("Error: the directory to save png images already exists.")
            return
        else:
            os.makedirs(output_directory)
            images_dir = os.path.join(output_directory, 'images')
            os.makedirs(images_dir)


    # list of files and subdirectories inside input_directory
    file_list = os.listdir(input_directory) # other alternative was os.walk
    
    # list of exams with errors in the images
    exam_err_list = []

    # list of exams with more or less than 4 images (one per view)
    exam_not_four_list = []
    
    # exam_list compatible with the one required by NYU's 2019 algorithm
    exam_list = []
    
    # dcm path list. Used for dcm_to_png in parallel
    dcm_path_list = []

    for entry in tqdm(file_list):
        # print (f'entry: {entry}')
        full_path = os.path.join(input_directory, entry)
        # if the entry is a directory we asume it is an exam and look for the dcm images inside.
        if os.path.isdir(full_path):
            exam_dcm_paths = get_dicom_files(full_path)
            exam_dict = {'horizontal_flip': '', 'L-CC': [], 'L-MLO': [], 'R-MLO': [], 'R-CC': []}
            exam_id = ''

            # If there are less or more than 4 images don't use it
            # In the future we should handle more than 4 images
            # if not len(exam_dcm_paths) == 4:
            #     exam_err_list.append(entry)
            #     continue

            for dcm_file in exam_dcm_paths:
                ds = pydicom.dcmread(dcm_file)
                horizontal_flip = ds.FieldOfViewHorizontalFlip
                laterality = ds.ImageLaterality
                view_angle= ds.ViewPosition

                # Check exam_id for all dcm
                if exam_id == '':
                    exam_id = ds.PatientID
                else:
                    # assert exam_id == ds.PatientID, "Exam id is not the same for all dcm images in folder " + entry
                    if not exam_id == ds.PatientID:
                        exam_err_list.append(entry)
                        break
                
                # Check horizontal_flip for all dcm
                if not exam_dict['horizontal_flip']:
                    exam_dict['horizontal_flip'] = horizontal_flip
                else:
                    # assert exam_dict['horizontal_flip'] == horizontal_flip , "Horizontal flip is not the same for every image in exam " + entry
                    if not exam_dict['horizontal_flip'] == horizontal_flip:
                        exam_err_list.append(entry)
                        break

                # Check if laterality and view_angle are as expected
                if laterality not in ['L', 'R']:
                    exam_err_list.append(entry)
                    break
                if view_angle not in ['CC', 'MLO']:
                    exam_err_list.append(entry)
                    break

                view = laterality + '-' + view_angle
                label = exam_id + '_' + laterality + '_' + view_angle
                exam_dict[view].append(label)

            else:
                for key in exam_dict.keys():
                    if not key:
                        exam_err_list.append(entry)
                        break
                if not len(exam_dcm_paths) == 4:
                    exam_dict["folder"] = entry
                    exam_dict["num_images"] = len(exam_dcm_paths)
                    exam_not_four_list.append(exam_dict)
                else:
                    exam_list.append(exam_dict)
                    dcm_path_list += exam_dcm_paths

    exam_list_path = os.path.join(output_directory, 'exam_list_before_cropping.pkl')
    pickling.pickle_to_file(exam_list_path, exam_list)
    print(f'Exams without errors: \n {len(exam_list)}')
    print(f'Exams with less/more Dicom images: \n {len(exam_not_four_list)}')
    print(f'Exams with error in Dicom images: \n {len(exam_err_list)}')
    # print(exam_not_four_list)


    # Generate pngs
    if generate_png:
        dicom_to_png_fix_out = partial(dicom_to_png, output_directory = images_dir)
        with Pool(num_processes) as pool:
            pool.map(dicom_to_png_fix_out, dcm_path_list)


    return exam_list, exam_not_four_list, exam_err_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform CRO DICOM images to PNG as downloaded from Ambra web interface.')
    parser.add_argument('-i', '--input-data-folder', help="Folder containing DICOM images. \n", required=True)
    parser.add_argument('-o', '--output-data-folder', help="Folder where PNG images will be stored. \n", required=True)
    args = parser.parse_args()


    exam_list, exam_not_four_list, exam_err_list = cro_dicom_scrapper(args.input_data_folder, args.output_data_folder, 12, False, 3)

    print(exams_info(exam_not_four_list))