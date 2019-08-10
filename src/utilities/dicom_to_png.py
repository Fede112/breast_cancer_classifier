# packages: pip install pypng pydicom
# sometimes you need gdcm to uncompress some specific dicom format:
# conda install -c conda-forge gdcm 

import png
import pydicom
import os
import argparse
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np


import sys

####
## Temporary addition to find src file (if called from bash: export PYTHONPATH=$(pwd):$PYTHONPATH)
current_dir = os.path.dirname(os.path.abspath(__file__))
# appending parent dir of current_dir to sys.path
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
# print(sys.path)
####

import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
import src.utilities.saving_images as saving_images
import src.utilities.data_handling as data_handling




def dicom_to_png(dcm_full_path, output_directory, bitdepth = 12):
    """
    Transform dicom format to png.
    """
    ds = pydicom.dcmread(dcm_full_path)
    image = ds.pixel_array
    # exam_id = ds.PatientID
    # laterality = ds.ImageLaterality
    # view_angle= ds.ViewPosition
    # label = exam_id + '_' + laterality + '_' + view_angle
    # full_png_path = os.path.join(output_directory, label + '.png')
    filename = os.path.basename(dcm_full_path)[:-4] + '.png'
    full_png_path = os.path.join(output_directory, filename)

    with open(full_png_path, 'wb') as f:
        writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
        writer.write(f, image.tolist())




def add_dcm_extension(dir_name):
    """
    Auxiliary function. Adds '.dcm' extension to all files inside dir_name.
    """
    file_list = os.listdir(dir_name)
    for entry in file_list:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            add_dcm_extension(full_path)
        elif not entry.endswith(".dcm"):
            os.rename(full_path, full_path + ".dcm")




def exams_not_four(exam_list, input_directory, output_directory, bitdepth = 12, generate_png = False, num_processes = 2):
    """
    Analayze exam_not_four_list.
    Generate png from dicom if wanted.
    """

    print('Analyzing exams with less or more dicom images...')

    # check exam_error_list
    views = {'L-CC': 0, 'R-CC': 0, 'L-MLO': 0, 'R-MLO': 0}
    more_four = 0
    dcm_path_list = []
    for exam_dict in exam_list:
        images_per_exam = 0
        for k,v in exam_dict.items():
            if k in views:
                views[k] += len(v)
                images_per_exam += len(v)
                for filename in v:
                    dcm_path_list.append( os.path.join(input_directory, filename[:filename.find('_')], filename ) + '.dcm' )
        if images_per_exam >4:
            more_four += 1

    print(f'Exams with more than four images: {more_four}')
    
    # Check if folder exists
    images_dir = os.path.join(output_directory, 'images_not_four')
    if os.path.exists(images_dir):
        # Prevent overwriting to an existing directory
        print("Error: the directory to save png images already exists.")
        return
    else:
        os.makedirs(images_dir)

    if generate_png:
        print(f'\nGenerating png images spawning {num_processes} processes...')
        dicom_to_png_fix_out = partial(dicom_to_png, output_directory = images_dir)
        with Pool(num_processes) as pool:
            pool.map(dicom_to_png_fix_out, dcm_path_list)
    
    print('Done!\n')
    return views




def _get_dicom_files(dir_name):
    """
    Auxilary function used in cro_dicom_scrapper
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
            dcm_files = dcm_files + _get_dicom_files(full_path)
        # If entry is a file that endswith .dcm, append to dcm_list
        elif entry.endswith(".dcm"):
            dcm_files.append(full_path)
                
    return dcm_files




def cro_dicom_scrapper(input_directory, output_directory):
    """
    It finds all the .dcm images inside the input_directory and generates the a base metadata for each exam.
    From this base_exam_list you can generate the metadata needed to run the NYU Classification algorithm, both
    single view an four view model. 
    If the exam has any error of labels or formatting it is skipped.

    The structure of the input_directory should be as follows:
    input_directory/exam1/ ; input_directory/exam2/ ; input_directory/exam3/ 
    Each subdirectory corresponds to a single exam.
    
    The names of the .dcm files (usually random) are renamed to match the metadata information.

    :param input_directory: path to the directory containig exams
    :param output_directory: path to the directory where to put the metadata
    :return: base_exam_list: metadata list, exam_error_list: metadata list of exams with any type of error.
    """

    # Complete output path
    exam_list_path = os.path.join(output_directory, 'base_exam_list.pkl')
    exam_error_list_path = os.path.join(output_directory, 'exam_error_list.pkl')

    # Prevents overwriting an existing file
    if os.path.isfile(exam_list_path):
        print("Error: an exam list file with that name already exists.")
        return
    else:
        # Check if output_directory exists, if not, creates it.
        os.makedirs(output_directory, exist_ok=True)

    print('\nScrapping dicom images and checking for errors...')

    # list of files and subdirectories inside input_directory
    file_list = os.listdir(input_directory) # other alternative was os.walk
    
    # list of exams with errors in the images
    exam_error_list = []

    # exam_list compatible with the one required by NYU's 2019 algorithm
    base_exam_list = []
    
    # dcm path list. Used for dcm_to_png in parallel
    dcm_path_list = []

    for entry in tqdm(file_list):
        # print (f'entry: {entry}')
        full_path = os.path.join(input_directory, entry)
        # if the entry is a directory we asume it is an exam and look for the dcm images inside.
        if os.path.isdir(full_path):
            exam_dcm_paths = _get_dicom_files(full_path)
            base_exam_dict = {'horizontal_flip': '', 'L-CC': [], 'L-MLO': [], 'R-MLO': [], 'R-CC': [], 'num_images': 0}
            exam_id = ''


            for dcm_file in exam_dcm_paths:
                ds = pydicom.dcmread(dcm_file)

                # Filter out Material Filter: Aluminum filter produces different types of images. Rhodium filter produces normal images. 
                filter_material = ds.FilterMaterial
                if filter_material != 'RHODIUM':
                    continue

                horizontal_flip = ds.FieldOfViewHorizontalFlip
                laterality = ds.ImageLaterality
                view_angle= ds.ViewPosition

                # Check exam_id for all dcm (now we are working with exam_id because thats how they are given to us)
                if exam_id == '':
                    exam_id = ds.PatientID
                else:
                    # assert exam_id == ds.PatientID, "Exam id is not the same for all dcm images in folder " + entry
                    if not exam_id == ds.PatientID:
                        exam_error_list.append(entry)
                        break
                
                # Check horizontal_flip for all dcm
                if not base_exam_dict['horizontal_flip']:
                    base_exam_dict['horizontal_flip'] = horizontal_flip
                else:
                    # assert base_exam_dict['horizontal_flip'] == horizontal_flip , "Horizontal flip is not the same for every image in exam " + entry
                    if not base_exam_dict['horizontal_flip'] == horizontal_flip:
                        exam_error_list.append(entry)
                        break

                # Check if laterality and view_angle are as expected
                if laterality not in ['L', 'R']:
                    exam_error_list.append(entry)
                    break
                if view_angle not in ['CC', 'MLO']:
                    exam_error_list.append(entry)
                    break

                view = laterality + '-' + view_angle
                label = exam_id + '_' + laterality + '_' + view_angle
                
                # At this point the image passed all checks
                # It is now added to the corresponding exam_list_all[view]

                # if more than one image per view, add a _num to filename
                if base_exam_dict[view]:
                    label += '_' + str(len(base_exam_dict[view]))
                base_exam_dict[view].append(label)

                # base_exam_dict num_images increases by one
                base_exam_dict['num_images'] += 1



                # rename the dicom images to match the names in exam_list
                dcm_file_new_name = os.path.join(os.path.dirname(dcm_file), label + '.dcm')
                os.rename(dcm_file, dcm_file_new_name)
                dcm_file = dcm_file_new_name

            else:
                base_exam_list.append(base_exam_dict)
                dcm_path_list += exam_dcm_paths

    

    # saving lists:
    pickling.pickle_to_file(exam_list_path, base_exam_list)
    # pickling.pickle_to_file(exam_error_list_path, exam_error_list)


    print('\n########################################')
    print(f'Exams without errors: \n {len(base_exam_list)}')
    # print(f'Exams with less/more Dicom images: \n {len(exam_not_four_list)}')
    print(f'Exams with error in Dicom images: \n {len(exam_error_list)}')
    print('########################################\n')
    print('Done!\n')

    return base_exam_list, exam_error_list


def generate_exam_list(base_exam_list, output_directory = 'sample_data', output_filename = 'exam_list_before_cropping'):
    """
    Generate an exam list for the four view classifier 
    from the base exam list that include all exams with additional fields per exam.
    """

    exam_list_path = os.path.join(output_directory, output_filename + '.pkl')

    # Prevents overwriting an existing file
    if os.path.isfile(exam_list_path):
        print(exam_list_path)
        print("Error: an exam list file with that name already exists.")
        return
    else:
        # Check if output_directory exists, if not, creates it.
        os.makedirs(output_directory, exist_ok=True)
        
    exam_list = []
    exam_keys = ['horizontal_flip', 'L-CC', 'L-MLO', 'R-MLO', 'R-CC']


    # check if base_exam has one image per view
    equal_four = 0
    for base_exam in base_exam_list:
        for k, v in base_exam.items():
            if not v:
                # print(base_exam)
                break
        else:
            if base_exam['num_images'] == 4:
                # filter the fields from base_exam to match exam_keys as required by NYU classifier algorithm. 
                exam = {k:v for k,v in base_exam.items() if k in exam_keys}
                exam_list.append(exam)
                # print(exam)
                equal_four += 1

    print(f'exams with four images: {equal_four}')

    # save exam_list in output_directory
    pickling.pickle_to_file(exam_list_path, exam_list)

    return exam_list


def generate_exam_single_list(base_exam_list, output_directory = 'sample_data', output_filename = 'exam_single_list_before_cropping', view = 'L-CC'):
    """
    Generate an exam list for the single view classifier
    from the base_exam_list that includes all views and additional information.

    :param base_exam_list: base exam list. The exams in that list can have more than one view and additional fields.
                                
    """

    # metadata_dict = dict(
    #     short_file_path=None,
    #     horizontal_flip=horizontal_flip,
    #     full_view=view,
    #     side=view[0],
    #     view=view[2:],
    # )

    # TBI: handle multiple images of the same view per exam.

    exam_list_path = os.path.join(output_directory, output_filename + '.pkl')

    # Prevents overwriting an existing file
    if os.path.isfile(exam_list_path):
        print("Error: an exam list file with that name already exists.")
        return
    else:
        # Check if output_directory exists, if not, creates it.
        os.makedirs(output_directory, exist_ok=True)
    

    assert view in ['L-CC', 'R-CC', 'L-MLO', 'R-MLO'], 'Wrong view'
    # exam_list = pickling.unpickle_from_file(exam_list_path)
    print(base_exam_list)
    exam_single_list = []
    for base_exam in base_exam_list:
        if base_exam[view]:
            exam_single = {}
            exam_single['short_file_path'] = base_exam[view][0].replace('-','_')
            exam_single['horizontal_flip'] = base_exam['horizontal_flip']
            exam_single['full_view'] = view
            exam_single['side'] = view[0]
            exam_single['view'] = view[2:]
            exam_single_list.append(exam_single)

    # save exam_list in output_directory
    pickling.pickle_to_file(exam_list_path, exam_single_list)

    return exam_single_list


def _generate_dcm_path_list(exam_list, input_directory):
    """
    Auxiliary function to generate the dcm path list for a particular exam_list. It is used in image_from_exam_list().
    Because exam_single_dict doesn't share the same structure than regular exams dictionaries we
    need to force the user to specify if the exam is unique or not.

    :param input_directory: path to the directory containig dicom files. One directory per exam.
    """

    # TBI: merge single view with four view exams dictionaries. This would imply diverging too much
    #       from the original NYU repo.

    dcm_path_list = []

    if 'short_file_path' not in exam_list[0].keys():
        lookup_keys = ['L-CC', 'L-MLO', 'R-MLO', 'R-CC']
    else:
        lookup_keys = ['short_file_path']
    for exam in exam_list:
        temp = [    os.path.join( input_directory, filename[:filename.find('_')], filename+'.dcm' ) 
                for k,filename in exam.items() if k in lookup_keys  ]

        dcm_path_list += temp
        
    return dcm_path_list


def images_from_exam_list(exam_list, input_directory, output_directory = os.path.join('sample_data', 'images'),
                            bitdepth = 12, num_processes = 2 ):
    """
    Generate png images from an exam list.
    The default output_directory is ./sample_data/images/
    The png are generated in parallel.

    :param input_directory: path to the directory containig dicom files. One directory per exam.
    :param output_directory: path to the directory where to put the images

    """   

    dcm_path_list = _generate_dcm_path_list(exam_list, input_directory)
    print(dcm_path_list)


    # Create folder were to store .pngs and metadata
    # Check if folder exists
    if os.path.exists(output_directory):
        # Prevent overwriting to an existing directory
        print("Error: the directory to save png images already exists.")
        return
    else:
        os.makedirs(output_directory)
        # output_directory = os.path.join(output_directory, 'images')
        # os.makedirs(output_directory)


    # Generate pngs
    print(f'\nGenerating png images spawning {num_processes} processes...')
    dicom_to_png_fix_out = partial(dicom_to_png, output_directory = output_directory)
    with Pool(num_processes) as pool:
        pool.map(dicom_to_png_fix_out, dcm_path_list)
    print('Done!\n')


def input_array_from_exam_list(exam_list, input_directory, output_directory):
    images_ls = []
    dcm_path_list = _generate_dcm_path_list(exam_list[:20], input_directory)
    for dcm_path in dcm_path_list:
        ds = pydicom.dcmread(dcm_path)
        flatten_image = ds.pixel_array.flatten()
        images_ls.append(flatten_image)

    print(len(images_ls))

    with open(output_directory,'wb') as file: 
        np.save(file, np.concatenate(images_ls,0))


# torch.cat(outputs, 0).numpy()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform CRO DICOM images to PNG as downloaded from Ambra web interface.')
    parser.add_argument('-i', '--input-data-folder', help="Folder containing DICOM images. \n", required=True)
    parser.add_argument('-o', '--output-data-folder', help="Folder where PNG images will be stored. \n", required=True)
    args = parser.parse_args()

    # add_dcm_extension('../data_cro/dicom_CRO_23072019/anon')

    # base_exam_list, exam_error_list = cro_dicom_scrapper(args.input_data_folder, args.output_data_folder)

    
    ###########
    # FOUR VIEW

    # base_exam_list = pickling.unpickle_from_file('../data_cro/dicom_CRO_23072019/sample_data/base_exam_list.pkl')

    # exam_list = generate_exam_list(base_exam_list, args.output_data_folder)

    # images_dir = os.path.join(args.output_data_folder, 'images')
    # images_from_exam_list(exam_list, args.input_data_folder, images_dir, bitdepth = 12, num_processes = 10 )
    

    #############
    # SINGLE VIEW

    # base_exam_list = pickling.unpickle_from_file('../data_cro/dicom_CRO_23072019/sample_data/base_exam_list.pkl')

    # exam_single_list = generate_exam_single_list(base_exam_list, args.output_data_folder, view = 'L-CC')

    # print(exam_single_list)
    # images_dir = os.path.join(args.output_data_folder, 'images_single')
    # images_from_exam_list(exam_single_list, args.input_data_folder, images_dir, bitdepth = 12, num_processes = 10 )
    

    #############
    # INPUT FILE

    exam_list = pickling.unpickle_from_file('../data_cro/dicom_CRO_23072019/sample_data/exam_single_list_before_cropping.pkl')

    output_directory = '../data_cro/dicom_CRO_23072019/sample_single_output/activations/input.pkl'
    input_array_from_exam_list(exam_list, args.input_data_folder, output_directory)
