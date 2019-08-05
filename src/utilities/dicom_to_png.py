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



# def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth=12):
#     """
#     Save 12-bit mammogram from dicom as rescaled 16-bit png file.
#     :param dicom_filename: path to input dicom file.
#     :param png_filename: path to output png file.
#     :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
#     """
#     image = pydicom.read_file(dicom_filename).pixel_array
#     with open(png_filename, 'wb') as f:
#         writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
#         writer.write(f, image.tolist())
  



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




def cro_dicom_scrapper(input_directory, output_directory, bitdepth = 12, generate_png = False, num_processes = 2):
    """
    It finds all the .dcm images inside the input_directory and generates the corresponding metadata
    needed to run the NYU Classification algorithm. If the exam has any error of labels or formatting it is skipped.

    The structure of the input_directory should be as follows:
    input_directory/exam1/ ; input_directory/exam2/ ; input_directory/exam3/ 
    Each subdirectory corresponds to a single exam.
    
    The names of the .dcm files (usually random) are renamed to match the metadata information.
    It also provides the option to generate the png files. The png are generated in parallel.

    :param input_directory: path to the directory containig exams
    :param output_directory: path to the directory where to put the images and metadata
    :return: exam_list: metadata list, exam_not_four_list: metadata list of exams with more/less than four images,
             exam_error_list: metadata list of exams with any type of error.
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


    print('\nScrapping dicom images and checking for errors...')

    # list of files and subdirectories inside input_directory
    file_list = os.listdir(input_directory) # other alternative was os.walk
    
    # list of exams with errors in the images
    exam_error_list = []

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
            exam_dcm_paths = _get_dicom_files(full_path)
            exam_dict = {'horizontal_flip': '', 'L-CC': [], 'L-MLO': [], 'R-MLO': [], 'R-CC': []}
            exam_id = ''

            # If there are less or more than 4 images don't use it
            # In the future we should handle more than 4 images
            # if not len(exam_dcm_paths) == 4:
            #     exam_error_list.append(entry)
            #     continue

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
                if not exam_dict['horizontal_flip']:
                    exam_dict['horizontal_flip'] = horizontal_flip
                else:
                    # assert exam_dict['horizontal_flip'] == horizontal_flip , "Horizontal flip is not the same for every image in exam " + entry
                    if not exam_dict['horizontal_flip'] == horizontal_flip:
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
                
                if exam_dict[view]:
                    label += '_' + str(len(exam_dict[view]))
                exam_dict[view].append(label)

                # rename the dicom images to match the names in exam_list
                dcm_file_new_name = os.path.join(os.path.dirname(dcm_file), label + '.dcm')
                os.rename(dcm_file, dcm_file_new_name)
                dcm_file = dcm_file_new_name


            else:
                for key in exam_dict.keys():
                    if not key:
                        exam_error_list.append(entry)
                        break
                if not len(exam_dcm_paths) == 4:
                    exam_not_four_list.append(exam_dict)
                else:
                    exam_list.append(exam_dict)
                    dcm_path_list += exam_dcm_paths

    
    # saving lists:
    exam_list_path = os.path.join(output_directory, 'exam_list_before_cropping.pkl')
    exam_not_four_list_path = os.path.join(output_directory, 'exam_not_four_list.pkl')
    exam_error_list_path = os.path.join(output_directory, 'exam_error_list.pkl')

    pickling.pickle_to_file(exam_list_path, exam_list)
    pickling.pickle_to_file(exam_not_four_list_path, exam_not_four_list)
    pickling.pickle_to_file(exam_error_list_path, exam_error_list)


    print('\n########################################')
    print(f'Exams without errors: \n {len(exam_list)}')
    print(f'Exams with less/more Dicom images: \n {len(exam_not_four_list)}')
    print(f'Exams with error in Dicom images: \n {len(exam_error_list)}')
    print('########################################\n')
    print('Done!\n')


    # Generate pngs
    if generate_png:
        print(f'\nGenerating png images spawning {num_processes} processes...')
        dicom_to_png_fix_out = partial(dicom_to_png, output_directory = images_dir)
        with Pool(num_processes) as pool:
            pool.map(dicom_to_png_fix_out, dcm_path_list)
        print('Done!\n')
    return exam_list, exam_not_four_list, exam_error_list


    

def generate_exam_single_list(exam_list_path, view):
    """
    Generate an exam list for a single view, e.g. 'L-CC',
    from the original exam list that includes all views.
    This exam_list is the one generated by cro_dicom_scrapper and 
    it is called 'exam_list_before_cropping.pkl' by convention.

    :param exam_list_path: path to the full exam_list. By default called 'exam_list_before_cropping.pkl'
    """
    assert view in ['L-CC', 'R-CC', 'L-MLO', 'R-MLO'], 'Wrong view'
    exam_list = pickling.unpickle_from_file(exam_list_path)
    print(exam_list)
    exam_single_list = []
    for exam in exam_list:
        exam_single = {}
        exam_single['short_file_path'] = exam[view][0].replace('-','_')
        exam_single['horizontal_flip'] = exam['horizontal_flip']
        exam_single['full_view'] = view
        exam_single['side'] = view[0]
        exam_single['view'] = view[2:]
        exam_single_list.append(exam_single)

    output_path = os.path.join(os.path.dirname(exam_list_path), 'exam_single_list_before_cropping.pkl')
    pickling.pickle_to_file(output_path, exam_single_list)
    return exam_single_list


    # metadata_dict = dict(
    #     short_file_path=None,
    #     horizontal_flip=horizontal_flip,
    #     full_view=view,
    #     side=view[0],
    #     view=view[2:],
    # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform CRO DICOM images to PNG as downloaded from Ambra web interface.')
    parser.add_argument('-i', '--input-data-folder', help="Folder containing DICOM images. \n", required=True)
    parser.add_argument('-o', '--output-data-folder', help="Folder where PNG images will be stored. \n", required=True)
    args = parser.parse_args()

    # add_dcm_extension('../data_cro/dicom_CRO_23072019/anon')

    exam_list, exam_not_four_list, exam_error_list = cro_dicom_scrapper(args.input_data_folder, args.output_data_folder, 12, False, 10)


    exam_not_four_list = pickling.unpickle_from_file(os.path.join(args.output_data_folder, 'exam_not_four_list.pkl'))
    # print(exam_not_four_list)
    exams_not_four(exam_not_four_list, args.input_data_folder, args.output_data_folder, 12, True, 10) 

    # exam_list = generate_exam_single_list('./sample_data/exam_list_before_cropping.pkl', 'L-CC')
    # print(f'exam single list dicom_to_png: {exam_list}')