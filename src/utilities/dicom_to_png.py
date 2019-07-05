# packages: pip install pypng pydicom
# sometimes you need gdcm to uncompress some specific dicom format:
# conda install -c conda-forge gdcm 

# ds = pydicom.dcmread(dicom_filename)


import png
import pydicom
import os


import sys

####
## Temporary addition to find src file (if called from bash: export PYTHONPATH=$(pwd):$PYTHONPATH)
current_dir = os.path.dirname(os.path.abspath(__file__))
print(sys.path)
# appending parent dir of current_dir to sys.path
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
print(sys.path)
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



def cro_dicom_scrapper(input_directory, output_directory, bitdepth = 12, generate_png = False):
    """
    It goes threw all the folders inside input_directory, each one corresponding to a single CRO exam.
    It finds all the .dcm images associated with that exam and creates the metadata and .png.
    :param input_directory: path to the directory containig CRO exams
    :param output_directory: path to the directory where to put the images and metadata

    """
    # Create folder were to store pngs and metadata
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


    list_of_files = os.listdir(input_directory) # other alternative was os.walk
    
    # We create an exam_list compatible with the NYU's 2019 algorithm
    exam_list = []
    print(list_of_files)
    for entry in list_of_files:
        full_path = os.path.join(input_directory, entry)
        if os.path.isdir(full_path):
            exam_dcm_paths = get_dicom_files(full_path)
            exam_dict = {'horizontal_flip': '', 'L-CC': [], 'L-MLO': [], 'R-MLO': [], 'R-CC': []}
            exam_id = ''

            for dcm_file in exam_dcm_paths:
                ds = pydicom.dcmread(dcm_file)
                horizontal_flip = ds.FieldOfViewHorizontalFlip
                laterality = ds.ImageLaterality
                view_angle= ds.ViewPosition

                # Check exam_id for all dcm
                if exam_id == '':
                    exam_id = ds.StudyID
                else:
                    assert exam_id == ds.StudyID, "Exam id is not the same for all dcm images in the specified folder."
                # Check horizontal_flip for all dcm
                if not exam_dict['horizontal_flip']:
                    exam_dict['horizontal_flip'] = horizontal_flip
                else:
                    assert exam_dict['horizontal_flip'] == horizontal_flip , "Horizontal flip is not the same for every image in exam."

                view = laterality + '-' + view_angle
                label = exam_id + '_' + laterality + '_' + view_angle
                exam_dict[view].append(label)

                # Generate pngs
                if generate_png:
                    full_png_path = os.path.join(images_dir, label + '.png')
                    image = ds.pixel_array
                    # try:
                    #     # F: saves the cropped image!
                    #     saving_images.save_image_as_png(image, full_png_path)
                    # except Exception as error:
                    #     print(full_file_path, "\n\tError while saving image.", str(error))

                    with open(full_png_path, 'wb') as f:
                        writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
                        writer.write(f, image.tolist())

            exam_list.append(exam_dict)

        # print(f"exam_list: \n {exam_list}")
        exam_list_path = os.path.join(output_directory, 'exam_list_before_cropping.pkl')
        pickling.pickle_to_file(exam_list_path, exam_list)

        


def get_dicom_files(dir_name):
    """
    For the given path, get the List of all files in the directory tree 
    """
    # create a list of file and sub directories 
    # names in the given directory 
    list_of_files = os.listdir(dir_name)
    dcm_files = []
    # Iterate over the entries
    for entry in list_of_files:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(full_path):
            dcm_files = dcm_files + get_dicom_files(full_path)
        # If entry is a file that endswith .dcm, append to dcm_list
        elif entry.endswith(".dcm"):
            dcm_files.append(full_path)
                
    return dcm_files



if __name__ == "__main__":
    cro_dicom_scrapper("./dicom_CRO/", "./sample_data_CRO/", 12, True)