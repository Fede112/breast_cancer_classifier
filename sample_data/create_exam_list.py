import os, sys
# print(sys.path)
current_dir = os.path.dirname(os.path.abspath(__file__))
# appending parent dir of current_dir to sys.path
sys.path.append(os.path.dirname(current_dir))


from multiprocessing import Pool
import argparse
from functools import partial
import scipy.ndimage
import numpy as np
import pandas as pd


import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
import src.utilities.saving_images as saving_images
import src.utilities.data_handling as data_handling





# list of exams (one dictionary per exam)
exam_list = pickling.unpickle_from_file('exam_list_before_cropping.pkl')
cropped_exam_list_path = '../sample_output/cropped_images/cropped_exam_list_test.pkl'
print(exam_list)
exam_list[0]['L-CC'] = []
# list per image (one dictionary per image). It contains same information than in list of exams + cropped information if present.
image_list = data_handling.unpack_exam_into_images(exam_list)



pickling.pickle_to_file(cropped_exam_list_path, exam_list)
