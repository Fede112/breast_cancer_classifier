import sqlite3
from sqlite3 import Error
import pandas as pd
import string
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
# appending parent dir of current_dir to sys.path
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))


from src.analysis import stats
# import src.utilities.reading_images as reading_images


# tools.cumsum_sample()

####################################
# Input checks
####################################

# if model == 'image-only':
#   index_1 = data_path.find('heatmap')
#   index_2 = pred_path.find('heatmap')
#   assert index_1 >= 0, 'Possible error: image-only model but image-heatmaps in data path.'
#   assert index_2 >= 0, 'Possible error: image-only model but image-heatmaps in pred path.'

# if model == 'image-heatmaps':
#   index_1 = data_path.find('only')
#   index_2 = pred_path.find('only')
#   assert index_1 >= 0, 'Possible error: image-heatmaps model but image-only in data path.'
#   assert index_2 >= 0, 'Possible error: image-heatmaps model but image-only in pred path.'



####################################
# Load data exam_list and predictions from model
####################################

# # data_path = '../sample_output/data.pkl'
# # pred_path = '../sample_output/image_predictions.csv'

# # read exam_list from the NYU output. It is a list of dictionaries, one per exam.
# with open(data_path, 'rb') as handle:
#     exam_list_dict = pickle.load(handle)

# # read predictions
# pred_full = pd.read_csv(pred_path, sep=',')


# exam_id_list = []
# # substring = substring.replace("-", ' ')
# for dic in exam_list_dict:
#   exam_id_list.append( dic['L-CC'][0].split('_')[0] )


# pred_full['exam_id'] = exam_id_list




####################################
# Read reports from sqlite3 and extract labels
####################################

 
# def sql_connection():
 
#     try:
 
#         con = sqlite3.connect('../dicom_CRO_23072019/reports/reports_23072019.db')
 
#         return con
 
#     except Error:
 
#         print(Error)
 
# def sql_table(con):
 
#     cursorObj = con.cursor()
 
#     # cursorObj.execute("CREATE TABLE employees(id integer PRIMARY KEY, name text, salary real, department text, position text, hireDate text)")
#     # cursorObj.execute('.tables')
 
#     con.commit()
 
# con = sql_connection()
 
# sql_table(con)



class ReportsDB:
    def __init__(self, database_path):

        # Create your connection.
        try:
            conn = sqlite3.connect(database_path)
        except Error:
            print(Error)

        # change encoding to latin. Default is utf-8
        conn.text_factory = lambda x: str(x, 'latin1')
        # conn.text_factory = bytes

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        assert len(tables) == 1, 'Database with a single table containing reports is expected!'

        # for loop although one element list to later extend the functionality
        for table_name in tables:
            # table_name is a tuple(name,)
            table_name = table_name[0]
            print(f'Table name: {table_name}')
            table = pd.read_sql_query("SELECT * from %s" % table_name, conn)
            # table.to_csv(table_name + '.csv', index_label='index')

        cursor.execute("SELECT * from " + table_name)
        colnames = cursor.description
        colnames = [row[0] for row in colnames]

        print(f'Column names of table {table_name}: {colnames}')
        assert 'patient_id' in colnames, 'patient_id column missing in ' + table_name
        assert 'study_uid' in colnames, 'study_uid column missing in ' + table_name
        assert 'report' in colnames, 'report column missing in ' + table_name

        self.df = pd.read_sql_query("SELECT * FROM step3_img_report", conn)
        # look up keywords
        self.keywords = ['RADS', 'intervento']
        cursor.close()

    def search_keyword(self, keyword):
        """
        Searchs for keyword in each report.
        Returns list of reports containing the keyword.
        """
        keyword_ls = []
        for i,row in self.df.iterrows():
          pat_id = row['patient_id']
          study_uid = row['study_uid']
          index = row['report'].find(keyword)
          if index >=0:
              keyword_ls.append((pat_id, study_uid))
        return keyword_ls

    def birad_label(self, keyword):
        """
        Searchs for the birad label in each report.
        Returns a list of tuple, one tuple per report.
        tuple: (pat_id, study_uid, birad)
        """
        index = row['report'].find(key_word_1)
        substring = row['report'][index:index+15]



repdb = ReportsDB('../dicom_CRO_23072019/reports/reports_23072019.db')
# print(repdb.df['report'][712])
# print('')
# print(repdb.df['report'][413])



# print(repdb.df['report'][221])
# print(repdb.df['report'][212])
# print('')
# print(repdb.df['report'][213])
# print('')
# print(repdb.df['report'][214])
# print('')







# # Create your connection.
# conn = sqlite3.connect('../dicom_CRO_23072019/reports/reports_23072019.db')

# # change encoding to latin. Default is utf-8
# conn.text_factory = lambda x: str(x, 'latin1')
# # conn.text_factory = bytes

# # reads table and transforms it into dataframe
# df = pd.read_sql_query("SELECT * FROM step3_img_report", conn)

# # key_words to filter the reports
# key_word_1 = 'RADS'
# key_word_2 = 'intervento'

# interv_list = []
# birad_list = []

# more_4 = 0
# for i,row in df.iterrows():

#   # pat_id = row['PZ'] 
#   pat_id = row['patient_id'] 
    
#   # index = row['REPORT'].find(key_word_2)
#   index = row['report'].find(key_word_2)
#   if index >=0:
#       interv_list.append((pat_id, 1))


# #   # PI: loop to find all BI-RADS occurence in the report
# #   # index = row['REPORT'].find(key_word_1)
# #   # substring = row['REPORT'][index:index+15]
#   index = row['report'].find(key_word_1)
#   substring = row['report'][index:index+15]
#   # remove punctuations
#   substring = substring.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
#   # substring = substring.replace("-", ' ')
#   # substring = substring.replace(')', ' ')
    
#   # adhoc replaces based on what I saw in the data
#   substring = substring.replace('S', ' ')
#   substring = substring.replace('n', ' ')
#   substring = substring.replace('a', ' ')
#   substring = substring.replace('b', ' ')
#   substring = substring.replace('c', ' ')
#   birad = [int(s) for s in substring.split() if s.isdigit()]

#   if len(birad):
#       birad = sum(birad) / len(birad)
#       birad_list.append((pat_id, birad))


#       if (birad > 4):
#           more_4 += 1


# print(f'more than 4: {more_4}')
# # print(birad_list)
# print(f'birad_list len: {len(birad_list)}')





####################################
# Add birad and intervento columns to pred_full dataframe
####################################


# pred_full['birad'] = None
# for pat_id, birad in birad_list:
#   pred_full.loc[pred_full['exam_id'] == pat_id, 'birad'] = birad


# pred_full['intervento'] = 0
# for pat_id, interv in interv_list:
#   pred_full.loc[pred_full['exam_id'] == pat_id, 'intervento'] = interv


# # prediction dataframe without birad nulls
# pred = pred_full[pred_full["birad"].notnull()]
# print(pred)




####################################
# Plots
####################################

# label = 'malignant'
# pred_1_2 = pred[ pred['birad'] < 2.5 ]['left_'+label] + pred[pred['birad'] < 2.5 ]['right_'+label]
# pred_3_4 = pred[ (pred['birad'] >= 2.5) & (pred['birad'] < 4.5) ]['left_'+label] + pred[ (pred['birad'] >= 2.5) & (pred['birad'] < 4.5) ]['right_'+label]
# pred_5_6 = pred[ pred['birad'] >= 4.5 ]['left_'+label] + pred[ pred['birad'] >= 4.5 ]['right_'+label]

# pred_4_6 = pred[ pred['birad'] >= 3.5 ]['left_'+label] + pred[ pred['birad'] >= 3.5 ]['right_'+label]
# pred_1_3 = pred[ pred['birad'] < 3.5 ]['left_'+label] + pred[ pred['birad'] < 3.5 ]['right_'+label]


# sort_1_2, cumsum_1_2 = cumsum_sample(pred_1_2)
# sort_3_4, cumsum_3_4 = cumsum_sample(pred_3_4)
# sort_4_6, cumsum_4_6 = cumsum_sample(pred_4_6)
# sort_1_3, cumsum_1_3 = cumsum_sample(pred_1_3)


# # plot the sorted data:
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.plot(sort_4_6, cumsum_4_6, '.-', label = 'BR 4-6')
# ax1.plot(sort_1_2, cumsum_1_2, '.-',label = 'BR 1-2')
# ax1.set_xlabel(label +' finding prob')
# ax1.set_ylabel('$p$')
# ax1.legend()


# label = 'benign'
# pred_1_2 = pred[pred['birad'] < 2.5 ]['left_'+label] + pred[pred['birad'] < 2.5 ]['right_'+label]
# pred_3_4 = pred[ (pred['birad'] >= 2.5) & (pred['birad'] < 4.5) ]['left_'+label] + pred[ (pred['birad'] >= 2.5) & (pred['birad'] < 4.5) ]['right_'+label]
# pred_5_6 = pred[ pred['birad'] >= 4.5 ]['left_'+label] + pred[ pred['birad'] >= 4.5 ]['right_'+label]
# pred_4_6 = pred[ pred['birad'] >= 3.5 ]['left_'+label] + pred[ pred['birad'] >= 3.5 ]['right_'+label]
# pred_1_3 = pred[ pred['birad'] < 3.5 ]['left_'+label] + pred[ pred['birad'] < 3.5 ]['right_'+label]


# sort_1_2, cumsum_1_2 = cumsum_sample(pred_1_2)
# sort_3_4, cumsum_3_4 = cumsum_sample(pred_3_4)
# sort_4_6, cumsum_4_6 = cumsum_sample(pred_4_6)
# sort_1_3, cumsum_1_3 = cumsum_sample(pred_1_3)


# ax2 = fig.add_subplot(122)
# ax2.plot(sort_4_6, cumsum_4_6, '.-', label = 'BR 4-6')
# ax2.plot(sort_1_2, cumsum_1_2, '.-',label = 'BR 1-2')
# ax2.set_xlabel(label + ' finding prob')
# ax2.set_ylabel('$p$')
# ax2.legend()


# fig.suptitle(model + ' model')
# plt.tight_layout()
# fig.subplots_adjust(top=0.92)
# plt.show()


# if __name__ == "__main__":

    # run example:
    # python read_reports_db.py -m '../sample_output/data.pkl' -p '../sample_output/image_predictions.csv' --model 'image-only'

    # parser = argparse.ArgumentParser(description='Prediction analysis.')
    # parser.add_argument('-m', '--metadata-path', required=True, type = str)
    # parser.add_argument('-p', '--predictions-path', required=True, type = str)
    # parser.add_argument('--model', required=True, choices=['image-only', 'image-heatmaps'], type = str)
        
    # args = parser.parse_args()

    # data_path = args.metadata_path
    # pred_path = args.predictions_path
    # model = args.model