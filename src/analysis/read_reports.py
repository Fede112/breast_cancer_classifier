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
# Read reports from sqlite3 and extract labels
####################################



class Reports:
    """
    Wrapper of reports sql database.
    Allows extraction of birad label.
    Allos extraction of additional keywords.
    """
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
        # Add mandatory birad label
        self.df['birads'] = None
      
        self.extract_birads()

        # look up keywords
        self.keywords = []
        
        cursor.close() 

    def extract_birads(self):
        """
        Searchs for the birad label in each report.
        Returns a list of tuple, one tuple per report.
        tuple: (pat_id, study_uid, birad)
        """
        for i,row in self.df.iterrows():

            index = row['report'].find('RADS')
            substring = row['report'][index:index+15]
            # remove punctuations
            substring = substring.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
            # substring = substring.replace("-", ' ')
            # substring = substring.replace(')', ' ')

            # adhoc replaces based on what I saw in the data
            substring = substring.replace('S', ' ')
            substring = substring.replace('n', ' ')
            substring = substring.replace('a', ' ')
            substring = substring.replace('b', ' ')
            substring = substring.replace('c', ' ')
            birads = [int(s) for s in substring.split() if s.isdigit()]

            if len(birads):
                birads = sum(birads) / len(birads)
                row['birads'] = birads
          


    def add_keyword(self, keyword):
        self.keywords.append(keyword)
        self.df[keyword] = None

    def search_keywords(self):
        """
        Searchs for keyword in each report.
        Returns list of reports containing the keyword.
        """

        # Add kw to df
        for kw in self.keywords:
            self.df[kw] = 0

        for i,row in self.df.iterrows():
            for kw in self.keywords:
                index = row['report'].find(kw)
                if index >=0:
                    self.df.at[i,kw] = 1




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



if __name__ == "__main__":
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


    ####################################
    # Read reports
    ####################################

    reports = Reports('../dicom_CRO_23072019/reports/reports_23072019.db')

    reports.add_keyword('interv')
    print(reports.keywords)
    reports.search_keywords()

    print(reports.df.head(100))


    ####################################
    # Load data exam_list and predictions from model
    ####################################

    data_path = '../dicom_CRO_23072019/sample_output/data.pkl'
    pred_path = '../dicom_CRO_23072019/sample_output/image_predictions.csv'

    # read exam_list from the NYU output. It is a list of dictionaries, one per exam.
    with open(data_path, 'rb') as handle:
        exam_list_dict = pickle.load(handle)

    # read predictions
    pred_full = pd.read_csv(pred_path, sep=',')


    exam_id_list = []
    # substring = substring.replace("-", ' ')
    for dic in exam_list_dict:
      exam_id_list.append( dic['L-CC'][0].split('_')[0] )


    pred_full['patient_id'] = exam_id_list


    # for col in reports.df.columns: 
    #     print(col) 


    pred = pd.merge(pred_full,
                 reports.df[['patient_id', 'birads', 'interv']], 
                 on='patient_id')


    # print(reports.df.where(reports.df['patient_id'] == pred_full['patient_id']).notna())


    print(f'shared elements: {np.sum(pred_full.patient_id.isin(reports.df.patient_id).astype(int))}')


    pred.head(10)

    print(f'Number of Exams: {len(pred)}')

    ####################################
    # Plots
    ####################################
    model = 'L+R - Four view'
    label = 'malignant'
    
    # Birads 1 or 1-2
    birads_rest = pred[ pred['birads'] <= 3.5 ]['left_'+label] + pred[pred['birads'] <= 3.5 ]['right_'+label]

    # Birads from 4 up
    birads_4_6 = pred[ pred['birads'] > 4.5 ]['left_'+label] + pred[ pred['birads'] > 4.5 ]['right_'+label]


    sort_rest, cumsum_rest = stats.cumsum_sample(birads_rest)
    sort_4_6, cumsum_4_6 = stats.cumsum_sample(birads_4_6)


    # plot the sorted data:
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(sort_4_6, cumsum_4_6, '.-', label = label + ' (BR 4,5,6)')
    ax1.plot(sort_rest, cumsum_rest, '.-', label = 'not ' + label ) 
    ax1.set_xlabel(label +' finding prob')
    ax1.set_ylabel('$p$')
    ax1.legend()


    label = 'benign'
    birads_rest = pred[ (pred['birads'] < 1.5) | (pred['birads'] >= 3.5) ]['left_'+label] + pred[ (pred['birads'] < 1.5) | (pred['birads'] >= 3.5) ]['right_'+label]
    birads_2_3 = pred[ (pred['birads'] >= 1.5) & (pred['birads'] < 3.5) ]['left_'+label] + pred[ (pred['birads'] >= 1.5) & (pred['birads'] < 3.5) ]['right_'+label]
    
    sort_rest, cumsum_rest = stats.cumsum_sample(birads_rest)
    sort_2_3, cumsum_2_3 = stats.cumsum_sample(birads_2_3)
    

    ax2 = fig.add_subplot(122)
    ax2.plot(sort_2_3, cumsum_2_3, '.-', label = label + ' (BR 2,3)')
    ax2.plot(sort_rest, cumsum_rest, '.-',label = 'not ' + label )
    ax2.set_xlabel(label + ' finding prob')
    ax2.set_ylabel('$p$')
    ax2.legend()


    fig.suptitle(model + ' model')
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()

    ####################################################

    side = 'right'
    model = side + ' breast - Four view'
    label = 'malignant'
    
    # Birads 1 or 1-2
    birads_rest = pred[ pred['birads'] <= 3.5 ][side+'_'+label] # + pred[pred['birads'] <= 3.5 ]['right_'+label]

    # Birads from 4 up
    birads_4_6 = pred[ pred['birads'] > 4.5 ][side+'_'+label] # + pred[ pred['birads'] > 4.5 ]['right_'+label]


    sort_rest, cumsum_rest = stats.cumsum_sample(birads_rest)
    sort_4_6, cumsum_4_6 = stats.cumsum_sample(birads_4_6)


    # plot the sorted data:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    ax1.plot(sort_4_6, cumsum_4_6, '.-', label = label + ' (BR 4,5,6)')
    ax1.plot(sort_rest, cumsum_rest, '.-', label = 'not ' + label ) 
    ax1.set_xlabel(label +' finding prob')
    ax1.set_ylabel('$p$')
    ax1.legend()


    label = 'benign'
    birads_rest = pred[ (pred['birads'] < 1.5) | (pred['birads'] >= 3.5) ][side+'_'+label] # +  pred[ (pred['birads'] < 1.5) | (pred['birads'] >= 3.5) ]['right_'+label]
    birads_2_3 = pred[ (pred['birads'] >= 1.5) & (pred['birads'] < 3.5) ][side+'_'+label] # + pred[ (pred['birads'] >= 1.5) & (pred['birads'] < 3.5) ]['right_'+label]
    
    sort_rest, cumsum_rest = stats.cumsum_sample(birads_rest)
    sort_2_3, cumsum_2_3 = stats.cumsum_sample(birads_2_3)
    

    ax2 = fig1.add_subplot(122)
    ax2.plot(sort_2_3, cumsum_2_3, '.-', label = label + ' (BR 2,3)')
    ax2.plot(sort_rest, cumsum_rest, '.-',label = 'not ' + label )
    ax2.set_xlabel(label + ' finding prob')
    ax2.set_ylabel('$p$')
    ax2.legend()


    fig1.suptitle(model + ' model')
    plt.tight_layout()
    fig1.subplots_adjust(top=0.92)
    plt.show()