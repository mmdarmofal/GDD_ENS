
import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

def process_data_bal(test_size, n_splits, upsample, path_to_ft, remove = False): 
	''' 
	process_data_bal will create train, test, validation folds for all classes with balanced sample sizes up to n=350 and save them to the folder the script is run in
	requires path to final, feature table outputted from generate_feature_table.py
	-removes duplicates as long as there are enough examples before upsampling, 
	input:
		test_size = integer representing proportion out of 100 which should be used to make the testing set (default = 20, meaning test size is .2 of overall dataset)
		n_splits = integer, number of ensembles to create
		path_to_ft = path to input ft to be split for training and testing
	output:
		None - saves data tables for future use 
	'''

	data = pd.read_csv(path_to_ft, sep=',', index_col = 0) 
	data = data.assign(PATIENT_ID=data.SAMPLE_ID.str[:9])
	data = data.assign(rep_drop = data.PATIENT_ID + data.Cancer_Type)
	#Removing those not annotated as a training sample (i.e. other, low purity)
	data = data[data.Classification_Category == 'train']
	labels = data.Cancer_Type
	ctypes = pd.read_csv('data/tumor_type_ordered.csv').Cancer_Type
	#split data into train and test before balancing
	data_train_labelled, data_test_labelled, labels_train_labelled, labels_test_labelled = train_test_split(data, labels, test_size=test_size/100, random_state = 0)
	
	#data drops the following labels
	data = data.drop(['SAMPLE_ID', 'PATIENT_ID', 'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'SAMPLE_TYPE', 'PRIMARY_SITE', 'METASTATIC_SITE', 'Cancer_Type', 'Classification_Category', 'rep_drop'], axis=1)
	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size/100, random_state = 0)
	label_vc = labels_train.value_counts()
	labels_train.value_counts().to_csv('output/train_N.csv')
	#remove unexpressed columns in training
	if remove == True:
		drop_cols = [col for col in data_train.columns if len(set(data_train[col]))==1]
		keep_cols = [col for col in data_train.columns if col not in drop_cols]
		data_train = data_train[keep_cols]
		data_test = data_test[keep_cols]
		drop_cols = pd.DataFrame(drop_cols)
		drop_cols.to_csv('output/dropped_cols.csv')
	#balance by oversampling to n = 350 for any types that dont have enough in training
	sampling_dic = {}
	for i in ctypes:
		sampling_dic[i] = max(list(labels_train).count(i), upsample)
	ros = RandomOverSampler(random_state=0, sampling_strategy = sampling_dic) 
	data_train_bal, labels_train_bal = ros.fit_resample(data_train, labels_train)
	data_train_labelled, labels_train_labelled = ros.fit_resample(data_train_labelled, labels_train_labelled)
	
	#remove duplicate patients in the test set
	data_test_labelled_noreps = data_test_labelled[~data_test_labelled.rep_drop.isin(data_train_labelled.rep_drop.values)]
	data_test_noreps = data_test[~data_test_labelled.rep_drop.isin(data_train_labelled.rep_drop.values)]
	labels_test_noreps = labels_test[~data_test_labelled.rep_drop.isin(data_train_labelled.rep_drop.values)]
	dropped_dups = data_test_labelled[data_test_labelled.rep_drop.isin(data_train_labelled.rep_drop.values)]

	#saving data tables -> 
	data_train_labelled.to_csv('output/ft_train_labelled.csv', header = True, index = True)
	data_test_labelled_noreps.to_csv('output/ft_test_labelled.csv', header = True, index = True)
	data_train_bal.to_csv('output/ft_train.csv', header = True, index = True)
	labels_train_bal.to_csv('output/labels_train.csv', header = True, index = True)
	data_test_noreps.to_csv('output/ft_test.csv', header = True, index = True)
	labels_test_noreps.to_csv('output/labels_test.csv', header = True, index = True)
	dropped_dups.to_csv('output/duplicate_patients.csv', header = True, index = True)

if __name__ == '__main__':
	#specs
	test_size = 20
	n_splits = 10
	upsample = 350
	path_to_ft = sys.argv[-1]
	if len(sys.argv) < 2:
		raise Exception("Not enough arguments")

	sys.path.insert(0, '../')

	process_data_bal(test_size, n_splits, upsample, path_to_ft, remove = True)
