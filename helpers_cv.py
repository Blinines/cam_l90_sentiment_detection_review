# -*- coding: utf-8 -*-
from os import listdir
from private import PATH_PROJECT

def sep_train_test(files_path, top_value_train):
	train_files = []
	test_files = []
	
	for index, file in enumerate(listdir(files_path)):
		if index < top_value_train and file.endswith(".tag"):
			train_files.append(PATH_PROJECT+files_path+file)
		if index >= top_value_train and file.endswith(".tag"):
			test_files.append(PATH_PROJECT+files_path+file)
	return train_files, test_files