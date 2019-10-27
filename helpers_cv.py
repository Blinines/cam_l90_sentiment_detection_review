# -*- coding: utf-8 -*-
from os import listdir
from private import PATH_PROJECT

def sep_train_test(files_path, top_value_train):
	train_files = []
	test_files = []
	
	for index, file_name in enumerate(listdir(files_path)):
		if index < top_value_train and file_name.endswith(".tag"):
			train_files.append(PATH_PROJECT+files_path+file_name)
		if index >= top_value_train and file_name.endswith(".tag"):
			test_files.append(PATH_PROJECT+files_path+file_name)
	return train_files, test_files


def folder_round_robin(files_path, mod):
	folders = {nb: [] for nb in range(mod)}
	for index, file_name in enumerate(listdir(files_path)):
		folders[index%mod].append(PATH_PROJECT+files_path+file_name)
	return folders


def get_train_test_rr_fold(folders, index):
	train_files = []
	test_files = folders[index]
	for fold in folders.keys():
		if fold != index:
			train_files += folders[fold]
	return train_files, test_files