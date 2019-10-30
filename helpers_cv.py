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


class RoundRobinCV:
	def __init__(self, clf, path_neg, path_pos, mod=10):
		self.clf = clf
		self.all_folders = {'NEG': self.folder_round_robin(files_path=path_neg, mod=mod),
							'POS': self.folder_round_robin(files_path=path_pos, mod=mod)}
		self.mod = mod
	

	def folder_round_robin(self, files_path, mod):
		folders = {nb: [] for nb in range(mod)}
		for index, file_name in enumerate(listdir(files_path)):
			folders[index%mod].append(PATH_PROJECT+files_path+file_name)
		return folders
	

	def get_train_test_rr_fold(self, folders, index):
		train_files = []
		test_files = folders[index]
		for fold in folders.keys():
			if fold != index:
				train_files += folders[fold]
		return train_files, test_files


	def cross_validate(self):
		results = {}
		for index in range(self.mod):
			X_train, X_test = {}, {}
			for val in ['NEG', 'POS']: # retrieving train files and test files
				curr_folders = self.all_folders[val]
				curr_train, curr_test = self.get_train_test_rr_fold(folders=curr_folders, index=index)
				X_train[val] = curr_train
				X_test[val] = curr_test

			curr_clf = self.clf
			curr_clf.fit(X_train)
			results[index] = curr_clf.predict(X_test)

		return results
	