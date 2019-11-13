# -*- coding: utf-8 -*-
from os import listdir
import numpy as np
from private.private import PATH_PROJECT

def sep_train_test(files_path, top_value_train):
	""" Separates train and test data. Takes the top_value_train first values for training """
	train_files = []
	test_files = []
	
	for index, file_name in enumerate(listdir(files_path)):
		if index < top_value_train:  # and file_name.endswith(".tag"):
			train_files.append(PATH_PROJECT+files_path+file_name)
		if index >= top_value_train:  # and file_name.endswith(".tag"):
			test_files.append(PATH_PROJECT+files_path+file_name)
	return train_files, test_files


def folder_round_robin(files_path, mod):
	""" Folders for RR CV """
	folders = {nb: [] for nb in range(mod)}
	for index, file_name in enumerate(listdir(files_path)):
		folders[index%mod].append(PATH_PROJECT+files_path+file_name)
	return folders


def get_joint_folders(all_folders, mod):
	all_x, all_y = [[] for _ in range(mod)], [[] for _ in range(mod)]
	for val in [0, 1]:
		for key in all_folders[val].keys():
			to_add = all_folders[val][key]
			all_x[key] += to_add
			all_y[key] += [val] * len(to_add)
	return all_x, all_y


class RoundRobinCV:
	def __init__(self, clf, path_neg, path_pos, mod=10):
		self.clf = clf
		all_folders_sep = {0: folder_round_robin(files_path=path_neg, mod=mod),
					       1: folder_round_robin(files_path=path_pos, mod=mod)}
		self.all_x, self.all_y = get_joint_folders(all_folders=all_folders_sep, mod=mod)
		self.mod = mod
	

	def get_train_test_rr_fold(self, all_x, all_y, index):
		X_train, y_train = [], []
		X_test, y_test = all_x[index], all_y[index]
		for i in range(len(all_x)):
			if i != index:
				X_train += all_x[i]
				y_train += all_y[i]
		return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


	def cross_validate(self):
		results = {index: {} for index in range(self.mod)}
		for index in range(self.mod):
			X_train, X_test, y_train, y_test = self.get_train_test_rr_fold(self.all_x, self.all_y, index)

			curr_clf = self.clf
			curr_clf.fit(X_train, y_train)
			predicted = curr_clf.predict(X_test)
			results[index]['predicted'] = predicted
			results[index]['true_values'] = list(y_test)
			results[index]['score'] = np.mean(predicted==y_test)

		return results
	