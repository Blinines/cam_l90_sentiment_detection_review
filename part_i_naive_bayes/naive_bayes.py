# -*- coding: utf-8 -*-
from helpers.helpers_cv import sep_train_test
from helpers.helpers_nb import create_bow, create_freq_bow, create_feat, \
					   create_feat_no_s, create_feat_n_gram, predict_naive_bayes
from helpers.helpers_gen import sign_test, get_accuracy
from ressources.settings import *


class NaiveBayes:
	def __init__(self, t, smoothing, freq_cutoff):
		# type in ['unigram', 'bigram', 'joint']
		# freq_cutoff = {'unigram': 1, 'bigram': 4} for example
		self.type_to_calc = {'unigram': [1], 'bigram': [2], 'joint': [1, 2]}
		self.type = t
		self.smoothing = smoothing
		self.freq_cutoff = freq_cutoff
	

	def is_valid_nb(self):
		if self.type not in ['unigram', 'bigram', 'joint']:
			return False
		return True
	

	def fit(self, X_train):
		# X_train = {'NEG': TRAIN_FILE_NEG, 'POS': TRAIN_FILE_POS}
		self.n_neg = len(X_train['NEG'])
		self.n_pos = len(X_train['POS'])
		self.n = self.n_neg + self.n_pos

		bow_neg, nb_word_neg = {}, 0
		bow_pos, nb_word_pos = {}, 0

		for nb in self.type_to_calc[self.type]:
			curr_bow_neg, curr_nb_word_neg = create_bow(X_train['NEG'], self.freq_cutoff[nb], nb)
			curr_bow_pos, curr_nb_word_pos = create_bow(X_train['POS'], self.freq_cutoff[nb], nb)

			bow_neg.update(curr_bow_neg)
			nb_word_neg += curr_nb_word_neg
			bow_pos.update(curr_bow_pos)
			nb_word_pos += curr_nb_word_pos

		self.distinct_w_count = len(set(bow_neg.keys()).union(set(bow_pos.keys())))
		self.freq_bow = {'NEG': create_freq_bow(bow_neg, nb_word_neg, self.smoothing, self.distinct_w_count), 
						 'POS': create_freq_bow(bow_pos, nb_word_pos, self.smoothing, self.distinct_w_count)}
	

	def predict(self, X_test):
		# X_test = {'NEG': TEST_FILE_NEG, 'POS': TEST_FILE_POS}
		y = {'NEG': [], 'POS': [], 'random': 0}
		for val in ['NEG', 'POS']:
			for file_path in X_test[val]:
				feat_no_s_1 = create_feat_no_s(file_path=file_path)

				feat = []
				for nb in self.type_to_calc[self.type]:
					feat += create_feat(feat_no_s_1=feat_no_s_1, file_path=file_path, n=nb)
				
				predicted_file, randomed = predict_naive_bayes(feat=feat, freq_bow=self.freq_bow, 
													 		   n_neg=self.n_neg, n_pos=self.n_pos, n=self.n)
				y[val].append(predicted_file)
				y['random'] += randomed
		return y


if False:
	TRAIN_FILE_NEG, TEST_FILE_NEG = sep_train_test(PATH_NEG_TAG, TRAIN_TEST_SEP_VALUE)
	TRAIN_FILE_POS, TEST_FILE_POS = sep_train_test(PATH_POS_TAG, TRAIN_TEST_SEP_VALUE)
	X_train = {'NEG': TRAIN_FILE_NEG, 'POS': TRAIN_FILE_POS}
	FREQ_CUTOFF = {1: FREQ_CUTOFF_UNIGRAM, 2: FREQ_CUTOFF_BIGRAM} 
	X_test = {'NEG': TEST_FILE_NEG, 'POS': TEST_FILE_POS}
	

	 #for t in ['unigram', 'bigram', 'joint']:
	for t in ['unigram']:
		print('Type: {0}'.format(t))
		
		clf_1 = NaiveBayes(t=t, smoothing=1, freq_cutoff=FREQ_CUTOFF)
		clf_1.fit(X_train=X_train)
		print("Distinct features: {0}".format(clf_1.distinct_w_count))
		# print(clf_1.freq_bow)
		y_1 = clf_1.predict(X_test=X_test)
		print("Smoothing {0}: accuracy is {1}".format(1, get_accuracy(y_1)))

		# clf_2 = NaiveBayes(t=t, smoothing=0, freq_cutoff=FREQ_CUTOFF)
		# clf_2.fit(X_train=X_train)
		# y_2 = clf_2.predict(X_test=X_test)
		# print("Smoothing {0}: accuracy is {1}".format(0, get_accuracy(y_2)))

		# print(sign_test(y_1, y_2))
		# print('')


