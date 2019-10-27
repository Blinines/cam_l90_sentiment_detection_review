# -*- coding: utf-8 -*-
from helpers_cv import sep_train_test
from helpers_nb import create_bow, create_freq_bow, create_feat, \
					   create_feat_with_s, create_feat_n_gram, predict_naive_bayes
from helpers import sign_test, get_accuracy
from settings import *


class NaiveBayes:
	def __init__(self, t, smoothing):
		# type in ['unigram', 'bigram', 'joint']
		self.type_to_calc = {'unigram': [1], 'bigram': [2], 'joint': [1, 2]}
		self.type = t
		self.smoothing = smoothing
	
	def is_valid_nb(self):
		if self.type not in ['unigram', 'bigram', 'joint']:
			return False
		return True
	
	def fit(self, X_train, freq_cutoff):
		# X_train = {'NEG': TRAIN_FILE_NEG, 'POS': TRAIN_FILE_POS}
		# freq_cutoff = {'unigram': 1, 'bigram': 4} for example
		self.n_neg = len(X_train['NEG'])
		self.n_pos = len(X_train['POS'])
		self.n = self.n_neg + self.n_pos

		bow_neg, nb_word_neg = {}, 0
		bow_pos, nb_word_pos = {}, 0

		for n in self.type_to_calc[self.type]:
			curr_bow_neg, curr_nb_word_neg = create_bow(X_train['NEG'], freq_cutoff[n], n)
			curr_bow_pos, curr_nb_word_pos = create_bow(X_train['POS'], freq_cutoff[n], n)

			bow_neg.update(curr_bow_neg)
			nb_word_neg += curr_nb_word_neg
			bow_pos.update(curr_bow_pos)
			nb_word_pos += curr_nb_word_pos

		self.freq_bow = {'NEG': create_freq_bow(bow_neg, nb_word_neg, self.smoothing), 
						 'POS': create_freq_bow(bow_pos, nb_word_pos, self.smoothing)}
	
	def predict(self, X_test):
		# X_test = {'NEG': TEST_FILE_NEG, 'POS': TEST_FILE_POS}
		y = {'NEG': [], 'POS': []}
		for val in ['NEG', 'POS']:
			for file_path in X_test[val]:
				feat_with_s_u = create_feat_with_s(file_path=file_path)

				feat = []
				for nb in self.type_to_calc[self.type]:
					feat += create_feat(feat_with_s=feat_with_s_u, file_path=file_path, n=nb)
				
				predicted_file = predict_naive_bayes(feat=feat, freq_bow=self.freq_bow, 
													 n_neg=self.n_neg, n_pos=self.n_pos, n=self.n)
				y[val].append(predicted_file)
		return y

TRAIN_FILE_NEG, TEST_FILE_NEG = sep_train_test(PATH_NEG_TAG, TRAIN_TEST_SEP_VALUE)
TRAIN_FILE_POS, TEST_FILE_POS = sep_train_test(PATH_POS_TAG, TRAIN_TEST_SEP_VALUE)
X_train = {'NEG': TRAIN_FILE_NEG, 'POS': TRAIN_FILE_POS}
FREQ_CUTOFF = {1: FREQ_CUTOFF_UNIGRAM, 2: FREQ_CUTOFF_BIGRAM} 
X_test = {'NEG': TEST_FILE_NEG, 'POS': TEST_FILE_POS}

 


for t in ['unigram', 'bigram', 'joint']:
# for t in ['unigram']:
	print('Type: {0}'.format(t))
	 
	clf_1 = NaiveBayes(t=t, smoothing=1)
	clf_1.fit(X_train=X_train, freq_cutoff=FREQ_CUTOFF)
	y_1 = clf_1.predict(X_test=X_test)
	print("Smoothing {0}: accuracy is {1}".format(1, get_accuracy(y_1)))

	clf_2 = NaiveBayes(t=t, smoothing=0)
	clf_2.fit(X_train=X_train, freq_cutoff=FREQ_CUTOFF)
	y_2 = clf_2.predict(X_test=X_test)
	print("Smoothing {0}: accuracy is {1}".format(0, get_accuracy(y_2)))

	print(sign_test(y_1, y_2))
	print('')
	