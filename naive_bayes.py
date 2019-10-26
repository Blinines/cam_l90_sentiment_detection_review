# -*- coding: utf-8 -*-
from helpers_cv import sep_train_test
from helpers_nb import create_bow_unigram, create_bow_bigram, create_freq_bow, \
					   create_feat_no_s, create_feat_with_s, create_feat_n_gram, \
					   predict_naive_bayes

from settings import *

class NaiveBayes:
	def __init__(self, type):
		# type in ['unigram', 'bigram', 'joint']
		self.type = type
	
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

		if self.type == 'unigram':
			bow_neg, nb_word_neg = create_bow_unigram(X_train['NEG'], freq_cutoff['unigram'])
			bow_pos, nb_word_pos = create_bow_unigram(X_train['POS'], freq_cutoff['unigram'])

			self.freq_bow = {'NEG': create_freq_bow(bow_neg, nb_word_neg), 
						  	 'POS': create_freq_bow(bow_pos, nb_word_pos)}
		
		elif self.type == 'bigram':
			bow_neg, nb_word_neg = create_bow_bigram(X_train['NEG'], freq_cutoff['bigram'])
			bow_pos, nb_word_pos = create_bow_bigram(X_train['POS'], freq_cutoff['bigram'])

			self.freq_bow = {'NEG': create_freq_bow(bow_neg, nb_word_neg), 
						  	 'POS': create_freq_bow(bow_pos, nb_word_pos)}
		
		else:  #self.type = 'joint'
			bow_neg_u, nb_word_neg_u = create_bow_unigram(X_train['NEG'], freq_cutoff['unigram'])
			bow_pos_u, nb_word_pos_u = create_bow_unigram(X_train['POS'], freq_cutoff['unigram'])
			bow_neg_b, nb_word_neg_b = create_bow_bigram(X_train['NEG'], freq_cutoff['bigram'])
			bow_pos_b, nb_word_pos_b = create_bow_bigram(X_train['POS'], freq_cutoff['bigram'])

			bow_neg_u.update(bow_neg_b)
			nb_word_neg = nb_word_neg_u+nb_word_neg_b
			bow_pos_u.update(bow_pos_b)
			nb_word_pos = nb_word_pos_u+nb_word_pos_b

			self.freq_bow = {'NEG': create_freq_bow(bow_neg_u, nb_word_neg), 
						  	 'POS': create_freq_bow(bow_pos_u, nb_word_pos)}
	
	def predict(self, X_test):
		# X_test = {'NEG': TEST_FILE_NEG, 'POS': TEST_FILE_POS}
		y = {'NEG': [], 'POS': []}
		for val in ['NEG', 'POS']:
			for file_path in X_test[val]:
				feat_with_s_u = create_feat_with_s(file_path=file_path)

				if self.type == 'unigram':
					feat = create_feat_no_s(feat_with_s=feat_with_s_u)
				elif self.type == 'bigram':
					feat = create_feat_n_gram(feat_with_s=feat_with_s_u, n=2)
				else: #self.type = 'joint'
					feat = create_feat_no_s(feat_with_s=feat_with_s_u) + \
						   create_feat_n_gram(feat_with_s=feat_with_s_u, n=2)
				
				predicted_file = predict_naive_bayes(feat=feat, freq_bow=self.freq_bow, 
													 n_neg=self.n_neg, n_pos=self.n_pos, n=self.n)
				y[val].append(predicted_file)
		return y


if False:
	TRAIN_FILE_NEG, TEST_FILE_NEG = sep_train_test(PATH_NEG_TAG, TRAIN_TEST_SEP_VALUE)
	TRAIN_FILE_POS, TEST_FILE_POS = sep_train_test(PATH_POS_TAG, TRAIN_TEST_SEP_VALUE)

	N_NEG = len(TRAIN_FILE_NEG)
	N_POS = len(TRAIN_FILE_POS)

	BOW_NEG_U, NB_WORD_NEG_U = create_bow_unigram(TRAIN_FILE_NEG, FREQ_CUTOFF_UNIGRAM)
	BOW_POS_U, NB_WORD_POS_U = create_bow_unigram(TRAIN_FILE_POS, FREQ_CUTOFF_UNIGRAM)

	BOW_NEG_B, NB_WORD_NEG_B = create_bow_unigram(TRAIN_FILE_NEG, FREQ_CUTOFF_BIGRAM)
	BOW_POS_B, NB_WORD_POS_B = create_bow_unigram(TRAIN_FILE_POS, FREQ_CUTOFF_BIGRAM)

	# Using unigrams only
	PREDICT_U = True
	if PREDICT_U:	
		FREQ_BOW_U = {'NEG': create_freq_bow(BOW_NEG_U, NB_WORD_NEG_U), 
					'POS': create_freq_bow(BOW_POS_U, NB_WORD_POS_U)}
		FP_TN_TABLE = [[0, 0], [0, 0]]

		for test_file_neg_path in TEST_FILE_NEG:
			predicted = predict_naive_bayes(test_file_neg_path, FREQ_BOW_U, N_NEG, N_POS, N_NEG+N_POS)
			if predicted == 0:
				FP_TN_TABLE[1][1] += 1
			else:
				FP_TN_TABLE[0][1] += 1

		for test_file_pos_path in TEST_FILE_POS:
			predicted = predict_naive_bayes(test_file_pos_path, FREQ_BOW_U, N_NEG, N_POS, N_NEG+N_POS)
			if predicted == 0:
				FP_TN_TABLE[1][0] += 1
			else:
				FP_TN_TABLE[0][0] += 1

		print('Unigram only')
		print(FP_TN_TABLE)
		print('')


TRAIN_FILE_NEG, TEST_FILE_NEG = sep_train_test(PATH_NEG_TAG, TRAIN_TEST_SEP_VALUE)
TRAIN_FILE_POS, TEST_FILE_POS = sep_train_test(PATH_POS_TAG, TRAIN_TEST_SEP_VALUE)
X_train = {'NEG': TRAIN_FILE_NEG, 'POS': TRAIN_FILE_POS}
FREQ_CUTOFF = {'unigram': FREQ_CUTOFF_UNIGRAM, 'bigram': FREQ_CUTOFF_BIGRAM} 
X_test = {'NEG': TEST_FILE_NEG, 'POS': TEST_FILE_POS}
#NEG': [TEST_FILE_NEG[0]], 'POS': []}


def get_accuracy(y):
	count, tot = 0, 0
	for elt in y['NEG']:
		if elt == 0:
			count += 1
		tot += 1
	for elt in y['POS']:
		if elt == 1:
			count += 1
		tot += 1
	return float(count)/tot 


for t in ['unigram', 'bigram', 'joint']:
# for t in ['bigram']:
	clf = NaiveBayes(type=t)
	clf.fit(X_train=X_train, freq_cutoff=FREQ_CUTOFF)
	y = clf.predict(X_test=X_test)
	print("Type {0}: accuracy is {1}".format(t, get_accuracy(y)))