# -*- coding: utf-8 -*-
from random import uniform
from math import log

from helpers_cv import sep_train_test
from helpers_nb import create_bow_unigram, create_bow_bigram, create_freq_bow

from settings import *

class NaiveBayes:
	def __init__(self, type):
		# type in ['unigram', 'bigram', 'joint']
		self.type = type
	
	def fit(self, X_train, freq_cutoff):
		# X_train = {'NEG': TRAIN_FILE_NEG, 'POS': TRAIN_FILE_POS}
		# freq_cutoff = {'unigram': 1, 'bigram': 4} for example
		self.n_neg = len(X_train['NEG'])
		self.n_pos = len(X_train['POS'])

		if self.type == 'unigram':
			bow_neg_u, nb_word_neg_u = create_bow_unigram(X_train['NEG'], freq_cutoff['unigram'])
			bow_pos_u, nb_word_pos_u = create_bow_unigram(X_train['POS'], freq_cutoff['unigram'])

			self.freq_bow_u = {'NEG': create_freq_bow(bow_neg_u, nb_word_neg_u), 
						  'POS': create_freq_bow(bow_pos_u, nb_word_pos_u)}
		
		else:
			pass
	
	def predict(self, X_test):
		# X_test = {'NEG': TEST_FILE_NEG, 'POS': TEST_FILE_POS}
		return


def calculate_proba_nb(feat, freq_bow, n_class, n):
	res = 0
	for f in feat: # iterating through all words of document
		if f in freq_bow.keys(): # word was seen during training
			res += log(freq_bow[f])
		else: # word unseen during training => algorithm stops
			return float('-inf')
	res += log(float(n_class)/n)
	return res

def predict_naive_bayes(file_path, freq_bow, n_neg, n_pos):
	n = n_neg + n_pos

	proba_neg = calculate_proba_nb(file_path, freq_bow['NEG'], n_neg, n)	
	proba_pos = calculate_proba_nb(file_path, freq_bow['POS'], n_pos, n)

	if proba_pos > proba_neg:
		return 1
	elif proba_neg < proba_pos:
		return 0
	else: # presumably both equal to minus infinity => random choice
		random_nb = uniform(0,1)
		return 1 if random_nb <= 0.5 else 0
	

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
		predicted = predict_naive_bayes(test_file_neg_path, FREQ_BOW_U, N_NEG, N_POS)
		if predicted == 0:
			FP_TN_TABLE[1][1] += 1
		else:
			FP_TN_TABLE[0][1] += 1

	for test_file_pos_path in TEST_FILE_POS:
		predicted = predict_naive_bayes(test_file_pos_path, FREQ_BOW_U, N_NEG, N_POS)
		if predicted == 0:
			FP_TN_TABLE[1][0] += 1
		else:
			FP_TN_TABLE[0][0] += 1

	print('Unigram only')
	print(FP_TN_TABLE)
	print('')
