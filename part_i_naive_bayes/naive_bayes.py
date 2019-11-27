# -*- coding: utf-8 -*-
import numpy as np
from helpers.helpers_nb import create_freq_bow, predict_naive_bayes
from helpers.helpers_bow import create_feat_no_s, create_feat_n_gram, create_feat, create_bow
from helpers.helpers_gen import sign_test
from ressources.settings import FREQ_CUTOFF


class NaiveBayes:
	def __init__(self, t, smoothing=1, freq_cutoff=FREQ_CUTOFF, feat_type='freq'):
		# type in ['unigram', 'bigram', 'joint']
		# freq_cutoff = {'unigram': 1, 'bigram': 4} for example
		self.type_to_calc = {'unigram': [1], 'bigram': [2], 'joint': [1, 2]}
		self.type = t
		self.smoothing = smoothing
		self.freq_cutoff = freq_cutoff
		self.feat_type = feat_type
		self.randomed = None
	

	def is_valid_nb(self):
		if self.type not in ['unigram', 'bigram', 'joint']:
			return False
		return True
	

	def fit(self, X_train, y_train):
		""" Creates frequency BoW for the NB model """
		self.n = len(y_train)
		self.n_pos = np.sum(y_train)
		self.n_neg = self.n - self.n_pos

		bow_neg, nb_word_neg = {}, 0
		bow_pos, nb_word_pos = {}, 0

		# Separating negative and positive documents for BoW
		X_train_neg, X_train_pos = [], []
		for index, path in enumerate(X_train):
			if y_train[index] == 0:
				X_train_neg.append(path)
			else:
				X_train_pos.append(path)

		# for each n-gram taken into account, get the corresponding BoW and updates the final one
		for nb in self.type_to_calc[self.type]:
			curr_bow_neg, curr_nb_word_neg = create_bow(X_train_neg, self.freq_cutoff[nb], nb)
			curr_bow_pos, curr_nb_word_pos = create_bow(X_train_pos, self.freq_cutoff[nb], nb)

			bow_neg.update(curr_bow_neg)
			nb_word_neg += curr_nb_word_neg
			bow_pos.update(curr_bow_pos)
			nb_word_pos += curr_nb_word_pos

		self.distinct_w_count = len(set(bow_neg.keys()).union(set(bow_pos.keys())))
		self.freq_bow = {'NEG': create_freq_bow(bow_neg, nb_word_neg, self.smoothing, self.distinct_w_count), 
						 'POS': create_freq_bow(bow_pos, nb_word_pos, self.smoothing, self.distinct_w_count)}
	

	def predict(self, X_test):
		y = []
		self.randomed = []
		
		for index, file_path in enumerate(X_test):
			feat_no_s_1 = create_feat_no_s(file_path=file_path)

			feat = []
			for nb in self.type_to_calc[self.type]:  # creates features of doc to predict
				feat += create_feat(feat_no_s_1=feat_no_s_1, n=nb)
			
			predicted_file, randomed = predict_naive_bayes(feat=feat, freq_bow=self.freq_bow, 
														   n_neg=self.n_neg, n_pos=self.n_pos, n=self.n,
														   feat_type=self.feat_type)
			y.append(predicted_file)
			self.randomed.append((index, randomed))
		return y
	
	 
	def get_randomed(self):
		return self.randomed
