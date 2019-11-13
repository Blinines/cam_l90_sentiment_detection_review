# -*- coding: utf-8 -*-
from math import log
from random import uniform


def create_freq_bow(bow, nb_word, smoothing, distinct_w_count):
	""" Transforms count bow to frequency bow """
	freq_bow = {}
	for word in bow.keys():
		freq_bow[word] = (bow[word] + smoothing)/float(nb_word + smoothing * distinct_w_count)
	freq_bow[0] = smoothing/float(nb_word + smoothing * distinct_w_count)
	return freq_bow


# Applying Bayes rule
def calculate_proba_nb(feat, freq_bow, n_class, n, feat_type):
	res = 0
	if feat_type == 'pres':  # Accounting only for presence, count is not important
		feat = list(dict.fromkeys(feat))
	for f in feat:  # iterating through all words of document
		if f.lower() in freq_bow.keys():  # word was seen during training
			res += log(freq_bow[f.lower()])
		else:  # word unseen during training => algorithm stops if no smoothing
			if freq_bow[0] == 0:
				return float('-inf')
			else:
				res += log(freq_bow[0])
	res += log(float(n_class)/n)
	return res

def predict_naive_bayes(feat, freq_bow, n_neg, n_pos, n, feat_type):
	# return (a,b)
	# a : prediction, b : 1 if predicted with random choice, 0 else

	proba_neg = calculate_proba_nb(feat, freq_bow['NEG'], n_neg, n, feat_type)	
	proba_pos = calculate_proba_nb(feat, freq_bow['POS'], n_pos, n, feat_type)

	if proba_pos > proba_neg:
		return (1, False)
	elif proba_neg > proba_pos:
		return (0, False)
	else: # presumably both equal to minus infinity => random choice
		random_nb = uniform(0,1)
		predicted = 1 if random_nb <= 0.5 else 0
		return (predicted, True)

