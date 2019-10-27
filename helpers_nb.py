# -*- coding: utf-8 -*-
import codecs
from collections import Counter
from math import log
from random import uniform


# Creating features from files
def create_feat_with_s(file_path):
	# From one file returning all words including beginning and end of sentence
	# Beginning => <s>, End => </s>
	features_with_s = ['<s>']
	f = open(file_path, 'r')
	for word_info in f:
		word_info_l = word_info.split()
		if len(word_info_l) > 0:
			features_with_s.append(word_info_l[0])
		else:
			features_with_s.append('</s>')
			features_with_s.append('<s>')
	return features_with_s


def create_feat_no_s(feat_with_s):
	# Removing sentence markers from the features
	feat_no_s = []
	for feat in feat_with_s:
		if feat not in ['<s>', '</s>']:
			feat_no_s.append(feat)
	return feat_no_s


def create_feat_n_gram(feat_with_s, n):
	# Creating features for n gram, incorporating sentence markers
	if len(feat_with_s) < n:
		return []
	feat_n_gram = [feat_with_s[:n]]
	for feat in feat_with_s[n:]:
		feat_n_gram.append(feat_n_gram[-1][-n+1:] + [feat])
	feat_n_gram = [' '.join(l) for l in feat_n_gram]
	return feat_n_gram


def create_feat(feat_with_s, file_path, n):
	# Creating features list for the file with n words
	if n == 1:
		return create_feat_no_s(feat_with_s=feat_with_s)
	else:
		return create_feat_n_gram(feat_with_s=feat_with_s, n=n)


# Creating BoW (count and frequency)
def create_bow(files_list, freq_cutoff, n):
	# BoW, count
	word_count = Counter()
	word_total_count = 0
	for file_path in files_list:
		if n == 1:
			feat = create_feat_no_s(feat_with_s=create_feat_with_s(file_path))
		else:
			feat_with_s = create_feat_with_s(file_path=file_path)
			feat = create_feat_n_gram(feat_with_s=feat_with_s, n=n)
		for f in feat:
			word_count[f] += 1
			word_total_count += 1

	return {word: word_count[word] for word in word_count.keys() if word_count[word] >= freq_cutoff}, word_total_count


def create_freq_bow(bow, nb_word, smoothing):
	# Bow, frequency
	freq_bow = {}
	for word in bow.keys():
		freq_bow[word] = (bow[word] + smoothing)/float(nb_word * (smoothing + 1))
	freq_bow[0] = smoothing/(float(nb_word * (smoothing + 1)))
	return freq_bow


# Applying Bayes rule
def calculate_proba_nb(feat, freq_bow, n_class, n):
	res = 0
	for f in feat: # iterating through all words of document
		if f in freq_bow.keys(): # word was seen during training
			res += log(freq_bow[f])
		else: # word unseen during training => algorithm stops
			if freq_bow[0] == 0:
				return float('-inf')
			else:
				res += log(freq_bow[0])
	res += log(float(n_class)/n)
	return res

def predict_naive_bayes(feat, freq_bow, n_neg, n_pos, n):

	proba_neg = calculate_proba_nb(feat, freq_bow['NEG'], n_neg, n)	
	proba_pos = calculate_proba_nb(feat, freq_bow['POS'], n_pos, n)

	if proba_pos > proba_neg:
		return 1
	elif proba_neg < proba_pos:
		return 0
	else: # presumably both equal to minus infinity => random choice
		random_nb = uniform(0,1)
		return 1 if random_nb <= 0.5 else 0


# file_path = 'C:/Users/Public/Documents/l90/data-tagged/NEG/cv403_6721.tag'
# test_with_s = create_feat_with_s(file_path=file_path)
# test_no_s = create_feat_no_s(feat_with_s=test_with_s)
# test_bigram = create_feat_n_gram(feat_with_s=test_with_s, n=2)
# a, b = create_bow(files_list=[file_path], freq_cutoff=0, n=1) 
# print(a)