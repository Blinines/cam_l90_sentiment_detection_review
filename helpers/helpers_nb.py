# -*- coding: utf-8 -*-
import codecs
from collections import Counter
from math import log
from random import uniform


# Creating features from files
def create_feat_no_s(file_path):
	# From one file returning all words 
	feat_no_s = []
	try:
		f = open(file_path, 'r')
		for word_info in f:
			word_info_l = word_info.split()
			if len(word_info_l) > 0:
				feat_no_s.append(word_info_l[0])
		return feat_no_s
	except:
		f = open(file_path, 'r', encoding='utf8')
		for word_info in f:
			word_info_l = word_info.split()
			if len(word_info_l) > 0:
				feat_no_s.append(word_info_l[0])
		return feat_no_s
	
	


def create_feat_n_gram(feat_no_s, n):
	# Creating features for n gram, incorporating sentence markers
	if len(feat_no_s) < n:
		return []
	feat_n_gram = [feat_no_s[:n]]
	for feat in feat_no_s[n:]:
		feat_n_gram.append(feat_n_gram[-1][-n+1:] + [feat])
	feat_n_gram = [' '.join(l) for l in feat_n_gram]
	return feat_n_gram


def create_feat(feat_no_s_1, file_path, n):
	# Creating features list for the file with n words
	if n == 1:
		return feat_no_s_1
	else:
		return create_feat_n_gram(feat_no_s=feat_no_s_1, n=n)


# Creating BoW (count and frequency)
def create_bow(files_list, freq_cutoff, n):
	# BoW, count
	word_count = Counter()
	word_total_count = 0
	for file_path in files_list:
		feat_no_s_1 = create_feat_no_s(file_path=file_path)
		feat = create_feat(feat_no_s_1=feat_no_s_1, file_path=file_path, n=n)
		for f in feat:
			word_count[f.lower()] += 1
			# word_count[f] += 1
			word_total_count += 1

	return {word: word_count[word] for word in word_count.keys() if word_count[word] >= freq_cutoff}, word_total_count


def create_freq_bow(bow, nb_word, smoothing, distinct_w_count):
	# Bow, frequency
	freq_bow = {}
	for word in bow.keys():
		freq_bow[word] = (bow[word] + smoothing)/float(nb_word + smoothing * distinct_w_count)
	freq_bow[0] = smoothing/float(nb_word + smoothing * distinct_w_count)
	return freq_bow


# Applying Bayes rule
def calculate_proba_nb(feat, freq_bow, n_class, n):
	res = 0
	for f in feat: # iterating through all words of document
		# if f in freq_bow.keys(): # word was seen during training
		if f.lower() in freq_bow.keys(): # word was seen during training
			res += log(freq_bow[f.lower()])
		else: # word unseen during training => algorithm stops
			if freq_bow[0] == 0:
				return float('-inf')
			else:
				res += log(freq_bow[0])
	res += log(float(n_class)/n)
	return res

def predict_naive_bayes(feat, freq_bow, n_neg, n_pos, n):
	# return (a,b)
	# a : prediction, b : 1 if predicted with random choice, 0 else

	proba_neg = calculate_proba_nb(feat, freq_bow['NEG'], n_neg, n)	
	proba_pos = calculate_proba_nb(feat, freq_bow['POS'], n_pos, n)

	if proba_pos > proba_neg:
		return (1, False)
	elif proba_neg > proba_pos:
		return (0, False)
	else: # presumably both equal to minus infinity => random choice
		random_nb = uniform(0,1)
		predicted = 1 if random_nb <= 0.5 else 0
		return (predicted, True)


# file_path_1 = 'C:/Users/Public/Documents/l90/data-tagged/POS/cv622_8147.tag'
# file_path_2 = 'C:/Users/Public/Documents/l90/data-tagged/NEG/cv007_4992.tag'
# test_with_s_1 = create_feat_with_s(file_path=file_path_1)
# test_no_s_1 = create_feat_no_s(feat_with_s=test_with_s_1)
# print(test_no_s_1)
# test_with_s_2 = create_feat_with_s(file_path=file_path_2)
# test_no_s_2 = create_feat_no_s(feat_with_s=test_with_s_2)
#print(test_no_s)
# test_bigram = create_feat_n_gram(feat_with_s=test_with_s, n=2)
# a, b = create_bow(files_list=[file_path_1, file_path_2], freq_cutoff=0, n=1) 
# print(a)
# print(b)
# print(len(test_no_s_1))
# print(len(test_no_s_2))