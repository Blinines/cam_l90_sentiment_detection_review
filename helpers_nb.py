# -*- coding: utf-8 -*-
import codecs
from collections import Counter
from math import log
from random import uniform

def create_bow_unigram(files_list, freq_cutoff):
	word_count = Counter()
	word_total_count = 0
	for file in files_list:
		with codecs.open(file, 'r') as f:
			for word_info in f:
				if len(word_info.split()) > 0:
					word = word_info.split()[0]
					word_count[word] += 1
					word_total_count += 1

	return {word: word_count[word] for word in word_count.keys() if word_count[word] >= freq_cutoff}, word_total_count


def create_bow_bigram(files_list, freq_cutoff):
	word_count = Counter()
	word_total_count = 0

	for file in files_list:
		with codecs.open(file, 'r') as f:
			tags = f.read().splitlines()
			tags = [tag.replace('\t', ' ') for tag in tags]
			nb_word = len(tags)

			# Treating first word
			first_word_file = tags[0].split()[0]
			word_count[' '.join(['<s>', first_word_file])] += 1
			word_total_count += 1
			first_word, second_word = first_word_file, None

			for i in range(1, nb_word):
				if first_word == '</s>':
					second_word = '<s>'
				else:
					second_word_info = tags[i].split()
					second_word = second_word_info[0] if len(second_word_info) > 0 else '</s>'

				word_count[' '.join([first_word, second_word])] += 1
				word_total_count += 1
				first_word = second_word
				

	return {word: word_count[word] for word in word_count.keys() if word_count[word] >= freq_cutoff}, word_total_count


def create_freq_bow(bow, nb_word):
	freq_bow = {}
	for word in bow.keys():
		freq_bow[word] = bow[word]/float(nb_word)
	return freq_bow


def create_feat_with_s(file_path):
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
	feat_no_s = []
	for feat in feat_with_s:
		if feat not in ['<s>', '</s>']:
			feat_no_s.append(feat)
	return feat_no_s


def create_feat_n_gram(feat_with_s, n):
	if len(feat_with_s) < n:
		return []
	feat_n_gram = [feat_with_s[:n]]
	for feat in feat_with_s[n:]:
		feat_n_gram.append(feat_n_gram[-1][-n+1:] + [feat])
	feat_n_gram = [' '.join(l) for l in feat_n_gram]
	return feat_n_gram


def calculate_proba_nb(feat, freq_bow, n_class, n):
	res = 0
	for f in feat: # iterating through all words of document
		if f in freq_bow.keys(): # word was seen during training
			res += log(freq_bow[f])
		else: # word unseen during training => algorithm stops
			return float('-inf')
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
# test = create_feat_with_s(file_path=file_path)
# print(create_feat_n_gram(test, 3))
