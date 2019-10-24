# -*- coding: utf-8 -*-
import codecs
from collections import Counter

def create_bow_unigram(files_list, freq_cutoff):
	word_count = Counter()
	word_total_count = 0
	for file in files_list:
		with codecs.open(file, 'r') as f:
			for word_info in f:
				try:
					word = word_info.split()[0]
					word_count[word] += 1
					word_total_count += 1
				except:
					pass

	return {word: word_count[word] for word in word_count.keys() if word_count[word] >= freq_cutoff}, word_total_count


def create_bow_bigram(files_list, freq_cutoff):
	word_count = Counter()
	word_total_count = 0
	for file in files_list:
		with codecs.open(file, 'r') as f:
			nb_word = len(f)
			for i in range(nb_word):
				if i < nb_word:
					try:
						word_1, word_2 = f[i].split()[0], f[i+1].split()[0]
						word_count[' '.join([word_1, word_2])] += 1
						word_total_count += 1
					except:
						pass

	return {word: word_count[word] for word in word_count.keys() if word_count[word] >= freq_cutoff}, word_total_count


def create_freq_bow(bow, nb_word):
	freq_bow = {}
	for word in bow.keys():
		freq_bow[word] = bow[word]/float(nb_word)
	return freq_bow