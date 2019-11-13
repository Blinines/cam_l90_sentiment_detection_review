# -*- coding: utf-8 -*-
from collections import Counter

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
