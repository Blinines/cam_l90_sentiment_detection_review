import glob
from math import log
from collections import Counter
from settings import PATH_NEG, PATH_POS

def sep_train_test(file_path, top_value_train):
	train_files = []
	test_files = []
	
	for index, file in enumerate(glob.glob(file_path)):
		if index < top_value_train:
			train_files.append(file)
		else:
			test_files.append(file)
	return train_files, test_files


def count_words(files_list, freq_cutoff):
	word_count = Counter()
	word_total_count = 0
	for file in files_list:
		f = open(file, 'r')
		for word_info in f:
			try:
				word = word_info.split()[0]
				word_count[word] += 1
				word_total_count += 1
			except:
				pass

	return {word: word_count[word] for word in word_count.keys() if word_count[word] >= freq_cutoff}, word_total_count


def create_freq_bow(bow, nb_word):
	freq_bow = {}
	for word in bow.keys():
		freq_bow[word] = bow[word]/float(nb_word)
	return freq_bow


def calculate_proba(file_path, freq_bow, n_class, n):
	res = 0
	f = open(file_path, 'r')
	for word_info in f:
		if len(word_info.split()) > 0:
			word = word_info.split()[0]
			if word in freq_bow.keys():
				res += log(freq_bow[word])
			else:
				res += log(0.0000000001)
	res += log(float(n_class)/n)
	return res

def predict(file_path, freq_bow, n_neg, n_pos):
	n = n_neg + n_pos

	proba_neg = calculate_proba(file_path, freq_bow['NEG'], n_neg, n)	
	proba_pos = calculate_proba(file_path, freq_bow['POS'], n_pos, n)

	return 1 if proba_pos > proba_neg else 0
	
	

TRAIN_FILE_NEG, TEST_FILE_NEG = sep_train_test(PATH_NEG, 900)
TRAIN_FILE_POS, TEST_FILE_POS = sep_train_test(PATH_POS, 900)

FREQ_CUTOFF = 4
BOW_NEG, NB_WORD_NEG = count_words(TRAIN_FILE_NEG, FREQ_CUTOFF)
BOW_POS, NB_WORD_POS = count_words(TRAIN_FILE_POS, FREQ_CUTOFF)
FREQ_BOW = {'NEG': create_freq_bow(BOW_NEG, NB_WORD_NEG), 
	'POS': create_freq_bow(BOW_POS, NB_WORD_POS)}

N_NEG = len(TRAIN_FILE_NEG)
N_POS = len(TRAIN_FILE_POS)
N = N_NEG + N_POS

FP_TN_TABLE = [[0, 0], [0, 0]]

for test_file_neg_path in TEST_FILE_NEG:
	predicted = predict(test_file_neg_path, FREQ_BOW, N_NEG, N_POS)
	if predicted == 0:
		FP_TN_TABLE[1][1] += 1
	else:
		FP_TN_TABLE[0][1] += 1

for test_file_pos_path in TEST_FILE_POS:
	predicted = predict(test_file_pos_path, FREQ_BOW, N_NEG, N_POS)
	if predicted == 0:
		FP_TN_TABLE[1][0] += 1
	else:
		FP_TN_TABLE[0][0] += 1

print(FP_TN_TABLE)


