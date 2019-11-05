# -*- coding: utf-8 -*-
from os import listdir
from helpers_nb import create_bow

from settings import PATH_PROJECT, PATH_NEG_TAG, PATH_POS_TAG, \
                     FREQ_CUTOFF_UNIGRAM, FREQ_CUTOFF_BIGRAM

all_files_name = []
for type_review in [PATH_NEG_TAG, PATH_POS_TAG]:
    for file_name in listdir(type_review):
        all_files_name.append(PATH_PROJECT+type_review+file_name)

# No feature frequency cutoff
bow_unigram_no_cutoff, u1 = create_bow(files_list=all_files_name, freq_cutoff=0, n=1)
bow_bigram_no_cutoff, b1 = create_bow(files_list=all_files_name, freq_cutoff=0, n=2)

print("Features without cutoff")
print("Nb of features - unigram : {0}".format(len(bow_unigram_no_cutoff.keys())))
print("Nb of features - bigram : {0}".format(len(bow_bigram_no_cutoff.keys())))
print("")

# Feature frequency cutoff
bow_unigram_cutoff, u2 = create_bow(files_list=all_files_name, freq_cutoff=FREQ_CUTOFF_UNIGRAM, n=1)
bow_bigram_cutoff, b2 = create_bow(files_list=all_files_name, freq_cutoff=FREQ_CUTOFF_BIGRAM, n=2)

print("Features with cutoff")
print("Nb of features - unigram : {0}".format(len(bow_unigram_cutoff.keys())))
print("Nb of features - bigram : {0}".format(len(bow_bigram_cutoff.keys())))
print("")
