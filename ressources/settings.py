# -*- coding: utf-8 -*-
from os import listdir

# Accessing datasets (in practice only using tagged versions)
PATH_NEG = "data/NEG/"
PATH_POS = "data/POS/"
PATH_NEG_TAG = "data-tagged/NEG/"
PATH_POS_TAG = "data-tagged/POS/"

# Cutoffs 
FREQ_CUTOFF_UNIGRAM = 4
FREQ_CUTOFF_BIGRAM = 7
FREQ_CUTOFF = {1: FREQ_CUTOFF_UNIGRAM, 2: FREQ_CUTOFF_BIGRAM}


# General
TRAIN_TEST_SEP_VALUE = 900
PATH_PROJECT = None


# For NB main_nb.py
TYPE_NB = ['unigram', 'bigram', 'joint']
SMOOTHING_NB = [0, 1]
FEAT_TYPE = ['freq']


try:
    from private.private import *
except:
    pass


# SVM - training doc2vec models
svm_train_folder_dir = ['aclImdb/test/neg/', 'aclImdb/test/pos/', 
                        'aclImdb/train/neg/', 'aclImdb/train/pos/', 'aclImdb/train/unsup/']
SVM_TRAIN_FILE_DIR = []
for folder in svm_train_folder_dir:
    SVM_TRAIN_FILE_DIR += [PATH_PROJECT + folder + file_name \
                          for file_name in listdir(PATH_PROJECT + folder)]
