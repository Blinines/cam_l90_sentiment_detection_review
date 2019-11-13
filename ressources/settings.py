# -*- coding: utf-8 -*-
PATH_NEG = "data/NEG/"
PATH_POS = "data/POS/"

PATH_NEG_TAG = "data-tagged/NEG/"
PATH_POS_TAG = "data-tagged/POS/"

TRAIN_TEST_SEP_VALUE = 900

FREQ_CUTOFF_UNIGRAM = 4
FREQ_CUTOFF_BIGRAM = 7
FREQ_CUTOFF = {1: FREQ_CUTOFF_UNIGRAM, 2: FREQ_CUTOFF_BIGRAM}

PATH_PROJECT = None

# For NB main_nb.py
TYPE_NB = ['unigram', 'bigram', 'joint']
SMOOTHING_NB = [0, 1]
FEAT_TYPE = ['freq', 'pres']

try:
    from private.private import *
except:
    pass