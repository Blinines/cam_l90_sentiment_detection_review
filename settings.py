# -*- coding: utf-8 -*-
PATH_NEG = "data/NEG/"
PATH_POS = "data/POS/"

PATH_NEG_TAG = "data-tagged/NEG/"
PATH_POS_TAG = "data-tagged/POS/"

TRAIN_TEST_SEP_VALUE = 900
FREQ_CUTOFF_UNIGRAM = 4
FREQ_CUTOFF_BIGRAM = 7

PATH_PROJECT = None

try:
    from private import *
except:
    pass