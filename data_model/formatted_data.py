# -*- coding: utf-8 -*-
from helpers.helpers_cv import folder_round_robin
from ressources.settings import PATH_NEG_TAG, PATH_POS_TAG


# Stratified RR folders for POS and NEG reviews
fold_rr_neg = folder_round_robin(files_path=PATH_NEG_TAG, mod=10)
X_train_neg = []
for i in range(2, 10):
    X_train_neg += fold_rr_neg[i]

fold_rr_pos = folder_round_robin(files_path=PATH_POS_TAG, mod=10)
X_train_pos = []
for i in range(2, 10):
    X_train_pos += fold_rr_pos[i]


# Training data
X_train = X_train_neg + X_train_pos
y_train = [0]*len(X_train_neg) + [1]*len(X_train_pos)

# Dev data
X_dev_neg, X_dev_pos = fold_rr_neg[1], fold_rr_pos[1]
X_dev = X_dev_neg + X_dev_pos
y_dev = [0]*len(X_dev_neg) + [1]*len(X_dev_pos) 


# Test data
X_test_blind_neg, X_test_blind_pos = fold_rr_neg[0], fold_rr_pos[0]
X_test_blind = X_test_blind_neg + X_test_blind_pos
y_test_blind = [0]*len(X_test_blind_neg) + [1]*len(X_test_blind_pos)
